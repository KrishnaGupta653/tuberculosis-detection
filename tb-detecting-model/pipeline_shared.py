"""
================================================================================
SHARED FEATURE EXTRACTION PIPELINE
================================================================================
This module contains ALL shared functions used by individual model trainers.
Eliminates code duplication across SVM, RandomForest, and GradientBoosting.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import gabor
import pywt
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION (SHARED)
# ═══════════════════════════════════════════════════════════════════════════

class Config:
    """Central configuration - identical across all trainers"""
    
    DATA_DIR = './dataset'
    CATEGORIES = ['Normal', 'TB']
    IMG_SIZE = 224
    
    MODEL_DIR = './saved_models'
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4
    N_JOBS = 4
    BATCH_SIZE = 32
    USE_CUDA = torch.cuda.is_available()
    PIN_MEMORY = torch.cuda.is_available()
    
    ENTROPY_WINDOW = 15
    MORPHOLOGY_ITERATIONS = 3
    MIN_CONTOUR_AREA = 1000
    
    GLCM_DISTANCES = [1, 3, 5]
    GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    WAVELET_FAMILY = 'db4'
    WAVELET_LEVELS = 3
    FRACTAL_BOXES = [2, 4, 8, 16, 32]
    
    CROSS_VAL_FOLDS = 5
    TEST_SIZE = 0.2
    RANDOM_SEED = 42
    
    # NEW: Class weighting for imbalanced data
    CLASS_WEIGHT = 'balanced'  # or {0: 1, 1: 10}
    
    # NEW: Cost matrix
    SAMPLE_WEIGHT_TRAIN = True


# ═══════════════════════════════════════════════════════════════════════════
# SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════

class EntropyGuidedSegmentor:
    """Entropy-based lung segmentation"""
    
    def __init__(self):
        self.entropy_window = Config.ENTROPY_WINDOW
    
    def compute_local_entropy(self, image):
        from skimage.filters.rank import entropy as rank_entropy
        from skimage.morphology import disk
        img_uint8 = (image / image.max() * 255).astype(np.uint8)
        entr = rank_entropy(img_uint8, disk(self.entropy_window // 2))
        return entr
    
    def multi_scale_clahe(self, image):
        scales = [8, 16, 32]
        enhanced_stack = []
        for grid_size in scales:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(grid_size, grid_size))
            enhanced = clahe.apply(image)
            enhanced_stack.append(enhanced)
        weights = [0.2, 0.5, 0.3]
        fused = np.zeros_like(image, dtype=np.float32)
        for w, img in zip(weights, enhanced_stack):
            fused += w * img.astype(np.float32)
        return fused.astype(np.uint8)
    
    def attention_weighted_morph(self, binary_mask, attention_map):
        attention_norm = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        kernel_base = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        result = np.zeros_like(binary_mask)
        block_size = 32
        h, w = binary_mask.shape
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = binary_mask[i:i+block_size, j:j+block_size]
                attn_block = attention_map[i:i+block_size, j:j+block_size]
                mean_attn = np.mean(attn_block)
                iterations = int(1 + mean_attn * Config.MORPHOLOGY_ITERATIONS)
                processed_block = cv2.morphologyEx(block, cv2.MORPH_CLOSE,
                                                  kernel_base, iterations=iterations)
                result[i:i+block_size, j:j+block_size] = processed_block
        return result
    
    def segment_lung_roi(self, image_input):
        """Returns (segmented_img, final_mask, entropy_map)"""
        if isinstance(image_input, np.ndarray):
            img = image_input.copy()
        else:
            img = cv2.imread(image_input)
            if img is None:
                return None, None, None
        
        img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        enhanced = self.multi_scale_clahe(gray)
        entropy_map = self.compute_local_entropy(enhanced)
        
        sensitivity = 0.5
        threshold_val = np.mean(entropy_map) + sensitivity * np.std(entropy_map)
        _, binary_entropy = cv2.threshold(entropy_map.astype(np.uint8),
                                         int(threshold_val), 255, cv2.THRESH_BINARY)
        _, binary_otsu = cv2.threshold(enhanced, 0, 255,
                                      cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary_combined = cv2.bitwise_and(binary_entropy, binary_otsu)
        refined_mask = self.attention_weighted_morph(binary_combined, entropy_map)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN,
                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        
        contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(gray)
        if contours:
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for cnt in sorted_contours:
                area = cv2.contourArea(cnt)
                if area > Config.MIN_CONTOUR_AREA:
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                        if circularity > 0.3:
                            cv2.drawContours(final_mask, [cnt], -1, 255, -1)
        
        segmented_img = cv2.bitwise_and(img, img, mask=final_mask)
        return segmented_img, final_mask, entropy_map


# ═══════════════════════════════════════════════════════════════════════════
# RADIOMICS FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

class FractalWaveletExtractor:
    """Radiomics: Fractal + Wavelet + GLCM + Gabor"""
    
    def compute_fractal_dimension(self, image):
        threshold = np.mean(image)
        binary = (image > threshold).astype(np.uint8)
        box_sizes = Config.FRACTAL_BOXES
        counts = []
        for box_size in box_sizes:
            try:
                shrunk = cv2.resize(binary,
                                   (binary.shape[1] // box_size,
                                    binary.shape[0] // box_size),
                                   interpolation=cv2.INTER_NEAREST)
                count = np.sum(shrunk > 0)
                counts.append(count)
            except:
                counts.append(0)
        counts = np.array(counts) + 1
        box_sizes = np.array(box_sizes)
        log_boxes = np.log(1.0 / box_sizes)
        log_counts = np.log(counts)
        coeffs = np.polyfit(log_boxes, log_counts, 1)
        fractal_dim = coeffs[0]
        return fractal_dim
    
    def extract_wavelet_features(self, image):
        features = []
        coeffs = pywt.wavedec2(image, Config.WAVELET_FAMILY, level=Config.WAVELET_LEVELS)
        for i, (cH, cV, cD) in enumerate(coeffs[1:], 1):
            for coeff, name in zip([cH, cV, cD], ['H', 'V', 'D']):
                features.extend([
                    np.mean(coeff),
                    np.std(coeff),
                    np.max(np.abs(coeff)),
                    np.percentile(coeff, 75) - np.percentile(coeff, 25)
                ])
        return np.array(features)
    
    def extract_gabor_features(self, image):
        features = []
        frequencies = [0.1, 0.3, 0.5]
        orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        for freq in frequencies:
            for theta in orientations:
                real, imag = gabor(image, frequency=freq, theta=theta)
                features.extend([
                    np.mean(real),
                    np.std(real),
                    np.mean(np.abs(real))
                ])
        return np.array(features)
    
    def extract_advanced_glcm(self, image):
        features = []
        image_norm = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
        glcm = graycomatrix(image_norm,
                           distances=Config.GLCM_DISTANCES,
                           angles=Config.GLCM_ANGLES,
                           levels=256,
                           symmetric=True,
                           normed=True)
        properties = ['contrast', 'dissimilarity', 'homogeneity',
                     'energy', 'correlation', 'ASM']
        for prop in properties:
            try:
                values = graycoprops(glcm, prop).ravel()
                features.extend([
                    np.mean(values),
                    np.std(values),
                    np.max(values),
                    np.min(values)
                ])
            except:
                features.extend([0, 0, 0, 0])
        return np.array(features)
    
    def extract_all_features(self, segmented_img, mask):
        """Extract ALL radiomics features"""
        if len(segmented_img.shape) == 3:
            gray = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = segmented_img
        
        masked_region = cv2.bitwise_and(gray, gray, mask=mask)
        
        try:
            fractal_dim = self.compute_fractal_dimension(masked_region)
            wavelet_feats = self.extract_wavelet_features(masked_region)
            gabor_feats = self.extract_gabor_features(masked_region)
            glcm_feats = self.extract_advanced_glcm(masked_region)
            
            combined = np.concatenate([
                [fractal_dim],
                wavelet_feats,
                gabor_feats,
                glcm_feats
            ])
            return combined
        except Exception as e:
            return None


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_image_paths(data_dir=None):
    """Load all image paths and labels from dataset directory"""
    if data_dir is None:
        data_dir = Config.DATA_DIR
    
    X_paths = []
    y_labels = []
    
    for label_idx, category in enumerate(Config.CATEGORIES):
        category_dir = os.path.join(data_dir, category)
        if not os.path.exists(category_dir):
            print(f"[WARNING] Category directory not found: {category_dir}")
            continue
        
        files = [f for f in os.listdir(category_dir) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for file in files:
            X_paths.append(os.path.join(category_dir, file))
            y_labels.append(label_idx)
    
    return np.array(X_paths), np.array(y_labels)


def extract_features_parallel(image_paths, labels, description="Feature Extraction"):
    """Extract radiomics features from all images in parallel"""
    segmentor = EntropyGuidedSegmentor()
    extractor = FractalWaveletExtractor()
    
    X_features = []
    valid_labels = []
    
    for img_path, label in tqdm(zip(image_paths, labels), total=len(image_paths), 
                                desc=description):
        try:
            segmented_img, mask, _ = segmentor.segment_lung_roi(img_path)
            if segmented_img is None or mask is None:
                continue
            
            features = extractor.extract_all_features(segmented_img, mask)
            if features is None:
                continue
            
            X_features.append(features)
            valid_labels.append(label)
        except Exception as e:
            continue
    
    X_features = np.array(X_features)
    valid_labels = np.array(valid_labels)
    
    return X_features, valid_labels


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_class_weights(y_train):
    """Compute class weights for imbalanced data"""
    unique, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    weights = {}
    for cls, count in zip(unique, counts):
        weights[cls] = total / (len(unique) * count)
    return weights


def get_sample_weights(y):
    """Get sample weights for each training example"""
    class_weights = get_class_weights(y)
    sample_weights = np.array([class_weights[label] for label in y])
    return sample_weights


os.makedirs(Config.MODEL_DIR, exist_ok=True)

if Config.USE_CUDA:
    print(f"\n{'='*80}")
    print(f"[GPU] ACCELERATION ENABLED")
    print(f"{'='*80}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"{'='*80}\n")
