"""
Create visual comparison of epoch accuracy
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

with open('epoch_comparison.json', 'r') as f:
    data = json.load(f)

results = data['all_results']
epochs = sorted(results.keys(), key=lambda x: int(x))

# Extract metrics
ssim_vals = [results[e]['ssim'] for e in epochs]
psnr_vals = [results[e]['psnr'] for e in epochs]
mae_vals = [results[e]['mae'] for e in epochs]
consistency_vals = [results[e]['consistency'] for e in epochs]
pixels_changed = [results[e]['pixels_changed'] for e in epochs]

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Epoch Comparison: Model Accuracy Across Checkpoints', fontsize=16, fontweight='bold')

# SSIM
ax = axes[0, 0]
bars = ax.bar(epochs, ssim_vals, color=['#ff7f0e' if float(e) < 75 else '#2ca02c' for e in epochs], alpha=0.8, edgecolor='black', linewidth=2)
ax.axhline(y=max(ssim_vals), color='green', linestyle='--', linewidth=2, label='Best')
ax.set_ylabel('SSIM Score', fontweight='bold')
ax.set_xlabel('Epoch', fontweight='bold')
ax.set_title('SSIM (Perception Quality) - Higher is Better ⭐', fontweight='bold')
ax.set_ylim([0.7, 0.8])
ax.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, ssim_vals)):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.002, f'{val:.4f}', ha='center', fontweight='bold')
best_ssim = max(enumerate(ssim_vals), key=lambda x: x[1])[0]
bars[best_ssim].set_color('#00c700')
bars[best_ssim].set_edgecolor('darkgreen')

# PSNR
ax = axes[0, 1]
bars = ax.bar(epochs, psnr_vals, color=['#2ca02c' if float(e) <= 50 else '#ff7f0e' for e in epochs], alpha=0.8, edgecolor='black', linewidth=2)
ax.axhline(y=max(psnr_vals), color='green', linestyle='--', linewidth=2, label='Best')
ax.set_ylabel('PSNR (dB)', fontweight='bold')
ax.set_xlabel('Epoch', fontweight='bold')
ax.set_title('PSNR (Signal Fidelity) - Higher is Better', fontweight='bold')
ax.set_ylim([12, 19])
ax.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, psnr_vals)):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.2, f'{val:.2f}', ha='center', fontweight='bold')
best_psnr = max(enumerate(psnr_vals), key=lambda x: x[1])[0]
bars[best_psnr].set_color('#00c700')
bars[best_psnr].set_edgecolor('darkgreen')

# MAE
ax = axes[0, 2]
bars = ax.bar(epochs, mae_vals, color=['#2ca02c' if float(e) <= 50 else '#ff7f0e' for e in epochs], alpha=0.8, edgecolor='black', linewidth=2)
ax.set_ylabel('MAE (Pixels)', fontweight='bold')
ax.set_xlabel('Epoch', fontweight='bold')
ax.set_title('MAE (Mean Absolute Error) - Lower is Better', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, mae_vals)):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.2f}', ha='center', fontweight='bold')
best_mae = min(enumerate(mae_vals), key=lambda x: x[1])[0]
bars[best_mae].set_color('#00c700')
bars[best_mae].set_edgecolor('darkgreen')

# Pixels Changed
ax = axes[1, 0]
bars = ax.bar(epochs, pixels_changed, color=['#d62728' if p < 80 else '#2ca02c' for p in pixels_changed], alpha=0.8, edgecolor='black', linewidth=2)
ax.axhline(y=80, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Min target')
ax.set_ylabel('Pixels Changed >10 (%)', fontweight='bold')
ax.set_xlabel('Epoch', fontweight='bold')
ax.set_title('Transformation Magnitude - Higher is Better (>80%)', fontweight='bold')
ax.set_ylim([70, 95])
ax.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, pixels_changed)):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.1f}%', ha='center', fontweight='bold')

# Consistency
ax = axes[1, 1]
bars = ax.bar(epochs, consistency_vals, color=['#ff7f0e' if c < 0.80 else '#2ca02c' for c in consistency_vals], alpha=0.8, edgecolor='black', linewidth=2)
ax.axhline(y=0.80, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Safe threshold')
ax.set_ylabel('Consistency Score', fontweight='bold')
ax.set_xlabel('Epoch', fontweight='bold')
ax.set_title('Intensity Consistency - Higher is Better (>0.7)', fontweight='bold')
ax.set_ylim([0.75, 1.0])
ax.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, consistency_vals)):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.4f}', ha='center', fontweight='bold')

# Overall Score Ranking
ax = axes[1, 2]
scores = data['scores']
score_vals = [scores[e] for e in epochs]
bars = ax.barh(epochs, score_vals, color=['#2ca02c' if float(e) == '0025' else '#ff7f0e' for e in epochs], alpha=0.8, edgecolor='black', linewidth=2)
ax.set_xlabel('Overall Score (0-1)', fontweight='bold')
ax.set_title('Overall Recommendation Score ⭐', fontweight='bold')
ax.set_xlim([0.7, 0.85])
ax.grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, score_vals)):
    ax.text(val - 0.01, bar.get_y() + bar.get_height()/2, f'{val:.4f}', ha='right', va='center', fontweight='bold', color='white')
bars[0].set_color('#00c700')
bars[0].set_edgecolor('darkgreen')

plt.tight_layout()
plt.savefig('epoch_comparison_visual.png', dpi=150, bbox_inches='tight')
print("✓ Visual comparison saved: epoch_comparison_visual.png")

# Create summary table
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('tight')
ax.axis('off')

table_data = [
    ['Epoch', 'SSIM ⭐', 'PSNR', 'MAE', 'Pixels>10%', 'Consistency', 'Overall Score'],
    ['0025', f"{ssim_vals[0]:.4f}", f"{psnr_vals[0]:.2f} dB", f"{mae_vals[0]:.2f}", f"{pixels_changed[0]:.1f}%", f"{consistency_vals[0]:.4f}", f"{score_vals[0]:.4f} 🏆"],
    ['0050', f"{ssim_vals[1]:.4f}", f"{psnr_vals[1]:.2f} dB", f"{mae_vals[1]:.2f}", f"{pixels_changed[1]:.1f}%", f"{consistency_vals[1]:.4f}", f"{score_vals[1]:.4f}"],
    ['0075', f"{ssim_vals[2]:.4f}", f"{psnr_vals[2]:.2f} dB", f"{mae_vals[2]:.2f}", f"{pixels_changed[2]:.1f}%", f"{consistency_vals[2]:.4f}", f"{score_vals[2]:.4f}"],
    ['0100', f"{ssim_vals[3]:.4f}", f"{psnr_vals[3]:.2f} dB", f"{mae_vals[3]:.2f}", f"{pixels_changed[3]:.1f}%", f"{consistency_vals[3]:.4f}", f"{score_vals[3]:.4f}"],
]

colors = [['#e8f4f8']*7, ['#b3e5fc']*7, ['#ffffff']*7, ['#ffffff']*7, ['#ffffff']*7]

table = ax.table(cellText=table_data, cellLoc='center', loc='center', cellColours=colors)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Bold header
for i in range(7):
    table[(0, i)].set_facecolor('#01579b')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Bold champion row
for i in range(7):
    table[(1, i)].set_facecolor('#c8e6c9')
    table[(1, i)].set_text_props(weight='bold')

plt.title('Epoch Performance Summary', fontsize=14, fontweight='bold', pad=20)
plt.savefig('epoch_summary_table.png', dpi=150, bbox_inches='tight')
print("✓ Summary table saved: epoch_summary_table.png")

plt.show()
