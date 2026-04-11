from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image

def test_single_image(image_path, checkpoint_path):
    print("STEP 1: Function started")

    device = Config.DEVICE

    print("STEP 2: Creating model")
    G = ResNetGenerator().to(device)

    print("STEP 3: Loading checkpoint:", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    print("Checkpoint keys:", checkpoint.keys())

    # SAFE loading
    if 'G_AB' in checkpoint:
        G.load_state_dict(checkpoint['G_AB'])
    elif 'G_state' in checkpoint:
        G.load_state_dict(checkpoint['G_state'])
    else:
        G.load_state_dict(checkpoint)

    G.eval()
    print("STEP 4: Model ready")

    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    print("STEP 5: Loading image")
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    print("STEP 6: Generating output")
    with torch.no_grad():
        fake = G(input_tensor)

    def denorm(x):
        return (x + 1) / 2

    input_img = denorm(input_tensor[0]).cpu()
    fake_img = denorm(fake[0]).cpu()

    diff = torch.abs(input_img - fake_img)

    print("STEP 7: Saving images")
    save_image(input_img, "input.png")
    save_image(fake_img, "generated.png")
    save_image(diff, "difference.png")

    print("Saved: input.png, generated.png, difference.png")