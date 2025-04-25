import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from torchmetrics.image.fid import FrechetInceptionDistance
import argparse

def load_images_from_folder(folder_path):
    """Load all images from a folder and convert them to tensors."""
    images = []
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # FID uses Inception which expects 299x299 images
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).byte())  # Convert to uint8 tensor
    ])
    
    for filename in os.listdir(folder_path):
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            img_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                images.append(img_tensor)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    if not images:
        raise ValueError(f"No valid images found in {folder_path}")
    
    return torch.stack(images)

def calculate_fid(real_folder, fake_folder):
    """Calculate FID score between two folders of images."""
    print(f"Loading real images from {real_folder}...")
    real_images = load_images_from_folder(real_folder)
    
    print(f"Loading fake images from {fake_folder}...")
    fake_images = load_images_from_folder(fake_folder)
    
    print(f"Loaded {len(real_images)} real images and {len(fake_images)} fake images")
    
    # Initialize FID
    fid = FrechetInceptionDistance(normalize=False)  # Images are already normalized by our transform
    
    # Update with real and fake images
    print("Computing FID score...")
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)
    
    # Compute FID
    fid_score = fid.compute()
    
    return fid_score.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate FID score between two folders of images')
    parser.add_argument('--real', type=str, required=True, help='Path to folder containing real images')
    parser.add_argument('--fake', type=str, required=True, help='Path to folder containing fake/generated images')
    
    args = parser.parse_args()
    
    fid_score = calculate_fid(args.real, args.fake)
    print(f"FID Score: {fid_score:.4f}")