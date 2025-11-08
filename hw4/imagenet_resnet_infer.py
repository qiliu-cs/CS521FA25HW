import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from skimage.segmentation import slic
from skimage.util import img_as_float

# Load the pre-trained ResNet18 model
model = models.resnet18(pretrained=True)
model.eval()  # Set model to evaluation mode

# Define the image preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]   
    )
])

# Load the ImageNet class index mapping
with open("imagenet_class_index.json") as f:
    class_idx = json.load(f)
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
idx2synset = [class_idx[str(k)][0] for k in range(len(class_idx))]
id2label = {v[0]: v[1] for v in class_idx.values()}

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

imagenet_path = './imagenet_samples'

# List of image file paths
image_files = [f for f in os.listdir(imagenet_path) if f.endswith(('.JPEG', '.jpg', '.png'))]

# Denormalization function for visualization
def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

# LIME Implementation
def lime_explanation(model, image_path, num_samples=1000, num_features=100):
    # Load and preprocess image
    original_image = Image.open(image_path).convert('RGB')
    original_image = original_image.resize((224, 224))
    original_array = np.array(original_image) / 255.0
    
    # Get original prediction
    input_tensor = preprocess(original_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        original_prob = probs[0, predicted_class].item()
    
    # Segment the image using SLIC superpixels
    segments = slic(original_array, n_segments=num_features, compactness=10, sigma=1)
    num_segments = segments.max() + 1
    
    # Generate perturbed samples
    perturbed_images = []
    perturbed_masks = np.random.randint(0, 2, size=(num_samples, num_segments))
    
    for mask in perturbed_masks:
        perturbed = original_array.copy()
        for seg_id in range(num_segments):
            if mask[seg_id] == 0:
                perturbed[segments == seg_id] = 0  # Turn off segment
        perturbed_images.append(perturbed)
    
    # Get predictions for perturbed images
    predictions = []
    batch_size = 32
    for i in range(0, len(perturbed_images), batch_size):
        batch = perturbed_images[i:i+batch_size]
        batch_tensors = []
        for img in batch:
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            tensor = preprocess(img_pil).unsqueeze(0)
            batch_tensors.append(tensor)
        
        batch_tensor = torch.cat(batch_tensors, dim=0).to(device)
        with torch.no_grad():
            outputs = model(batch_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            batch_preds = probs[:, predicted_class].cpu().numpy()
            predictions.extend(batch_preds)
    
    predictions = np.array(predictions)
    
    # Fit linear model using weighted least squares
    distances = np.sum((perturbed_masks - 1) ** 2, axis=1)
    kernel_width = 0.25 * num_segments
    weights = np.exp(-distances / kernel_width)
    
    # Add regularization to avoid singularity
    X = perturbed_masks
    y = predictions
    ridge_lambda = 0.01
    
    XtW = X.T * weights
    coefficients = np.linalg.solve(
        XtW @ X + ridge_lambda * np.eye(num_segments),
        XtW @ y
    )
    
    # Create explanation heatmap
    explanation = np.zeros((224, 224))
    for seg_id in range(num_segments):
        explanation[segments == seg_id] = coefficients[seg_id]
    
    return explanation, predicted_class, original_prob, segments

# SmoothGrad Implementation
def smoothgrad_explanation(model, image_path, num_samples=50, noise_level=0.15):
    # Load and preprocess image
    original_image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(original_image).unsqueeze(0).to(device)
    
    # Get original prediction
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()
    
    # Accumulate gradients
    total_gradients = torch.zeros_like(input_tensor)
    
    for _ in range(num_samples):
        # Create a noisy input tensor with requires_grad
        noise = torch.randn_like(input_tensor) * noise_level
        noisy_input = input_tensor + noise
        noisy_input.requires_grad_(True)
        noisy_input.retain_grad()  # Retain gradient for non-leaf tensor
        
        # Forward pass
        output = model(noisy_input)
        
        # Backward pass for the predicted class
        model.zero_grad()
        if noisy_input.grad is not None:
            noisy_input.grad.zero_()
        
        output[0, predicted_class].backward()
        
        # Accumulate gradients
        if noisy_input.grad is not None:
            total_gradients += noisy_input.grad.detach()
    
    # Average gradients
    smoothed_gradients = total_gradients / num_samples
    
    # Convert to numpy and take absolute values
    gradients_np = smoothed_gradients.squeeze().cpu().detach().numpy()
    
    # Aggregate across color channels (take mean of absolute values)
    explanation = np.mean(np.abs(gradients_np), axis=0)
    
    # Get prediction probability
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        original_prob = probs[0, predicted_class].item()
    
    return explanation, predicted_class, original_prob

# Visualization Function
def visualize_explanations(image_path, lime_exp, smoothgrad_exp, predicted_class, prob, save_path):
    """
    Visualize original image and both explanation methods
    """
    original_image = Image.open(image_path).convert('RGB')
    original_image = original_image.resize((224, 224))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title(f'Original Image\n{idx2label[predicted_class]}\nProb: {prob:.3f}')
    axes[0].axis('off')
    
    # LIME explanation
    axes[1].imshow(original_image)
    lime_overlay = axes[1].imshow(lime_exp, cmap='RdBu_r', alpha=0.6, vmin=-np.abs(lime_exp).max(), vmax=np.abs(lime_exp).max())
    axes[1].set_title('LIME Explanation')
    axes[1].axis('off')
    plt.colorbar(lime_overlay, ax=axes[1], fraction=0.046)
    
    # SmoothGrad explanation
    axes[2].imshow(original_image)
    sg_overlay = axes[2].imshow(smoothgrad_exp, cmap='hot', alpha=0.6)
    axes[2].set_title('SmoothGrad Explanation')
    axes[2].axis('off')
    plt.colorbar(sg_overlay, ax=axes[2], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# Comparison Metrics
def compute_ranking_correlation(lime_exp, smoothgrad_exp):
    # Flatten explanations
    lime_flat = lime_exp.flatten()
    smoothgrad_flat = smoothgrad_exp.flatten()
    
    # Compute correlations
    kendall_tau, kendall_p = stats.kendalltau(lime_flat, smoothgrad_flat)
    spearman_rho, spearman_p = stats.spearmanr(lime_flat, smoothgrad_flat)
    
    return kendall_tau, spearman_rho

# Main Processing
print("Processing images and generating explanations...\n")

results = []
os.makedirs('explanations', exist_ok=True)

for idx, img_file in enumerate(image_files[:5]):
    img_path = os.path.join(imagenet_path, img_file)
    print(f"\n{'='*60}")
    print(f"Processing image {idx+1}/5: {img_file}")
    print(f"{'='*60}")
    
    # Generate LIME explanation
    print("Generating LIME explanation...")
    lime_exp, pred_class, prob, segments = lime_explanation(model, img_path)
    
    # Generate SmoothGrad explanation
    print("Generating SmoothGrad explanation...")
    smoothgrad_exp, _, _ = smoothgrad_explanation(model, img_path)
    
    # Compute correlations
    kendall, spearman = compute_ranking_correlation(lime_exp, smoothgrad_exp)
    
    # Visualize
    save_path = f'explanations/explanation_{idx+1}_{img_file}'
    visualize_explanations(img_path, lime_exp, smoothgrad_exp, pred_class, prob, save_path)
    
    # Store results
    results.append({
        'image': img_file,
        'prediction': idx2label[pred_class],
        'probability': prob,
        'kendall_tau': kendall,
        'spearman_rho': spearman
    })
    
    print(f"\nPrediction: {idx2label[pred_class]} (confidence: {prob:.3f})")
    print(f"Kendall-Tau correlation: {kendall:.4f}")
    print(f"Spearman Rank correlation: {spearman:.4f}")
    print(f"Visualization saved to: {save_path}")

# Summary Statistics
print(f"\n{'='*60}")
print("SUMMARY: Correlation Statistics Across All Images")
print(f"{'='*60}")

avg_kendall = np.mean([r['kendall_tau'] for r in results])
avg_spearman = np.mean([r['spearman_rho'] for r in results])

print(f"\nAverage Kendall-Tau correlation: {avg_kendall:.4f}")
print(f"Average Spearman Rank correlation: {avg_spearman:.4f}")

print("\nDetailed Results:")
print(f"{'Image':<25} {'Prediction':<20} {'Prob':<8} {'Kendall':<10} {'Spearman':<10}")
print("-" * 80)
for r in results:
    print(f"{r['image']:<25} {r['prediction']:<20} {r['probability']:<8.3f} {r['kendall_tau']:<10.4f} {r['spearman_rho']:<10.4f}")
