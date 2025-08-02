import matplotlib.pyplot as plt
import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ✅ Run GradCAM and plot results
def visualize_gradcam(model, val_loader, class_names, target_layer, save_path="gradcam.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    inputs, labels, _ = next(iter(val_loader))
    inputs, labels = inputs.to(device), labels.to(device)

    # Just use one image from the batch for visualization
    input_img = inputs[0].unsqueeze(0)  # shape (1, C, H, W)
    label = labels[0]

    with torch.no_grad():
        outputs = model(input_img)
        pred_class = outputs.argmax(dim=1).item()

    cam = GradCAM(
        model=model,
        target_layers=[target_layer]
    )
    grayscale_cam = cam(
        input_tensor=input_img,
        targets=[ClassifierOutputTarget(pred_class)],
        aug_smooth=True,
        eigen_smooth=True
    )[0]

    input_image = input_img[0].detach().cpu()
    input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min() + 1e-6)
    input_image = input_image.permute(1, 2, 0).numpy()
    vis = show_cam_on_image(input_image, grayscale_cam, use_rgb=True)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(input_image)
    plt.title(f"GT: {class_names[label.item()]}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(vis)
    plt.title(f"Pred: {class_names[pred_class]}")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)

    plt.show()
    plt.close()

def plot_loss_curve(train_losses, val_losses, save_path="loss_curve.png"):
    """
    Plots and saves the loss curves.

    Args:
        train_losses (list): List of training loss values.
        val_losses (list): List of validation loss values.
        save_path (str): File path to save the plot image.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
