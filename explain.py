import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms, models

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_backward_hook(self.save_gradients)

    def save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def save_activations(self, module, input, output):
        self.activations = output

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)
        grads = self.gradients.cpu().data.numpy()[0]
        fmap = self.activations.cpu().data.numpy()[0]
        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(fmap.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * fmap[i]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        return cam

def overlay_cam(img: Image.Image, cam: np.ndarray):
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cam_resized = cv2.resize(cam, (img_cv.shape[1], img_cv.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_cv, 0.5, heatmap, 0.5, 0)
    return overlay

def explain(query_path, matched_path, output_path="alerts/output.jpg", device='cpu'):
    resnet_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
    resnet_model.eval()
    gradcam = GradCAM(resnet_model, target_layer=resnet_model.layer4[2].conv3)

    q_img = Image.open(query_path).convert("RGB")
    m_img = Image.open(matched_path).convert("RGB")

    preprocess_tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    q_tensor = preprocess_tensor(q_img).unsqueeze(0).to(device)
    m_tensor = preprocess_tensor(m_img).unsqueeze(0).to(device)

    cam_q = gradcam.generate(q_tensor)
    cam_m = gradcam.generate(m_tensor)

    q_overlay = overlay_cam(q_img, cam_q)
    m_overlay = overlay_cam(m_img, cam_m)

    q_h, q_w = q_overlay.shape[:2]
    m_overlay_resized = cv2.resize(m_overlay, (q_w, q_h))
    combined_img = np.concatenate([q_overlay, m_overlay_resized], axis=1)

    cv2.imwrite(output_path, combined_img)

    return output_path