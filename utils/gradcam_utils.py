import torch
import numpy as np
import cv2

def generate_gradcam(model, img_tensor, class_idx, layer_name='features.denseblock4.denselayer16.conv2'):
    """
    Genera el mapa de calor Grad-CAM para la clase predicha.
    """
    activations = {}
    gradients = {}
    
    def save_activation(name):
        def hook(module, input, output):
            activations[name] = output
        return hook
    
    def save_gradient(name):
        def hook(module, grad_input, grad_output):
            gradients[name] = grad_output[0]
        return hook
    
    # Obtener la capa convolucional objetivo
    target_layer = model.features.denseblock4.denselayer16.conv2
    
    hook_act = target_layer.register_forward_hook(save_activation(layer_name))
    hook_grad = target_layer.register_backward_hook(save_gradient(layer_name))
    
    model.eval()
    output = model(img_tensor)
    
    model.zero_grad()
    loss = output[0, class_idx]
    loss.backward()
    
    act = activations[layer_name][0]  
    grad = gradients[layer_name][0]   
    
    hook_act.remove()
    hook_grad.remove()
    
    pooled_grad = torch.mean(grad, dim=(1, 2), keepdim=True)
    heatmap = torch.sum(pooled_grad * act, dim=0)
    heatmap = torch.nn.functional.relu(heatmap)
    heatmap = heatmap / torch.max(heatmap)
    
    heatmap = heatmap.detach().cpu().numpy()
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    

    img_vis = img_tensor[0].cpu().numpy()  
    img_vis = np.transpose(img_vis, (1, 2, 0))  
    img_vis = (img_vis / 1024 * 127.5 + 127.5).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
    
    superimposed = cv2.addWeighted(img_bgr, 0.6, heatmap_colored, 0.4, 0)
    return superimposed