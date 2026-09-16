import torch
import numpy as np

class GradCAM:
    """Computes spatial attention heatmaps for satellite imagery representations."""
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        # Target the final Convolutional layer inside SpatialEncoder
        target_layer = self.model.spatial_enc.net[0]

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, img_tensor, wx_tensor, tab_tensor):
        self.model.eval()
        
        # Forward pass
        yield_pred, _ = self.model(img_tensor, wx_tensor, tab_tensor)
        self.model.zero_grad()
        
        # Backward pass on yield output
        yield_pred.backward(torch.ones_like(yield_pred))

        # Calculate Grad-CAM weights
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]

        for i in range(activations.size(0)):
            activations[i] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize heatmap to [0, 1]
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)

        return heatmap, yield_pred.item()

if __name__ == "__main__":
    print("Grad-CAM module initialized successfully.")