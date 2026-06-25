

import torch
import torch.nn as nn
import torchvision.models as models


class StudentNet(nn.Module):

    def __init__(self, num_classes=100):
        super().__init__()
        self.model = models.resnet18(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        self.features = {}

        # Projection layers to match teacher channel dims
        self.proj_layer3 = nn.Conv2d(256, 1024, kernel_size=1)
        self.proj_layer4 = nn.Conv2d(512, 2048, kernel_size=1)

        self._register_hooks()

    # -----------------------------
    # Feature hooks
    # -----------------------------
    def _register_hooks(self):

        def hook_fn(name):
            def fn(module, input, output):
                self.features[name] = output
            return fn

        # Match teacher abstraction levels
        self.model.layer3.register_forward_hook(hook_fn("layer3"))
        self.model.layer4.register_forward_hook(hook_fn("layer4"))

    # -----------------------------
    # Forward pass
    # -----------------------------
    def forward(self, x):

        logits = self.model(x)

        return {
            "logits": logits,
            "features": {
                "layer3": self.proj_layer3(self.features.get("layer3")),
                "layer4": self.proj_layer4(self.features.get("layer4")),
            }
        }