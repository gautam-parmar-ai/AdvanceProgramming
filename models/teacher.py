import torch
import torch.nn as nn
import torchvision.models as models


class TeacherNet(nn.Module):

    def __init__(self, num_classes=100):
        super(TeacherNet, self).__init__()

        # Pretrained backbone
        self.model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )

        # Replace classifier
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        # Feature hooks storage
        self.features = {}
        self._register_hooks()

    def _register_hooks(self):

        def hook_fn(name):
            def fn(module, input, output):
                self.features[name] = output
            return fn

        self.model.layer3.register_forward_hook(hook_fn("layer3"))
        self.model.layer4.register_forward_hook(hook_fn("layer4"))

    def forward(self, x):
        logits = self.model(x)

        return {
            "logits": logits,
            "features": {
                "layer3": self.features.get("layer3", None),
                "layer4": self.features.get("layer4", None)
            }
        }