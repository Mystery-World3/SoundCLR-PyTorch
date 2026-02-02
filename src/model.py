import torch.nn as nn
import torchvision.models as models

class ESC50Model(nn.Module):
    def __init__(self, n_classes=50, mode='pretrain'):
        super().__init__()
        self.mode = mode
        
        # Backbone
        resnet = models.resnet18(pretrained=True)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1]) # Hapus FC layer
        
        dim_mlp = resnet.fc.in_features
        
        # Head 1: Projection (for Pretrain)
        self.projector = nn.Sequential(
            nn.Linear(dim_mlp, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        # Head 2: Classifier (for Finetune)
        self.classifier = nn.Linear(dim_mlp, n_classes)
        
    def forward(self, x):
        h = self.encoder(x).squeeze()
        
        if self.mode == 'pretrain':
            return self.projector(h)
        else:
            return self.classifier(h)