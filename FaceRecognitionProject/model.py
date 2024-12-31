import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3

class FaceVerificationModel(nn.Module):
    """
    Face verification model using InceptionV3 as the base model
    
    Methods:
    - __init__: Initialize the model
    - forward: Forward pass of the model
    
    Args:
    - embedding_size: Dimension of the output embeddings
    
    Forward pass:
    - Takes an image tensor as input
    - Returns the normalized embeddings of the input image
    """
    def __init__(self, embedding_size=128):
        super(FaceVerificationModel, self).__init__()
        self.base_model = inception_v3(pretrained=True)
        self.base_model.aux_logits = False
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

    def forward(self, x):
        if x.size(2) != 299 or x.size(3) != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = self.base_model(x)
        x = F.normalize(x, p=2, dim=1)
        return x