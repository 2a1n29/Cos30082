import torch.nn as nn
from torchvision import models

#The model class
class BirdClassifier(nn.Module):
    def __init__(self, number_of_class=200, tuning_start_layer=5):
        super(BirdClassifier, self).__init__()
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
      
        if tuning_start_layer < 0:
            for param in self.base_model.parameters():
                param.requires_grad = True
        else:
            layers = list(self.base_model.children())[:tuning_start_layer]
            for layer in layers:
                for param in layer.parameters():
                    param.requires_grad = False

        number_of_feature = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(number_of_feature, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, number_of_class)
        )
    
    def forward(self, x):
        return self.base_model(x)