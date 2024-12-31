import os
from io import BytesIO
import requests
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

class ValidationMonitor:
    def __init__(self, patience=8, verbose=False, delta=0):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.highest_score = None
        self.stop_training = False
        self.min_validation_loss = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.highest_score is None:
            self.highest_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.highest_score + self.delta:
            self.counter += 1
            print(f'Stopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.stop_training = True
        else:
            self.highest_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss changed from ({self.min_validation_loss:.2f} to {val_loss:.2f}).  Model saved !')
        path = 'models'
        model_saved_dir = os.path.join(path, f'checkpoint.pt')
        os.makedirs(os.path.dirname(model_saved_dir), exist_ok=True)
        torch.save(model.state_dict(), model_saved_dir)
        self.min_validation_loss = val_loss