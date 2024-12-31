import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import FaceVerificationModel
from data_loader import FaceVerificationDataset, TRAIN_PATH, VAL_PATH, TEST_PATH
import os

torch.backends.cudnn.enabled = False
torch.cuda.empty_cache()

# Tensorboard logging
log_dir = 'runs/training_logs'
writer = SummaryWriter(log_dir=log_dir)

model_save_path = 'models/train/best_model/'
best_model_path = os.path.join(model_save_path, 'best_model.pt')
os.makedirs(model_save_path, exist_ok=True)

class CustomTrainer:
    """
    Custom trainer for training a face verification model
    
    Methods:
    - train_epoch: Train the model for one epoch
    - validate_epoch: Validate the model for one epoch
    - train: Train the model for multiple epochs
    
    Args:
    - model: Face verification model
    - train_loader: DataLoader for training dataset
    - val_loader: DataLoader for validation dataset
    - criterion: Triplet loss function
    - optimizer: Optimizer for training the model
    - device: Device to run the model on
    - num_epochs: Number of epochs to train the model
    """
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.current_epoch = 0
        self.best_val_loss = float('inf')

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        for anchor, positive, negative in self.train_loader:
            anchor, positive, negative = anchor.to(self.device), positive.to(self.device), negative.to(self.device)
            
            # Forward pass
            anchor_embedding = self.model(anchor)
            positive_embedding = self.model(positive)
            negative_embedding = self.model(negative)
            loss = self.criterion(anchor_embedding, positive_embedding, negative_embedding)
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(self.train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, self.current_epoch)
        writer.add_scalar('Learning Rate', self.optimizer.param_groups[0]['lr'], self.current_epoch)
        return avg_train_loss

    def validate_epoch(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for anchor, positive, negative in self.val_loader:
                anchor, positive, negative = anchor.to(self.device), positive.to(self.device), negative.to(self.device)
                anchor_embedding = self.model(anchor)
                positive_embedding = self.model(positive)
                negative_embedding = self.model(negative)
                loss = self.criterion(anchor_embedding, positive_embedding, negative_embedding)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(self.val_loader)
        writer.add_scalar('Loss/val', avg_val_loss, self.current_epoch)        
        return avg_val_loss

    def train(self):
        for epoch in range(self.num_epochs):
            print(f'Epoch [{epoch+1}/{self.num_epochs}]')
            self.current_epoch = epoch
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            # Save the best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), best_model_path)
                print(f'Saved best model with val loss: {val_loss:.4f}')

def main():
    max_samples = 1000
    train_dataset = FaceVerificationDataset(TRAIN_PATH, max_samples=max_samples)
    val_dataset = FaceVerificationDataset(VAL_PATH, max_samples=max_samples)
    test_dataset = FaceVerificationDataset(TEST_PATH, max_samples=max_samples)
    
    print(f'----- Dataset Information -----')
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(val_dataset)}')
    print(f'Test dataset size: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f'----- DataLoader Information -----')
    print(f'Train loader size: {len(train_loader)}')
    print(f'Validation loader size: {len(val_loader)}')
    print(f'Test loader size: {len(test_loader)}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FaceVerificationModel(embedding_size=128).to(device)

    model_save_path = 'models/train/best_model/best_model.pt'
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        print(f'Loaded model from: {model_save_path}')

    optimizer = optim.Adam(model.parameters(), lr=0.0001)    
    criterion = nn.TripletMarginLoss(margin=1.2, p=2)

    trainer = CustomTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=30
    )

    trainer.train()

if __name__ == '__main__':
    main()
