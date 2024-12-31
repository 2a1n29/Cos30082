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
from validation_monitor import ValidationMonitor
   
#Step for trainning     
def train_step(model, loader, device, criterion, optimizer):
    model.train()
    accurate_predictions = 0
    total_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        accurate_predictions += torch.sum(preds == labels.data)
    loss = total_loss / len(loader.dataset)
    accuracy = accurate_predictions.double() / len(loader.dataset)
    return loss, accuracy

#Step for validation
def val_step(model, loader, device, criterion):
    model.eval()
    accurate_predictions = 0
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            accurate_predictions += torch.sum(preds == labels.data)
    loss = total_loss / len(loader.dataset)
    accuracy = accurate_predictions.double() / len(loader.dataset)
    return loss, accuracy

#Trainning the model
def train_model(model, train_loader, val_loader, number_of_epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    validation_monitor = ValidationMonitor(patience=10, verbose=True, delta=0.001)
    logs = []
    
    for epoch in range(number_of_epochs):
        train_loss, train_accuracy = train_step(model, train_loader, device, criterion, optimizer)
        val_loss, val_accuracy = val_step(model, val_loader, device, criterion)

        logs.append({
            'Epoch': epoch + 1,
            'Train Loss': train_loss,
            'Train Accuracy': train_accuracy,
            'Validation Loss': val_loss,
            'Validation Accuracy': val_accuracy
        })

        print(f'Epoch {epoch+1}/{number_of_epochs} --- Loss: {train_loss:.2f}, Accuracy: {train_accuracy:.2f}')
        print(f'Validation --- Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        scheduler.step(val_loss)
        validation_monitor(val_loss, model)
        if validation_monitor.stop_training:
            print("Early stopping triggered")
            break

        logging_df = pd.DataFrame(logs)
        logging_df.to_csv('training_information.csv', index=False)
        
    print('Model trained and data about it is located at training_information.csv')
    return model

#Evaluating the model
def evaluate_model(model, test_loader, number_of_class):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval() 
    correct_predictions = 0
    total = 0
    true_class_predictions = list(0. for i in range(number_of_class))
    total_class_counts = list(0. for i in range(number_of_class))

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct_predictions += (preds == labels).sum().item()
            total += labels.size(0)
            
            for i in range(len(labels)):
                label = labels[i]
                true_class_predictions[label] += (preds[i] == label).item()
                total_class_counts[label] += 1

    top_1_accuracy = correct_predictions / total
    average_accuracy_per_class = sum([true_class_predictions[i] / total_class_counts[i] for i in range(number_of_class) if total_class_counts[i] != 0]) / number_of_class
    
    print(f'Top-1 accuracy: {top_1_accuracy:.4f}')
    print(f'Average accuracy per class: {average_accuracy_per_class:.4f}')
    
    return {'top_1_accuracy': top_1_accuracy,'average_accuracy_per_class': average_accuracy_per_class}

#Loading the model
def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

#Predicting function
def predict(model, image_path, label_map):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        print(f'Prediction: {predicted}')
        predicted_class_index = predicted.item()
        print(f'Class index: {predicted_class_index}')
        predicted_class_name = label_map[predicted_class_index]
        print(f'Class name: {predicted_class_name}')
    return predicted_class_name