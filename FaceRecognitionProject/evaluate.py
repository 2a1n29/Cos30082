import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import FaceVerificationModel
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

def compute_similarity_score(model, img1_path, img2_path, device):
    img1 = load_image(img1_path).to(device)
    img2 = load_image(img2_path).to(device)

    model.eval()
    with torch.no_grad():
        embedding1 = model(img1)
        embedding2 = model(img2)
        similarity = F.cosine_similarity(embedding1, embedding2).item()
    
    return similarity

def main():
    # Paths to the images you want to compare
    img1_path = 'app/saved/captured_face.jpg'
    img2_path = 'app/known_face/haianh.jpg'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FaceVerificationModel().to(device)
    model_save_path = 'models/pretrained/best_model.pt'
    model.load_state_dict(torch.load(model_save_path, map_location=device))

    similarity = compute_similarity_score(model, img1_path, img2_path, device)
    print(f'Similarity score between {img1_path} and {img2_path}: {similarity}')

    # Assuming you have multiple pairs for ROC and AUC calculation
    pairs = [
        ('app/saved/captured_face.jpg', 'app/known_face/haianh.jpg', 1),
        ('app/saved/captured_face.jpg', 'app/known_face/trump.jpg', 0),
    ]

    similarity_scores = []
    labels = []

    for img1_path, img2_path, label in pairs:
        similarity = compute_similarity_score(model, img1_path, img2_path, device)
        similarity_scores.append(similarity)
        labels.append(label)

    fpr, tpr, _ = roc_curve(labels, similarity_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    print(f'AUC: {roc_auc}')

if __name__ == '__main__':
    main()
