import cv2
import os
from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as F
from scipy.spatial import distance
import torchvision
from facetorch import FaceAnalyzer
import operator
from typing import Tuple, List
from omegaconf import OmegaConf

known_faces_directory = 'app/known_face'
log_dir = 'runs/training_logs'

# Load an image and apply transformations
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

# Create embeddings for all images in a directory
def create_embeddings(model, image_directory):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()  
    embeddings = {}

    for filename in os.listdir(image_directory):
        print(f'Creating embeddings for images in {image_directory} for {filename}')
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            person_name = filename.split('.')[0]
            image_path = os.path.join(image_directory, filename)
            image_tensor = load_image(image_path).to(device)
            with torch.no_grad():
                embedding = model(image_tensor).cpu()
            embeddings[person_name] = embedding.squeeze(0)  # Remove batch dimension

    return embeddings

# Identify a face in an image using a set of known embeddings
def identify_face(model, input_image_path, known_embeddings, threshold=0.5, distance_metric='cosine'):
    print(f'Identifying face in {input_image_path}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_image = load_image(input_image_path).to(device)
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        input_embedding = model(input_image).cpu().squeeze(0)  # Remove batch dimension

    best_score = None
    identified_person = "Unknown"
    
    for name, embedding in known_embeddings.items():
        print(f'Calculating distance for {name}')
        
        if distance_metric == 'cosine':
            # Calculate the cosine similarity
            score = torch.nn.functional.cosine_similarity(input_embedding.unsqueeze(0), embedding.unsqueeze(0)).item()
            print(f'Similarity for {name}: {score}')
            
            if best_score is None or score > best_score:
                best_score = score
                identified_person = name if score >= threshold else "Unknown"
        
        elif distance_metric == 'euclidean':
            # Calculate the Euclidean distance
            score = torch.nn.functional.pairwise_distance(input_embedding.unsqueeze(0), embedding.unsqueeze(0)).item()
            print(f'Distance for {name}: {score}')
            
            if best_score is None or score < best_score:
                best_score = score
                identified_person = name if score <= threshold else "Unknown"
        
        elif distance_metric == 'manhattan':
            # Calculate the Manhattan distance
            score = torch.sum(torch.abs(input_embedding - embedding)).item()
            print(f'Manhattan distance for {name}: {score}')
            
            if best_score is None or score < best_score:
                best_score = score
                identified_person = name if score <= threshold else "Unknown"
        
        elif distance_metric == 'chebyshev':
            # Calculate the Chebyshev distance
            score = torch.max(torch.abs(input_embedding - embedding)).item()
            print(f'Chebyshev distance for {name}: {score}')
            
            if best_score is None or score < best_score:
                best_score = score
                identified_person = name if score <= threshold else "Unknown"

    print(f'Identified person: {identified_person} with {"similarity" if distance_metric == "cosine" else "distance"} {best_score}')
    return identified_person


# Eye aspect ratio calculation
def eye_aspect_ratio(eye):
    return (distance.euclidean(eye[1], eye[5]) + distance.euclidean(eye[2], eye[4])) / (2.0 * distance.euclidean(eye[0], eye[3]))

# Load the FaceTorch configuration
cfg = OmegaConf.load('app/facetorch/config.merged.yml')
analyzer = FaceAnalyzer(cfg.analyzer)

# Perform inference on an image
def inference(path_image: str) -> Tuple:
    response = analyzer.run(
        path_image=path_image,
        batch_size=cfg.batch_size,
        fix_img_size=cfg.fix_img_size,
        return_img_data=cfg.return_img_data,
        include_tensors=cfg.include_tensors,
        path_output=None,
    )
    
    pil_image = torchvision.transforms.functional.to_pil_image(response.img)
    
    fer_dict_str = str({face.indx: face.preds["fer"].label for face in response.faces})
    deepfake_dict_str = str({face.indx: face.preds["deepfake"].label for face in response.faces})
    va_dict_str = str({face.indx: face.preds["va"].other for face in response.faces})
    
    out_tuple = (pil_image, fer_dict_str, deepfake_dict_str, va_dict_str)
    return out_tuple

# Draw a frame around the face in the image
def draw_face_frame(frame):
    height, width = frame.shape[:2]
    frame_size = int(0.75 * height)  # Frame size is 75% of the frame height
    top_left = (int((width - frame_size) / 2), int((height - frame_size) / 2))
    bottom_right = (top_left[0] + frame_size, top_left[1] + frame_size)
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
    return frame, top_left, bottom_right

# Crop the frame to the face region
def crop_to_face_frame(frame, top_left, bottom_right):
    return frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

# Perform final operations after capturing the image
def evaluate_model_on_test_data(model, test_loader) -> Tuple[List[float], List[int]]:
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    predictions = []
    labels = []

    with torch.no_grad():
        for anchor, positive, negative in test_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            anchor_embedding = model(anchor).cpu()
            positive_embedding = model(positive).cpu()
            negative_embedding = model(negative).cpu()

            pos_similarity = F.cosine_similarity(anchor_embedding, positive_embedding).numpy()
            neg_similarity = F.cosine_similarity(anchor_embedding, negative_embedding).numpy()

            predictions.extend(pos_similarity)
            labels.extend([1] * len(pos_similarity))

            predictions.extend(neg_similarity)
            labels.extend([0] * len(neg_similarity))

    return predictions, labels

