import torch
import torch.nn as nn
from torchvision import transforms, models
from huggingface_hub import hf_hub_download
from PIL import Image
import torch.nn.functional as F
from MTL import MTL

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_resnet50_semart():
    num_classes_school = 26
    num_classes_type = 10

    model_path = hf_hub_download(
        repo_id="Irina1402/resnet50-semart", 
        filename="model.pth"
    )

    # Load the updated MTL model
    model = MTL(num_classes_school, num_classes_type)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    school_labels = sorted([
        "American", "Austrian", "Belgian", "Bohemian", "Catalan", "Danish", "Dutch", "English", "Finnish",
        "Flemish", "French", "German", "Greek", "Hungarian", "Irish", "Italian", "Netherlandish", "Norwegian",
        "Other", "Polish", "Portuguese", "Russian", "Scottish", "Spanish", "Swedish", "Swiss"
    ])

    type_labels = sorted([
        "genre", "historical", "interior", "landscape", "mythological", "other",
        "portrait", "religious", "still-life", "study"
    ])

    return {
        "model": model,
        "school_labels": school_labels,
        "type_labels": type_labels,
        "num_classes_school": num_classes_school
    }


def load_model_resnet50_balanced():
    num_classes_school = 8
    num_classes_type = 8

    model_path = hf_hub_download(
        repo_id="Irina1402/resnet50-balanced", 
        filename="model.pth"
    )

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes_school + num_classes_type)
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    school_labels = sorted([
        "French", "American", "Russian", "British", "Italian", "Spanish", "German", "Dutch"
    ])

    type_labels = sorted([
        "portrait", "landscape", "abstract", "genre painting", "religious painting",
        "cityscape", "sketch and study", "still life"
    ])


    return {
        "model": model,
        "school_labels": school_labels,
        "type_labels": type_labels,
        "num_classes_school": num_classes_school
    }


models_registry = {
    "model_semart": load_model_resnet50_semart(),
    "model_balanced": load_model_resnet50_balanced()
}


def classify_image(image: Image.Image, model_name, confidence_threshold=0.20, strong_threshold=0.80, topk=3):
    if model_name not in models_registry:
        return {"error": f"Modelul '{model_name}' nu este disponibil."}

    model_data = models_registry[model_name]
    model = model_data["model"]
    school_labels = model_data["school_labels"]
    type_labels = model_data["type_labels"]
    num_classes_school = model_data["num_classes_school"]

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        if model_name == "model_semart":
            # Two outputs directly
            school_output, type_output = model(input_tensor)
        else:
            # Single output to split manually
            output = model(input_tensor)
            school_output = output[:, :num_classes_school]
            type_output = output[:, num_classes_school:]

    school_probs = F.softmax(school_output, dim=1).squeeze()
    type_probs = F.softmax(type_output, dim=1).squeeze()

    # SCHOOL
    topk_school = torch.topk(school_probs, k=topk)
    school_top1_idx = topk_school.indices[0].item()
    school_top1_prob = school_probs[school_top1_idx].item()

    if school_top1_prob >= strong_threshold:
        school_predictions = [{
            "label": school_labels[school_top1_idx],
            "score": round(school_top1_prob * 100, 1)  # procent
        }]
    else:
        school_predictions = [
            {
                "label": school_labels[i.item()],
                "score": round(school_probs[i].item() * 100, 1)
            }
            for i in topk_school.indices
            if school_probs[i].item() >= confidence_threshold
        ]
        if not school_predictions:
            school_predictions = [{"label": "Unknown", "score": None}]

    # TYPE 
    topk_type = torch.topk(type_probs, k=topk)
    type_top1_idx = topk_type.indices[0].item()
    type_top1_prob = type_probs[type_top1_idx].item()

    if type_top1_prob >= strong_threshold:
        type_predictions = [{
            "label": type_labels[type_top1_idx],
            "score": round(type_top1_prob * 100, 1)
        }]
    else:
        type_predictions = [
            {
                "label": type_labels[i.item()],
                "score": round(type_probs[i].item() * 100, 1)
            }
            for i in topk_type.indices
            if type_probs[i].item() >= confidence_threshold
        ]
        if not type_predictions:
            type_predictions = [{"label": "Unknown", "score": None}]

    return {
        "school_prediction": school_predictions,
        "type_prediction": type_predictions
    }
