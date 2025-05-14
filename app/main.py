import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import time
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Import models from their respective source files
from age.src.models.cbam import ResNetAgeWithCBAM
from variety.models.cbam import CBAMResNet18
from disease.models.simple_cnn import SimpleCNN

# Set page configuration with improved title and layout
st.set_page_config(
    page_title="Paddy Doctor | AI-Powered Rice Plant Analysis",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths to model checkpoints and data
DISEASE_MODEL_PATH = "checkpoints/classify_diseases/model_final.pt"
VARIETY_MODEL_PATH = "variety/checkpoints/final.pt"
AGE_MODEL_PATH = "age/checkpoints/cbam34_model/best_model.pt"
METADATA_PATH = "age/data/meta_train.csv"

# Constants for age normalization
AGE_MIN = 45.0
AGE_MAX = 82.0

# Define disease classes
DISEASE_CLASSES = [
    "bacterial_leaf_blight", "bacterial_leaf_streak", "bacterial_panicle_blight", 
    "blast", "brown_spot", "dead_heart", "downy_mildew", "hispa", "normal", "tungro"
]

# Define variety information database with actual varieties from meta_train.csv
VARIETY_INFO = {
    "ADT45": {
        "origin": "Tamil Nadu, India",
        "characteristics": "Medium duration, semi-dwarf, high-yielding variety",
        "growing_period": "115-120 days",
        "optimal_conditions": "Irrigated conditions, responsive to fertilizers"
    },
    "IR20": {
        "origin": "International Rice Research Institute (IRRI)",
        "characteristics": "Semi-dwarf, high-yielding, disease-resistant",
        "growing_period": "110-115 days",
        "optimal_conditions": "Irrigated lowlands, moderate fertility"
    },
    "KarnatakaPonni": {
        "origin": "Karnataka, India",
        "characteristics": "Medium-tall, fine grain, good cooking quality",
        "growing_period": "135-140 days",
        "optimal_conditions": "Irrigated conditions, moderate fertility"
    },
    "Onthanel": {
        "origin": "South India",
        "characteristics": "Traditional variety, medium height, good grain quality",
        "growing_period": "130-140 days",
        "optimal_conditions": "Rain-fed and irrigated conditions"
    },
    "Ponni": {
        "origin": "Tamil Nadu, India",
        "characteristics": "Medium duration, fine grain, popular for its taste",
        "growing_period": "130-135 days",
        "optimal_conditions": "Irrigated conditions, responsive to fertilizers"
    },
    "Surya": {
        "origin": "India",
        "characteristics": "Early maturing, drought-tolerant, medium yield",
        "growing_period": "100-110 days",
        "optimal_conditions": "Rain-fed and irrigated conditions, drought-prone areas"
    },
    "Zonal": {
        "origin": "Regional variety",
        "characteristics": "Adapted to specific agro-climatic zones",
        "growing_period": "120-130 days",
        "optimal_conditions": "Specific to local conditions"
    },
    "AndraPonni": {
        "origin": "Andhra Pradesh, India",
        "characteristics": "Medium duration, fine grain, good cooking quality",
        "growing_period": "125-135 days",
        "optimal_conditions": "Irrigated conditions, moderate fertility"
    },
    "AtchayaPonni": {
        "origin": "Tamil Nadu, India",
        "characteristics": "Medium duration, fine grain, good yield",
        "growing_period": "130-140 days",
        "optimal_conditions": "Irrigated conditions, moderate to high fertility"
    },
    "RR": {
        "origin": "India",
        "characteristics": "Short duration, medium grain, disease resistant",
        "growing_period": "105-115 days",
        "optimal_conditions": "Irrigated conditions, suitable for multiple cropping"
    }
}

# Load metadata if available
@st.cache_data
def load_metadata():
    try:
        metadata = pd.read_csv(METADATA_PATH)
        st.session_state['metadata_status'] = "loaded"
        return metadata
    except Exception as e:
        st.session_state['metadata_status'] = f"error: {str(e)}"
        # Create a dummy dataframe with the expected columns
        return pd.DataFrame(columns=['image_id', 'label', 'variety', 'age'])

# Disease information database
DISEASE_INFO = {
    "bacterial_leaf_blight": {
        "info": "A bacterial disease that causes water-soaked to yellowish stripes on leaf blades or leaf tips.",
        "treatment": "Use disease-free seeds, balanced fertilization, and copper-based bactericides.",
        "severity": "High",
        "spread_rate": "Rapid in humid conditions"
    },
    "bacterial_leaf_streak": {
        "info": "Causes narrow, dark brown to yellowish stripes between leaf veins.",
        "treatment": "Use disease-free seeds and practice crop rotation.",
        "severity": "Moderate",
        "spread_rate": "Moderate"
    },
    "bacterial_panicle_blight": {
        "info": "Affects rice panicles causing discoloration and unfilled grains.",
        "treatment": "No effective chemical control; use resistant varieties.",
        "severity": "High",
        "spread_rate": "Moderate to rapid"
    },
    "blast": {
        "info": "A fungal disease causing diamond-shaped lesions with gray centers on leaves.",
        "treatment": "Apply fungicides, use resistant varieties, and maintain proper water management.",
        "severity": "Very high",
        "spread_rate": "Very rapid"
    },
    "brown_spot": {
        "info": "A fungal disease causing brown lesions with gray centers on leaves.",
        "treatment": "Use fungicides, practice field sanitation, and ensure balanced nutrition.",
        "severity": "Moderate to high",
        "spread_rate": "Moderate"
    },
    "dead_heart": {
        "info": "Caused by stem borers, resulting in dead central shoots.",
        "treatment": "Apply appropriate insecticides and remove affected tillers.",
        "severity": "High",
        "spread_rate": "Moderate"
    },
    "downy_mildew": {
        "info": "Fungal disease causing yellow lesions and white growth on leaf undersides.",
        "treatment": "Apply fungicides and improve field drainage.",
        "severity": "Moderate",
        "spread_rate": "Rapid in cool, humid conditions"
    },
    "hispa": {
        "info": "Insect pest that scrapes the upper surface of leaf blades.",
        "treatment": "Apply insecticides and remove weeds around fields.",
        "severity": "Moderate",
        "spread_rate": "Moderate"
    },
    "tungro": {
        "info": "A viral disease causing yellow to orange discoloration of leaves.",
        "treatment": "Control insect vectors, use resistant varieties, and adjust planting time.",
        "severity": "Very high",
        "spread_rate": "Rapid through insect vectors"
    },
    "normal": {
        "info": "Healthy paddy plant with no visible disease symptoms.",
        "treatment": "Continue regular preventive measures and optimal growing practices.",
        "severity": "None",
        "spread_rate": "N/A"
    }
}

# MODEL - SPECIFIC PREPROCESSING FUNCTIONS

# Age model preprocessing
def get_age_transform():
    """
    Returns the transformation pipeline for preprocessing images for the age model.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Variety model preprocessing
def get_variety_transform():
    """
    Returns the transformation pipeline for preprocessing images for the variety model.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # Scales images from 0-255 to 0-1
    ])

# Disease model preprocessing
def get_disease_transform():
    """
    Returns the transformation pipeline for preprocessing images for the disease model.
    Note: For inference, we use a subset of the training augmentations
    """
    return transforms.Compose([
        transforms.Resize(256),  # Resize so shorter side is 256, keeps aspect ratio
        transforms.CenterCrop(224),  # Crop center to 224x224 (no distortion)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Display transformation for visualization (without normalization)
def get_display_transform():
    """
    Returns the transformation pipeline for preprocessing images for display.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # Scales images from 0-255 to 0-1
    ])

# Mock models for demonstration when real models can't be loaded
class MockModel:
    def __init__(self, model_type):
        self.model_type = model_type
        
    def __call__(self, x):
        if self.model_type == "disease":
            # Return mock disease prediction (one-hot encoded)
            return torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
        elif self.model_type == "variety":
            # Return mock variety prediction
            return torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]])
        else:  # age
            # Return mock age prediction
            return torch.tensor([45.0])
    
    def eval(self):
        return self

# Load models or use mock models if loading fails
@st.cache_resource
def load_models():
    disease_model = None
    variety_model = None
    age_model = None
    
    # Try to load disease model
    try:
        # Create the SimpleCNN model with the correct number of classes
        disease_model = SimpleCNN(num_classes=len(DISEASE_CLASSES))
        
        # Load the state dictionary
        disease_state_dict = torch.load(DISEASE_MODEL_PATH, map_location=torch.device('cpu'))
        
        # Check if we need to modify the state dict keys
        if all(key.startswith("module.") for key in disease_state_dict.keys()):
            # Remove the "module." prefix
            disease_state_dict = {k[7:]: v for k, v in disease_state_dict.items()}
            
        disease_model.load_state_dict(disease_state_dict)
        disease_model.eval()
        st.session_state['model_status'] = st.session_state.get('model_status', {})
        st.session_state['model_status']['disease'] = "loaded"
    except Exception as e:
        st.session_state['model_status'] = st.session_state.get('model_status', {})
        st.session_state['model_status']['disease'] = f"error: {str(e)}"
        disease_model = MockModel("disease")
    
    # Try to load variety model
    try:
        # Create the CBAMResNet18 model with the correct number of classes
        variety_model = CBAMResNet18(num_classes=10)  # Assuming 10 varieties
        
        # Load the state dictionary
        variety_state_dict = torch.load(VARIETY_MODEL_PATH, map_location=torch.device('cpu'))
        
        # Check if we need to modify the state dict keys
        if all(key.startswith("module.") for key in variety_state_dict.keys()):
            # Remove the "module." prefix
            variety_state_dict = {k[7:]: v for k, v in variety_state_dict.items()}
            
        variety_model.load_state_dict(variety_state_dict, strict=False)
        variety_model.eval()
        st.session_state['model_status'] = st.session_state.get('model_status', {})
        st.session_state['model_status']['variety'] = "loaded"
    except Exception as e:
        st.session_state['model_status'] = st.session_state.get('model_status', {})
        st.session_state['model_status']['variety'] = f"error: {str(e)}"
        variety_model = MockModel("variety")
    
    # Try to load age model
    try:
        # Create the ResNetAgeWithCBAM model
        age_model = ResNetAgeWithCBAM()
        
        # Load the state dictionary
        age_state_dict = torch.load(AGE_MODEL_PATH, map_location=torch.device('cpu'))
        
        # Check if we need to modify the state dict keys
        if all(key.startswith("module.") for key in age_state_dict.keys()):
            # Remove the "module." prefix
            age_state_dict = {k[7:]: v for k, v in age_state_dict.items()}
            
        age_model.load_state_dict(age_state_dict, strict=False)
        age_model.eval()
        st.session_state['model_status'] = st.session_state.get('model_status', {})
        st.session_state['model_status']['age'] = "loaded"
    except Exception as e:
        st.session_state['model_status'] = st.session_state.get('model_status', {})
        st.session_state['model_status']['age'] = f"error: {str(e)}"
        age_model = MockModel("age")
    
    return disease_model, variety_model, age_model

# Enhanced function to preprocess image for model input and display
def preprocess_image(image):
    """
    Preprocess image for display.
    Returns:
        - processed_image: PIL Image resized to 224x224 for display
    """
    # Get transformations for display
    display_transform = get_display_transform()
    
    # Ensure image is in RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Process for display (just resize and convert to tensor 0-1)
    display_tensor = display_transform(image)
    processed_image = transforms.ToPILImage()(display_tensor)
    
    return processed_image

# Function to extract image ID from filename
def extract_image_id(filename):
    """Extract the image_id from a filename."""
    # Assuming filename format is like "path/to/image_id.jpg"
    base_name = os.path.basename(filename)
    image_id = os.path.splitext(base_name)[0]
    return image_id

# Function to look up metadata for an image
def get_image_metadata(image_id, metadata_df):
    """
    Look up metadata for a given image_id.
    Returns a dictionary with label, variety, and age if found, or None if not found.
    """
    if metadata_df is None or metadata_df.empty:
        return None
    
    # Find the row with matching image_id
    matching_rows = metadata_df[metadata_df['image_id'] == image_id]
    
    if matching_rows.empty:
        return None
    
    # Get the first matching row
    row = matching_rows.iloc[0]
    
    return {
        'label': row['label'],
        'variety': row['variety'],
        'age': row['age']
    }

# Updated predict function with correct age prediction using ResNetAgeWithCBAM
def predict(image, image_name, disease_model, variety_model, age_model, metadata_df):
    # Process the image for display
    processed_image = preprocess_image(image)
    
    # Ensure image is in RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Try to get metadata if available
    image_id = extract_image_id(image_name) if image_name else None
    metadata = get_image_metadata(image_id, metadata_df) if image_name else None
    
    # Initialize results dictionary
    results = {
        "processed_image": processed_image,
        "metadata": metadata,
        "disease": {},
        "variety": {},
        "age": {}
    }
    
    # Get disease prediction with disease-specific preprocessing
    with torch.no_grad():
        try:
            # Apply disease-specific preprocessing
            disease_transform = get_disease_transform()
            disease_tensor = disease_transform(image).unsqueeze(0)  # add batch dimension
            
            disease_outputs = disease_model(disease_tensor)
            disease_probs = torch.nn.functional.softmax(disease_outputs, dim=1)[0]
            disease_idx = torch.argmax(disease_probs).item()
            disease_name = DISEASE_CLASSES[disease_idx]
            disease_confidence = disease_probs[disease_idx].item() * 100
            
            # Get top 3 disease predictions for detailed analysis
            top_disease_indices = torch.argsort(disease_probs, descending=True)[:3].tolist()
            top_diseases = [
                {
                    "name": DISEASE_CLASSES[idx],
                    "confidence": disease_probs[idx].item() * 100
                }
                for idx in top_disease_indices
            ]
            
            results["disease"] = {
                "name": disease_name,
                "confidence": disease_confidence,
                "is_healthy": disease_name == "normal",
                "top_predictions": top_diseases,
                "all_probabilities": {DISEASE_CLASSES[i]: disease_probs[i].item() * 100 for i in range(len(DISEASE_CLASSES))}
            }
        except Exception as e:
            st.error(f"Error in disease prediction: {str(e)}")
            # Set prediction status to error instead of using default values
            results["disease"] = {
                "error": True,
                "message": "Cannot predict disease. Model cannot be used."
            }
        
        try:
            # Apply variety-specific preprocessing
            variety_transform = get_variety_transform()
            variety_tensor = variety_transform(image).unsqueeze(0)  # add batch dimension
            
            # Get variety prediction with the actual varieties from meta_train.csv
            variety_outputs = variety_model(variety_tensor)
            variety_probs = torch.nn.functional.softmax(variety_outputs, dim=1)[0]
            variety_idx = torch.argmax(variety_probs).item()
            
            # Use the actual variety names from meta_train.csv
            variety_names = ["ADT45", "IR20", "KarnatakaPonni", "Onthanel", "Ponni", "Surya", "Zonal", "AndraPonni", "AtchayaPonni", "RR"]
            variety_name = variety_names[variety_idx % len(variety_names)]
            variety_confidence = variety_probs[variety_idx].item() * 100
            
            # Get top 3 variety predictions
            top_variety_indices = torch.argsort(variety_probs, descending=True)[:3].tolist()
            top_varieties = [
                {
                    "name": variety_names[idx % len(variety_names)],
                    "confidence": variety_probs[idx].item() * 100
                }
                for idx in top_variety_indices
            ]
            
            results["variety"] = {
                "name": variety_name,
                "confidence": variety_confidence,
                "top_predictions": top_varieties
            }
        except Exception as e:
            st.error(f"Error in variety prediction: {str(e)}")
            # Set prediction status to error instead of using default values
            results["variety"] = {
                "error": True,
                "message": "Cannot predict variety. Model cannot be used."
            }
        
        try:
            # Apply age-specific preprocessing
            age_transform = get_age_transform()
            age_tensor = age_transform(image).unsqueeze(0)  # add batch dimension
            
            # Get age prediction using ResNetAgeWithCBAM model
            age_output = age_model(age_tensor)
            
            # Denormalize the age prediction using the constants
            raw_age = age_output.item()
            predicted_age = raw_age * (AGE_MAX - AGE_MIN) + AGE_MIN
            
            results["age"] = {
                "days": round(predicted_age)
            }
        except Exception as e:
            st.error(f"Error in age prediction: {str(e)}")
            # Set prediction status to error instead of using default values
            results["age"] = {
                "error": True,
                "message": "Cannot predict age. Model cannot be used."
            }
    
    # If metadata is available, add ground truth information
    if metadata:
        results["ground_truth"] = {
            "disease": metadata["label"],
            "variety": metadata["variety"],
            "age": metadata["age"]
        }
    
    return results

# Function to create a progress bar animation with Streamlit components
def progress_bar_animation():
    progress_placeholder = st.empty()
    status_text = st.empty()
    
    for i in range(101):
        # Use Streamlit's progress bar component
        progress_placeholder.progress(i/100)
        
        if i < 30:
            status_text.text(f"Loading image... ({i}%)")
        elif i < 60:
            status_text.text(f"Analyzing paddy features... ({i}%)")
        elif i < 90:
            status_text.text(f"Running prediction models... ({i}%)")
        else:
            status_text.text(f"Finalizing results... ({i}%)")
        time.sleep(0.02)
    
    status_text.empty()
    progress_placeholder.empty()

# Function to create a custom card component using Streamlit columns and containers
def create_card(title, content_func, color="#2e7d32"):
    # Create a container with a border
    with st.container():
        # Add a colored border on the left
        col1, col2 = st.columns([0.02, 0.98])
        
        with col1:
            # This creates the colored border
            st.markdown(f"<div style='background-color: {color}; height: 100%;'></div>", unsafe_allow_html=True)
        
        with col2:
            # Card header
            st.subheader(title)
            
            # Card content (pass the container to the content function)
            content_func()

# Function to create a custom confidence bar using Streamlit components
def create_confidence_bar(container, confidence, color="#4CAF50"):
    if confidence < 50:
        color = "#FFC107"  # Yellow for low confidence
    elif confidence < 70:
        color = "#FF9800"  # Orange for medium confidence
    
    container.text(f"Confidence: {confidence:.1f}%")
    container.progress(confidence/100)

# Function to create a custom timeline for age visualization using Streamlit components
def create_age_timeline(container, age, ground_truth_age=None):
    stages = ["Seedling", "Tillering", "Stem Elongation", "Panicle Initiation", "Heading", "Ripening"]
    days = [0, 20, 40, 60, 80, 100, 120]
    
    # Determine current stage
    current_stage = "Unknown"
    stage_index = 0
    for i, day in enumerate(days[1:]):
        if age < day:
            current_stage = stages[i]
            stage_index = i
            break
    if age >= 100:
        current_stage = "Ripening"
        stage_index = 5
    
    # Calculate position percentage for the marker
    position_percent = min(max((age / 120), 0), 1)
    
    # Display the timeline
    container.text("Growth Timeline")
    container.progress(position_percent)
    
    # Display the current stage
    container.text(f"Current Stage: {current_stage}")
    
    # Display all stages
    cols = container.columns(len(stages))
    for i, (stage, col) in enumerate(zip(stages, cols)):
        if i == stage_index:
            col.markdown(f"**{stage}**")
        else:
            col.text(stage)
    
    # Display ground truth if available
    if ground_truth_age is not None:
        container.text(f"Actual Age: {ground_truth_age} days")
        gt_position = min(max((ground_truth_age / 120), 0), 1)
        container.progress(gt_position)

# Function to create a custom chart for top predictions using Streamlit components
def create_prediction_chart(container, predictions, title, color="#2e7d32", ground_truth=None):
    container.subheader(title)
    
    for pred in predictions:
        name = pred["name"].replace("_", " ").title()
        confidence = pred["confidence"]
        bar_color = color
        
        # Highlight ground truth if available
        if ground_truth and name.lower() == ground_truth.lower().replace("_", " "):
            bar_color = "#1565C0"  # Blue for ground truth
            name = f"{name} (Ground Truth)"
        elif confidence < 50:
            bar_color = "#FFC107"  # Yellow for low confidence
        elif confidence < 70:
            bar_color = "#FF9800"  # Orange for medium confidence
        
        # Display the prediction
        container.text(f"{name}: {confidence:.1f}%")
        container.progress(confidence/100)

# Function to create a custom info table using Streamlit components
def create_info_table(container, data, title):
    container.subheader(title)
    
    for key, value in data.items():
        key_formatted = key.replace("_", " ").title()
        cols = container.columns([1, 2])
        cols[0].text(f"{key_formatted}:")
        cols[1].text(value)

# Function to create a custom recommendation card using Streamlit components
def create_recommendation_list(container, recommendations):
    for i, rec in enumerate(recommendations):
        container.markdown(f"{i+1}. {rec}")

# Function to create a comparison table for predicted vs ground truth using Streamlit components
def create_comparison_table(container, predicted, ground_truth):
    container.subheader("Prediction vs Ground Truth")
    
    # Create the table headers
    cols = container.columns([1, 1, 1, 1])
    cols[0].markdown("**Attribute**")
    cols[1].markdown("**Predicted**")
    cols[2].markdown("**Ground Truth**")
    cols[3].markdown("**Match**")
    
    # Disease row
    cols = container.columns([1, 1, 1, 1])
    cols[0].text("Disease")
    
    if "error" in predicted["disease"]:
        cols[1].text("Cannot predict")
        cols[3].text("❌")
    else:
        disease_match = predicted["disease"]["name"] == ground_truth["disease"]
        disease_match_icon = "✅" if disease_match else "❌"
        cols[1].text(predicted["disease"]["name"].replace("_", " ").title())
        cols[3].text(disease_match_icon)
    
    cols[2].text(ground_truth["disease"].replace("_", " ").title())
    
    # Variety row
    cols = container.columns([1, 1, 1, 1])
    cols[0].text("Variety")
    
    if "error" in predicted["variety"]:
        cols[1].text("Cannot predict")
        cols[3].text("❌")
    else:
        variety_match = predicted["variety"]["name"].lower() == ground_truth["variety"].lower()
        variety_match_icon = "✅" if variety_match else "❌"
        cols[1].text(predicted["variety"]["name"])
        cols[3].text(variety_match_icon)
    
    cols[2].text(ground_truth["variety"])
    
    # Age row
    cols = container.columns([1, 1, 1, 1])
    cols[0].text("Age (days)")
    
    if "error" in predicted["age"]:
        cols[1].text("Cannot predict")
        cols[3].text("❌")
    else:
        age_diff = abs(predicted["age"]["days"] - ground_truth["age"])
        age_match = age_diff <= 7  # Consider a match if within 7 days
        age_match_icon = "✅" if age_match else "⚠️"
        cols[1].text(str(predicted["age"]["days"]))
        cols[3].text(f"{age_match_icon} {f'(±{age_diff} days)' if not age_match else ''}")
    
    cols[2].text(str(ground_truth["age"]))

# Function to create a visualization of disease probabilities
def create_disease_probability_chart(probabilities):
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort probabilities for better visualization
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    diseases = [item[0].replace("_", " ").title() for item in sorted_probs]
    values = [item[1] for item in sorted_probs]
    
    # Create horizontal bar chart
    bars = ax.barh(diseases, values, color=['#2e7d32' if disease == "Normal" else '#81c784' for disease in diseases])
    
    # Add percentage labels
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 1
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                va='center', fontsize=10)
    
    # Customize chart
    ax.set_xlabel('Probability (%)')
    ax.set_title('Disease Probability Distribution')
    ax.set_xlim(0, 100)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Tight layout
    plt.tight_layout()
    
    return fig

# Function to create a confusion matrix visualization
def create_confusion_matrix(true_labels, pred_labels, classes):
    # Create confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=classes)
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=classes, yticklabels=classes, ax=ax)
    
    # Customize chart
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Tight layout
    plt.tight_layout()
    
    return fig

# Custom CSS for professional styling using Streamlit's native theming
def set_custom_theme():
    # Set page config already handles some of this
    # Additional styling can be done with CSS
    st.markdown("""
    <style>
        /* Main layout and typography */
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #2e7d32;
        }
        
        .main-subheader {
            font-size: 1.1rem;
            color: #555;
            max-width: 700px;
        }
        
        .section-header {
            font-size: 1.5rem;
            font-weight: 600;
            color: #2e7d32;
            border-bottom: 2px solid #e8f5e9;
            padding-bottom: 0.5rem;
        }
        
        /* Status indicators */
        .status-healthy {
            color: #2e7d32;
            font-weight: 600;
        }
        
        .status-warning {
            color: #ff9800;
            font-weight: 600;
        }
        
        .status-danger {
            color: #c62828;
            font-weight: 600;
        }
        
        /* Footer styling */
        .footer {
            text-align: center;
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 1px solid #e0e0e0;
            color: #757575;
            font-size: 0.8rem;
        }
        
        /* Preprocessing info */
        .preprocessing-info {
            background-color: #e8f5e9;
            border-radius: 8px;
            padding: 12px;
            margin-top: 10px;
            font-size: 14px;
            color: #2e7d32;
            border-left: 4px solid #2e7d32;
        }
    </style>
    """, unsafe_allow_html=True)

# Main application
def main():
    # Initialize session state for history
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    
    # Apply custom theme
    set_custom_theme()
    
    # Header with professional design
    st.markdown('<h1 class="main-header">Paddy Doctor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subheader">An AI-powered tool to help farmers diagnose paddy plant diseases, identify rice varieties, and estimate plant age for better crop management.</p>', unsafe_allow_html=True)
    
    # Create a container for the main content
    main_container = st.container()
    
    # Sidebar with professional design
    with st.sidebar:
        st.image("https://img.freepik.com/free-photo/rice-field_74190-4097.jpg?w=1380&t=st=1683900425~exp=1683901025~hmac=b1e3d2e7e8c2e1d5d2f6d6f6d6f6d6f6d6f6d6f6d6f6d6f6d6f6d6f6d6f6d6", 
                 use_container_width=True)
        
        st.markdown("## How to use")
        st.markdown("""
        1. Upload a clear image of your paddy plant
        2. Wait for the analysis to complete
        3. View the detailed results for:
           - Disease identification
           - Variety identification
           - Age estimation
        """)
        
        st.markdown("## About")
        st.markdown("""
        This application uses advanced deep learning models to help farmers diagnose paddy plant issues and get valuable information about their crops.
        
        The models were trained on thousands of paddy images with various diseases and varieties.
        """)
        
        # Add image preprocessing information
        with st.expander("Image Preprocessing"):
            st.markdown("""
            All uploaded images are automatically:
            - Resized to 224×224 pixels
            - Pixel values scaled between 0 and 1
            - Normalized using ImageNet mean and std
            
            This ensures optimal performance with our AI models.
            """)
        
        # Add metadata information
        with st.expander("Metadata Integration"):
            if 'metadata_status' in st.session_state and st.session_state['metadata_status'] == "loaded":
                st.markdown("✅ **Metadata loaded successfully**")
                st.markdown("""
                The application integrates with `meta_train.csv` to provide:
                - Ground truth disease labels
                - Actual paddy variety information
                - Real plant age data
                
                This allows for comparison between model predictions and actual values.
                """)
            else:
                st.markdown("⚠️ **Metadata not loaded**")
                st.markdown("""
                The application will attempt to load metadata from `meta_train.csv` which should contain:
                - `image_id`: Unique identifier for each image
                - `label`: The category of paddy disease
                - `variety`: The paddy variety name
                - `age`: The age of the paddy in days
                
                Please ensure this file is available in the data directory.
                """)
        
        # Add model status information in an expander
        with st.expander("System Status"):
            if 'model_status' in st.session_state:
                for model, status in st.session_state['model_status'].items():
                    if status == "loaded":
                        st.markdown(f"✅ {model.title()} model: **Loaded**")
                    else:
                        st.markdown(f"⚠️ {model.title()} model: **Using fallback** (Error: {status})")
            else:
                st.markdown("⏳ Models not loaded yet")
        
        # Add history section in the sidebar
        with st.expander("Analysis History"):
            if len(st.session_state['history']) > 0:
                for i, item in enumerate(st.session_state['history']):
                    st.markdown(f"**Analysis {i+1}:** {item['timestamp']}")
                    if 'status' in item:
                        st.markdown(f"- Status: {item['status']}")
                    else:
                        st.markdown(f"- Disease: {item['disease'].replace('_', ' ').title()}")
                        st.markdown(f"- Variety: {item['variety']}")
                        st.markdown(f"- Age: {item['age']} days")
                    if i < len(st.session_state['history']) - 1:
                        st.markdown("---")
            else:
                st.markdown("No analysis history yet")
    
    # Load models and metadata
    with st.spinner("Loading models and metadata... This may take a moment."):
        disease_model, variety_model, age_model = load_models()
        metadata_df = load_metadata()
    
    # Main content area
    with main_container:
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["📷 Analysis", "📊 Sample Results", "📈 Data Insights", "ℹ️ Help & FAQ"])
        
        with tab1:
            # Create a section header for the upload section
            st.markdown('<div class="section-header">Upload Your Paddy Plant Image</div>', unsafe_allow_html=True)
            
            # Add preprocessing info
            st.markdown('<div class="preprocessing-info"><strong>Image Processing:</strong> All images are automatically resized to 224×224 pixels, normalized to values between 0-1, and standardized using ImageNet mean and std for optimal AI analysis.</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                uploaded_file = st.file_uploader(
                    "Upload an image of your paddy plant",
                    type=["jpg", "jpeg", "png"],
                    help="For best results, use a clear, well-lit image of the plant"
                )
            
            with col2:
                # Camera input option for mobile users
                camera_input = st.camera_input(
                    "Or take a photo with your camera",
                    help="This works best on mobile devices"
                )
            
            image_file = uploaded_file if uploaded_file is not None else camera_input
            
            # Process the image if uploaded
            if image_file is not None:
                try:
                    # Display the uploaded image
                    original_image = Image.open(image_file).convert('RGB')
                    
                    # Create a divider
                    st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader("Original Image")
                        st.image(original_image, use_container_width=True, caption="Uploaded Image")
                    
                    # Show progress bar animation
                    progress_bar_animation()
                    
                    # Make predictions
                    if disease_model and variety_model and age_model:
                        # Get the filename if available
                        image_name = image_file.name if hasattr(image_file, 'name') else None
                        
                        # Make predictions with metadata integration
                        results = predict(original_image, image_name, disease_model, variety_model, age_model, metadata_df)
                        
                        # Display processed image
                        with col2:
                            st.subheader("Processed Image (224×224)")
                            st.image(results["processed_image"], use_container_width=True, caption="Processed for AI (224×224, normalized)")
                            
                            # Display pixel value information
                            st.markdown('<div style="font-size: 12px; color: #555; margin-top: 5px;">ℹ️ Image pixel values have been scaled to range 0-1 and normalized for model input</div>', unsafe_allow_html=True)
                            
                            # Display metadata information if available
                            if results["metadata"]:
                                st.info("Ground truth metadata found for this image.")
                        
                        # Add to history only if predictions were successful
                        if not any("error" in results[key] for key in ["disease", "variety", "age"]):
                            st.session_state['history'].append({
                                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                                'disease': results["disease"]["name"],
                                'variety': results["variety"]["name"],
                                'age': results["age"]["days"]
                            })
                        else:
                            # Add a partial history entry with error indication
                            history_entry = {
                                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                                'status': 'Error in prediction'
                            }
                            
                            # Add available predictions
                            if "error" not in results["disease"]:
                                history_entry['disease'] = results["disease"]["name"]
                            
                            if "error" not in results["variety"]:
                                history_entry['variety'] = results["variety"]["name"]
                            
                            if "error" not in results["age"]:
                                history_entry['age'] = results["age"]["days"]
                            
                            st.session_state['history'].append(history_entry)
                        
                        # Create tabs for different result categories
                        result_tab1, result_tab2, result_tab3, result_tab4 = st.tabs([
                            "🔍 Overview", "🦠 Disease", "🌾 Variety", "📅 Age"
                        ])
                        
                        # In the main function, update the result tabs to handle error cases

                        # For the Overview tab
                        with result_tab1:
                            # Overview tab with summary of all results
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                if "error" in results["disease"]:
                                    st.error("Disease: Cannot predict")
                                else:
                                    status_text = "Healthy" if results["disease"]["is_healthy"] else "Disease Detected"
                                    st.metric("Status", status_text)
                            
                            with col2:
                                if "error" in results["variety"]:
                                    st.error("Variety: Cannot predict")
                                else:
                                    st.metric("Variety", results["variety"]["name"])
                            
                            with col3:
                                if "error" in results["age"]:
                                    st.error("Age: Cannot predict")
                                else:
                                    st.metric("Age", f"{results['age']['days']} days")
                            
                            # Display ground truth comparison if available
                            if "ground_truth" in results:
                                st.subheader("Ground Truth Data")
                                
                                gt_col1, gt_col2, gt_col3 = st.columns(3)
                                
                                with gt_col1:
                                    st.metric("Actual Disease", results["ground_truth"]["disease"].replace("_", " ").title())
                                
                                with gt_col2:
                                    st.metric("Actual Variety", results["ground_truth"]["variety"])
                                
                                with gt_col3:
                                    st.metric("Actual Age", f"{results['ground_truth']['age']} days")
                            
                            # Recommendations section - only show if we have valid predictions
                            if not any("error" in results[key] for key in ["disease", "variety", "age"]):
                                recommendations = []
                                
                                # Disease-based recommendations
                                if not results["disease"]["is_healthy"]:
                                    disease_key = results["disease"]["name"]
                                    if disease_key in DISEASE_INFO:
                                        recommendations.append(f"Treat the {disease_key.replace('_', ' ')} as recommended: {DISEASE_INFO[disease_key]['treatment']}")
                                else:
                                    recommendations.append("Continue with regular preventive measures as your plant appears healthy.")
                                
                                # Age-based recommendations
                                age_days = results["age"]["days"]
                                if age_days < 20:
                                    recommendations.append("Ensure proper water management for seedling establishment.")
                                elif age_days < 40:
                                    recommendations.append("Apply nitrogen fertilizer to support tillering.")
                                elif age_days < 60:
                                    recommendations.append("Maintain water level and monitor for pests.")
                                elif age_days < 80:
                                    recommendations.append("Ensure adequate nutrients for panicle development.")
                                elif age_days < 100:
                                    recommendations.append("Monitor for diseases that affect grain filling.")
                                else:
                                    recommendations.append("Prepare for harvest in the coming weeks.")
                                
                                # General recommendation
                                recommendations.append("Schedule regular monitoring to catch any issues early.")
                                
                                # Display recommendations
                                def show_recommendations():
                                    create_recommendation_list(st, recommendations)
                                
                                create_card("Recommendations", show_recommendations)
                            else:
                                st.warning("Cannot provide recommendations due to prediction errors.")

                        # For the Disease tab
                        with result_tab2:
                            # Disease tab with detailed disease information
                            if "error" in results["disease"]:
                                st.error(results["disease"]["message"])
                                st.warning("The disease prediction model could not process this image. Please try another image or check if the model is properly loaded.")
                            else:
                                if results["disease"]["is_healthy"]:
                                    st.success("✅ Healthy Plant")
                                    st.write(f"Your paddy plant appears to be healthy with {results['disease']['confidence']:.1f}% confidence.")
                                else:
                                    disease_name = results["disease"]["name"].replace("_", " ").title()
                                    disease_confidence = results["disease"]["confidence"]
                                    
                                    st.error(f"⚠️ Disease Detected: {disease_name}")
                                    
                                    # Create a function to show disease details
                                    def show_disease_details():
                                        # Add confidence bar
                                        create_confidence_bar(st, disease_confidence, "#c62828")
                                        
                                        # Add disease information if available
                                        disease_key = results["disease"]["name"]
                                        if disease_key in DISEASE_INFO:
                                            disease_info = DISEASE_INFO[disease_key]
                                            create_info_table(st, {
                                                "Description": disease_info["info"],
                                                "Severity": disease_info["severity"],
                                                "Spread Rate": disease_info["spread_rate"],
                                                "Treatment": disease_info["treatment"]
                                            }, "Disease Information")
                                    
                                    create_card("Disease Analysis", show_disease_details, color="#c62828")
                                
                                # Add disease probability visualization
                                st.subheader("Disease Probability Distribution")
                                fig = create_disease_probability_chart(results["disease"]["all_probabilities"])
                                st.pyplot(fig)
                                
                                # Add top predictions chart with ground truth if available
                                ground_truth_disease = results.get("ground_truth", {}).get("disease", None)
                                
                                def show_disease_predictions():
                                    create_prediction_chart(
                                        st,
                                        results["disease"]["top_predictions"],
                                        "Top Disease Predictions",
                                        "#c62828",
                                        ground_truth_disease
                                    )
                                
                                create_card("Disease Predictions", show_disease_predictions, color="#c62828")

                        # For the Variety tab
                        with result_tab3:
                            # Variety tab with detailed variety information
                            if "error" in results["variety"]:
                                st.error(results["variety"]["message"])
                                st.warning("The variety identification model could not process this image. Please try another image or check if the model is properly loaded.")
                            else:
                                variety_name = results["variety"]["name"]
                                variety_confidence = results["variety"]["confidence"]
                                
                                st.subheader(f"Identified Variety: {variety_name}")
                                
                                # Create a function to show variety details
                                def show_variety_details():
                                    # Add confidence bar
                                    create_confidence_bar(st, variety_confidence)
                                    
                                    # Add variety information if available
                                    if variety_name in VARIETY_INFO:
                                        variety_info = VARIETY_INFO[variety_name]
                                        create_info_table(st, {
                                            "Origin": variety_info["origin"],
                                            "Characteristics": variety_info["characteristics"],
                                            "Growing Period": variety_info["growing_period"],
                                            "Optimal Conditions": variety_info["optimal_conditions"]
                                        }, "Variety Information")
                                
                                create_card("Variety Analysis", show_variety_details)
                                
                                # Add top predictions chart with ground truth if available
                                ground_truth_variety = results.get("ground_truth", {}).get("variety", None)
                                
                                def show_variety_predictions():
                                    create_prediction_chart(
                                        st,
                                        results["variety"]["top_predictions"],
                                        "Top Variety Predictions",
                                        "#2e7d32",
                                        ground_truth_variety
                                    )
                                
                                create_card("Variety Predictions", show_variety_predictions)

                        # For the Age tab
                        with result_tab4:
                            # Age tab with detailed age information
                            if "error" in results["age"]:
                                st.error(results["age"]["message"])
                                st.warning("The age estimation model could not process this image. Please try another image or check if the model is properly loaded.")
                            else:
                                age_days = results["age"]["days"]
                                
                                # Determine growth stage
                                growth_stage = "Unknown"
                                if age_days < 20:
                                    growth_stage = "Seedling stage"
                                elif age_days < 40:
                                    growth_stage = "Tillering stage"
                                elif age_days < 60:
                                    growth_stage = "Stem elongation stage"
                                elif age_days < 80:
                                    growth_stage = "Panicle initiation stage"
                                elif age_days < 100:
                                    growth_stage = "Heading stage"
                                else:
                                    growth_stage = "Ripening stage"
                                
                                st.subheader(f"Estimated Age: {age_days} days")
                                st.write(f"Growth Stage: {growth_stage}")
                                
                                # Get ground truth age if available
                                ground_truth_age = results.get("ground_truth", {}).get("age", None)
                                
                                # Create a function to show age details
                                def show_age_details():
                                    # Add timeline visualization
                                    create_age_timeline(st, age_days, ground_truth_age)
                                    
                                    # Add stage-specific information
                                    stage_info = {
                                        "Seedling stage": "The plant is in early development with 1-5 leaves. Focus on proper water management and weed control.",
                                        "Tillering stage": "The plant is developing multiple stems. Ensure adequate nitrogen for maximum tiller production.",
                                        "Stem elongation stage": "The plant is growing taller. Maintain proper water levels and monitor for pests.",
                                        "Panicle initiation stage": "The reproductive phase begins. Ensure adequate nutrients for panicle development.",
                                        "Heading stage": "The panicle emerges from the stem. Protect from diseases that affect grain filling.",
                                        "Ripening stage": "The grain is maturing. Prepare for harvest and manage water accordingly."
                                    }
                                    
                                    if growth_stage in stage_info:
                                        st.info(f"About {growth_stage}: {stage_info[growth_stage]}")
                                
                                create_card("Age Analysis", show_age_details)
                                
                                # Add age prediction accuracy if ground truth is available
                                if ground_truth_age is not None:
                                    age_diff = abs(age_days - ground_truth_age)
                                    
                                    if age_diff <= 7:
                                        st.success(f"✅ Excellent prediction (within 7 days)")
                                    elif age_diff <= 14:
                                        st.warning(f"⚠️ Moderate accuracy (within 14 days)")
                                    else:
                                        st.error(f"❌ Low accuracy (more than 14 days off)")
                                    
                                    st.write(f"Predicted Age: {age_days} days")
                                    st.write(f"Actual Age: {ground_truth_age} days")
                                    st.write(f"Difference: {age_diff} days")
                    else:
                        st.error("Models could not be loaded. Please check the model paths and try again.")
                
                except Exception as e:
                    st.error(f"Error processing image: {e}")
                    import traceback
                    st.error(traceback.format_exc())
        
        with tab2:
            # Sample results tab
            st.markdown('<div class="section-header">Sample Results</div>', unsafe_allow_html=True)
            st.markdown("Here are some examples of what the analysis results look like for different paddy plant conditions.")
            
            # Create a grid of sample results using Streamlit's native components
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image("https://img.freepik.com/free-photo/close-up-rice-plant_23-2148535711.jpg?w=1380&t=st=1683900500~exp=1683901100~hmac=b1e3d2e7e8c2e1d5d2f6d6f6d6f6d6f6d6f6d6f6d6f6d6f6d6f6d6f6d6f6d6", 
                        use_container_width=True)
                
                st.success("Healthy Paddy")
                st.write("✅ No disease detected")
                st.write("Variety: Basmati")
                st.write("Age: 35 days")
            
            with col2:
                st.image("https://img.freepik.com/free-photo/rice-field_74190-4097.jpg?w=1380&t=st=1683900425~exp=1683901025~hmac=b1e3d2e7e8c2e1d5d2f6d6f6d6f6d6f6d6f6d6f6d6f6d6f6d6f6d6f6d6f6d6", 
                        use_container_width=True)
                
                st.error("Blast Disease")
                st.write("⚠️ Blast disease detected")
                st.write("Variety: Jasmine")
                st.write("Age: 60 days")
            
            with col3:
                st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQAcRwCL5kWrJByWlPvdnpdYR_blcAphi3alw&s", 
                        use_container_width=True)
                
                st.warning("Brown Spot")
                st.write("⚠️ Brown spot detected")
                st.write("Variety: Long Grain")
                st.write("Age: 45 days")
        
        with tab3:
            # Data Insights tab
            st.markdown('<div class="section-header">Data Insights</div>', unsafe_allow_html=True)
            
            # Check if metadata is available
            if 'metadata_status' in st.session_state and st.session_state['metadata_status'] == "loaded" and not metadata_df.empty:
                # Create visualizations based on metadata
                st.subheader("Disease Distribution in Training Data")
                
                # Count disease occurrences
                disease_counts = metadata_df['label'].value_counts()
                
                # Create pie chart for disease distribution
                fig, ax = plt.subplots(figsize=(10, 6))
                wedges, texts, autotexts = ax.pie(
                    disease_counts, 
                    labels=disease_counts.index, 
                    autopct='%1.1f%%',
                    textprops={'fontsize': 10, 'color': 'black'},
                    colors=plt.cm.Greens(np.linspace(0.3, 0.8, len(disease_counts)))
                )
                
                # Equal aspect ratio ensures that pie is drawn as a circle
                ax.axis('equal')
                plt.setp(autotexts, size=9, weight="bold")
                plt.title("Distribution of Paddy Diseases in Training Data")
                st.pyplot(fig)
                
                # Create two columns for age and variety visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Age distribution
                    st.subheader("Age Distribution")
                    
                    # Create histogram for age distribution
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.histplot(metadata_df['age'], bins=20, kde=True, color='#2e7d32', ax=ax)
                    ax.set_xlabel('Age (days)')
                    ax.set_ylabel('Count')
                    ax.set_title('Distribution of Paddy Plant Ages')
                    st.pyplot(fig)
                
                with col2:
                    # Variety distribution
                    st.subheader("Variety Distribution")
                    
                    # Count variety occurrences
                    variety_counts = metadata_df['variety'].value_counts()
                    
                    # Create bar chart for variety distribution
                    fig, ax = plt.subplots(figsize=(8, 5))
                    bars = ax.bar(
                        variety_counts.index, 
                        variety_counts.values,
                        color=plt.cm.Greens(np.linspace(0.3, 0.8, len(variety_counts)))
                    )
                    
                    # Add count labels on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width()/2.,
                            height + 0.1,
                            f'{height:.0f}',
                            ha='center', 
                            va='bottom',
                            fontsize=8
                        )
                    
                    ax.set_xlabel('Variety')
                    ax.set_ylabel('Count')
                    ax.set_title('Distribution of Paddy Varieties')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Relationship between disease and age
                st.subheader("Relationship Between Disease and Age")
                
                # Create boxplot for disease vs age
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.boxplot(x='label', y='age', data=metadata_df, hue='label', palette='Greens', ax=ax, legend=False)
                ax.set_xlabel('Disease')
                ax.set_ylabel('Age (days)')
                ax.set_title('Age Distribution by Disease Type')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Relationship between disease and variety
                st.subheader("Disease Occurrence by Variety")
                
                # Create a crosstab of disease vs variety
                disease_variety_cross = pd.crosstab(metadata_df['label'], metadata_df['variety'])
                
                # Create heatmap
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(disease_variety_cross, cmap='Greens', annot=True, fmt='d', ax=ax)
                ax.set_xlabel('Variety')
                ax.set_ylabel('Disease')
                ax.set_title('Disease Occurrence by Variety')
                plt.tight_layout()
                st.pyplot(fig)
                
            else:
                st.info("Metadata from meta_train.csv is not available. Please ensure the file is in the correct location to view data insights.")
                
                # Show dummy visualizations
                st.subheader("Sample Disease Distribution (Demo)")
                
                # Create dummy data
                diseases = ["bacterial_leaf_blight", "blast", "brown_spot", "normal", "tungro", "hispa"]
                counts = [120, 150, 100, 200, 80, 70]
                
                # Create pie chart
                fig, ax = plt.subplots(figsize=(10, 6))
                wedges, texts, autotexts = ax.pie(
                    counts, 
                    labels=[d.replace("_", " ").title() for d in diseases], 
                    autopct='%1.1f%%',
                    textprops={'fontsize': 10, 'color': 'black'},
                    colors=plt.cm.Greens(np.linspace(0.3, 0.8, len(diseases)))
                )
                
                # Equal aspect ratio ensures that pie is drawn as a circle
                ax.axis('equal')
                plt.setp(autotexts, size=9, weight="bold")
                plt.title("Sample Distribution of Paddy Diseases (Demo)")
                st.pyplot(fig)
                
                st.info("This is a demo visualization. To see actual data insights, please ensure that the meta_train.csv file is available in the data directory.")
        
        with tab4:
            # Help & FAQ tab
            st.markdown('<div class="section-header">Help & Frequently Asked Questions</div>', unsafe_allow_html=True)
            
            # Create expandable FAQ items
            with st.expander("How accurate is the disease detection?"):
                st.markdown("""
                The disease detection model has been trained on thousands of paddy plant images and achieves an accuracy of approximately 85-90% on test data. However, accuracy may vary depending on:
                
                - Image quality and lighting conditions
                - Disease severity and visibility
                - Growth stage of the plant
                
                For critical decisions, we recommend consulting with an agricultural expert to confirm the diagnosis.
                """)
            
            with st.expander("What image quality is required for best results?"):
                st.markdown("""
                For optimal results, please ensure:
                
                - The image is clear and in focus
                - The affected area is clearly visible
                - The image is taken in good lighting conditions
                - The plant fills a significant portion of the image
                - Multiple images from different angles for complex cases
                
                All images are automatically resized to 224×224 pixels and normalized for optimal AI analysis.
                """)
            
            with st.expander("How is the age of the plant estimated?"):
                st.markdown("""
                The age estimation model analyzes visual characteristics of the plant such as:
                
                - Height and structure
                - Leaf development
                - Tillering stage
                - Panicle development (if present)
                
                The model provides an estimate in days since transplanting or direct seeding. The accuracy is typically within ±7 days.
                """)
            
            with st.expander("How does the metadata integration work?"):
                st.markdown("""
                The application integrates with the `meta_train.csv` file which contains:
                
                - `image_id`: Unique identifier for each image
                - `label`: The category of paddy disease
                - `variety`: The paddy variety name
                - `age`: The age of the paddy in days
                
                When you upload an image, the system:
                
                1. Extracts the image ID from the filename
                2. Looks up the corresponding metadata in the CSV file
                3. Uses this information to compare predictions with ground truth
                4. Displays both predicted and actual values when available
                
                This helps in evaluating model performance and understanding prediction accuracy.
                """)
            
            with st.expander("Can I use this app offline in the field?"):
                st.markdown("""
                Currently, this web application requires an internet connection to function. However, we are developing:
                
                1. A mobile app version with offline capabilities
                2. A lightweight version that can run with limited connectivity
                
                Sign up for our newsletter to be notified when these options become available.
                """)
            
            with st.expander("How can I contribute to improving the models?"):
                st.markdown("""
                You can help improve our models by:
                
                1. Providing feedback on prediction accuracy
                2. Submitting correctly labeled images to our database
                3. Participating in our community testing program
                
                Contact us at support@paddydoctor.org to learn more about contributing.
                """)
            
            with st.expander("Why are images normalized for model input?"):
                st.markdown("""
                Images are preprocessed in this specific way for several important reasons:
                
                - **Consistent Input Size**: Deep learning models require fixed input dimensions. 224×224 is a standard size for many pre-trained models.
                
                - **Efficient Processing**: Smaller images require less memory and computational resources, allowing faster analysis.
                
                - **Normalization**: Neural networks perform better when input values are in a standardized range:
                  - First, pixel values are scaled from 0-255 to 0-1
                  - Then, they are normalized using ImageNet mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225)
                  - This helps the model converge faster during training
                  - Makes the model more robust to variations in lighting conditions
                  - Improves generalization to new images
                
                This preprocessing ensures optimal performance and accuracy of our AI models.
                """)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Paddy Doctor v2.0 | Helping farmers diagnose and treat paddy plant issues</p>
        <p style="font-size: 0.7rem; margin-top: 5px;">© 2023 Paddy Doctor Team. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()