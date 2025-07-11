# --- START OF FILE app.py ---

import os
import io
import base64
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
import numpy as np
import timm # Import timm
from flask import Flask, request, render_template, redirect, url_for, flash
from collections import OrderedDict # For probability sorting if needed

# --- Configuration ---
MODEL_SAVE_PATH = 'model/best_typhoon_model_vitsmall_grade_cinumber_only.pth'
IMG_SIZE = 224
SCALE_FACTOR = 10.0 # CInumber scaling (must match training)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Grade Mapping (CRITICAL: MUST match the one used during training!) ---
# Updated with the map from your training log (2025-04-05 20:55:30)
GRADE_MAP = {
    'Severe Tropical Storm (STS)': 0,
    'Tropical Depression (TD)': 1,
    'Tropical Storm (TS)': 2,
    'Typhoon (TY)': 3
}

# --- Automatically derive variables from GRADE_MAP ---
CLASS_NAMES = list(GRADE_MAP.keys())
NUM_GRADES = len(GRADE_MAP) # This will now correctly be 4
# Create an inverse map for easy lookup from index to name
INV_GRADE_MAP = {v: k for k, v in GRADE_MAP.items()}

UPLOAD_FOLDER = 'static/uploads' # Optional: If saving uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# --- Flask App Setup ---
app = Flask(__name__)
app.secret_key = os.urandom(24) # Needed for flash messages
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER # Optional
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True) # Optional

# --- Model Class Definitions (Copy EXACTLY from training script) ---
class CINumberHead(nn.Module):
    """Regression head for CInumber, taking flat features."""
    def __init__(self, in_features, num_outputs=1):
        super().__init__()
        # ViT output before head is typically [B, embed_dim], no pooling needed
        self.dropout = nn.Dropout(0.4) # Keep dropout
        self.fc = nn.Linear(in_features, num_outputs)

    def forward(self, x):
        # Input x is assumed to be [B, in_features]
        x = self.dropout(x)
        x = self.fc(x)
        return x

class ViTMultiTask(nn.Module):
    """ViT-Small based model for Grade classification and CInumber regression."""
    def __init__(self, num_grades, vit_model_name='vit_small_patch16_224'):
        super().__init__()
        # Load ViT Small structure WITHOUT pre-trained weights here,
        # as we will load our trained state_dict later.
        self.backbone = timm.create_model(vit_model_name, pretrained=False)

        # Get the feature dimension (embedding size) from the ViT model
        feature_dim = self.backbone.head.in_features # Access the input feature dim of the original head

        # Remove the original classification head of the ViT model
        self.backbone.head = nn.Identity()

        # Head for CINumber (regression) - takes ViT features directly
        self.cinumber_head = CINumberHead(feature_dim, 1) # Use the simplified head

        # Head for Grade classification
        concatenated_feature_dim = feature_dim + 1 # ViT features + 1 (CInum prediction)
        self.grade_head = nn.Sequential(
            nn.LayerNorm(concatenated_feature_dim),
            nn.Dropout(0.5),
            nn.Linear(concatenated_feature_dim, num_grades) # Final classification layer uses num_grades
        )

    def forward(self, x):
        # Backbone feature extraction
        features = self.backbone(x) # Output: [B, feature_dim]

        # CInumber Prediction
        out_cinumber = self.cinumber_head(features) # Output: [B, 1]

        # Concatenate main features and auxiliary output
        concatenated_features = torch.cat([features, out_cinumber], dim=1) # Output: [B, feature_dim + 1]

        # Grade Prediction
        out_grade = self.grade_head(concatenated_features) # Output: [B, num_grades]

        outputs = {'grade': out_grade, 'cinumber': out_cinumber}
        return outputs

# --- Model Loading ---
def load_model(model_path, num_grades_to_load, device):
    """Loads the pre-trained model."""
    print(f"Attempting to load model for {num_grades_to_load} classes from: {model_path}")
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return None
    try:
        # Instantiate the model structure with the correct number of output grades
        model_instance = ViTMultiTask(num_grades=num_grades_to_load).to(device)
        # Load the state dict - use map_location for flexibility
        state_dict = torch.load(model_path, map_location=device)

        # --- Optional: Inspect keys if loading fails ---
        # print("Keys in loaded state_dict:", state_dict.keys())
        # print("Keys in model structure:", model_instance.state_dict().keys())
        # ---

        model_instance.load_state_dict(state_dict)
        model_instance.eval() # Set to evaluation mode
        print(f"Model loaded successfully onto {device}.")
        return model_instance
    except RuntimeError as e:
        print(f"ERROR loading state_dict (likely architecture mismatch): {e}")
        print("Check if NUM_GRADES derived from GRADE_MAP matches the model checkpoint.")
        return None
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return None

# Load the model globally when the app starts, using NUM_GRADES from the (corrected) GRADE_MAP
model = load_model(MODEL_SAVE_PATH, NUM_GRADES, DEVICE)
if model is None:
     print("CRITICAL: Failed to load the model. The application predictions will not work.")
     # Depending on your needs, you might want to raise an exception here
     # raise RuntimeError("Could not load the prediction model.")

# --- Image Transformation (Use Validation/Test transforms from training) ---
val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    # Use the same normalization as training!
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # Or if you used timm's defaults during training:
    # transforms.Normalize(mean=timm.data.IMAGENET_DEFAULT_MEAN, std=timm.data.IMAGENET_DEFAULT_STD)
])

def transform_image(image_bytes):
    """Transforms raw image bytes into a model-ready tensor."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return val_test_transform(image).unsqueeze(0) # Add batch dimension [1, C, H, W]
    except UnidentifiedImageError:
        print("ERROR: Cannot identify image file (corrupted or invalid format).")
        return None
    except Exception as e:
        print(f"ERROR transforming image: {e}")
        return None

def get_prediction(image_tensor):
    """Performs inference using the loaded model."""
    if model is None:
        print("ERROR: Model is not loaded, cannot make prediction.")
        # Return distinct values to indicate failure
        return "Error: Model not loaded", "N/A", None

    try:
        image_tensor = image_tensor.to(DEVICE)
        with torch.no_grad():
            outputs = model(image_tensor)

        # Process Grade Prediction
        grade_logits = outputs['grade']
        # Apply softmax to logits to get probabilities for the single image in the batch
        grade_probs = torch.softmax(grade_logits, dim=1)[0]
        # Get the index of the highest probability
        predicted_grade_idx = torch.argmax(grade_probs).item()
        # Map the index back to the class name using the inverse map
        predicted_grade_name = INV_GRADE_MAP.get(predicted_grade_idx, f"Unknown Index {predicted_grade_idx}")

        # Create a dictionary of class probabilities, sorted for clarity (optional)
        probabilities = {CLASS_NAMES[i]: prob.item() for i, prob in enumerate(grade_probs)}
        # Sort by probability descending (optional)
        # sorted_probabilities = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))
        # Format as percentages
        formatted_probabilities = {k: f"{v*100:.2f}%" for k, v in probabilities.items()}


        # Process CInumber Prediction
        # Output is likely [1, 1], get the scalar value
        predicted_cinumber_scaled = outputs['cinumber'].item()
        # Scale back to the original range
        predicted_cinumber = predicted_cinumber_scaled * SCALE_FACTOR

        return predicted_grade_name, f"{predicted_cinumber:.2f}", formatted_probabilities

    except Exception as e:
        print(f"ERROR during prediction: {e}")
        # Return distinct values to indicate failure
        return "Error during prediction", "N/A", None

def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    # Pass class names to the template for display
    display_class_names = CLASS_NAMES if CLASS_NAMES else ["N/A - Check GRADE_MAP"]

    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part selected', 'error')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename.
        if file.filename == '':
            flash('No file selected', 'warning')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            try:
                img_bytes = file.read() # Read file content as bytes
                # Transform image
                tensor = transform_image(img_bytes)

                if tensor is None:
                    flash('Could not process image file (invalid format or corrupted?).', 'error')
                    return redirect(request.url)

                # Get prediction
                grade, cinumber, probabilities = get_prediction(tensor)

                if grade.startswith("Error:") or probabilities is None:
                     flash(f'Prediction failed: {grade}. Check server logs.', 'error')
                     return redirect(request.url)

                # Encode image bytes to Base64 to display on the result page
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')

                # Render result template, passing the predictions and image data
                return render_template('result.html',
                                       grade=grade,
                                       cinumber=cinumber,
                                       probabilities=probabilities,
                                       img_data=img_base64)

            except Exception as e:
                flash(f'An unexpected error occurred: {e}', 'error')
                print(f"ERROR in POST request handling: {e}") # Log detailed error server-side
                # import traceback # Optional: for detailed debugging
                # print(traceback.format_exc())
                return redirect(request.url)

        else:
            flash(f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}', 'error')
            return redirect(request.url)

    # GET request: just show the upload form
    return render_template('index.html', class_names=display_class_names)

# --- Run the App ---
if __name__ == '__main__':
    print("-" * 50)
    print(f"Starting Flask app...")
    print(f"Model Path: {MODEL_SAVE_PATH}")
    print(f"Using GRADE_MAP: {GRADE_MAP}") # Print the map being used
    print(f"Model Loaded: {'Yes' if model else 'No'}")
    print(f"Running on Device: {DEVICE}")
    print(f"Expected Grade Classes ({NUM_GRADES}): {CLASS_NAMES}")
    print(f"CInumber Scale Factor: {SCALE_FACTOR}")
    print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print("-" * 50)
    # Use host='0.0.0.0' to make it accessible on your local network
    # Use debug=True only for development, set to False for production
    app.run(debug=True, host='0.0.0.0', port=5000)

# --- END OF FILE app.py ---