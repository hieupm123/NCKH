# --- START OF FILE ResNet18_DvorakNoPattern.py ---

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF # Import functional transforms
import torchvision.models as models
import pandas as pd
from PIL import Image
import os
import numpy as np
import io
import time
import random
from multiprocessing import freeze_support # Optional but good practice for Windows packaging

# Import for metrics and plotting
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns # For prettier confusion matrix

# --- Constants and Global Definitions ---
# Log and Model Paths
LOG_DIR = "LLog"
METRICS_DIR = "ResNet18_Grade" # Can be the same or different
MODEL_DIR = "model"
LOG_FILE_PATH = os.path.join(LOG_DIR, "training_log_Resnet18_GradeOnly.txt")
METRICS_SAVE_PATH = os.path.join(METRICS_DIR, "evaluation_metrics_Resnet18_GradeOnly.txt")
CONFUSION_MATRIX_VAL_PATH = os.path.join(METRICS_DIR, "confusion_matrix_val_Resnet18_GradeOnly.png")
CONFUSION_MATRIX_TEST_PATH = os.path.join(METRICS_DIR, "confusion_matrix_test_Resnet18_GradeOnly.png")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'best_typhoon_model_resnet18_gradeonly.pth')

# CSV Paths
TRAIN_CSV_PATH = 'dataset/train_csv_4.csv'
VAL_CSV_PATH = 'dataset/val_csv_4.csv'
TEST_CSV_PATH = 'dataset/test_csv_4.csv'

# Hyperparameters
IMG_SIZE = 224
LEARNING_RATE = 5e-4
BATCH_SIZE = 64
EPOCHS = 50 # Adjust as needed
WEIGHT_DECAY = 1e-5
SCHEDULER_PATIENCE = 40
SCHEDULER_FACTOR = 0.2
EARLY_STOPPING_PATIENCE = 40
W_GRADE = 1.0 # Single task weight

# --- Helper Functions ---

def ensure_dir_exists(path):
    """Ensures the directory for a given file path exists."""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
        except OSError as e:
            print(f"Error creating directory {directory}: {e}")
            # Depending on severity, you might want to raise the error
            # raise

# Ensure directories exist at the start
ensure_dir_exists(LOG_FILE_PATH)
ensure_dir_exists(METRICS_SAVE_PATH) # Also ensures METRICS_DIR
ensure_dir_exists(CONFUSION_MATRIX_VAL_PATH) # Redundant if METRICS_DIR is the same, but safe
ensure_dir_exists(CONFUSION_MATRIX_TEST_PATH) # Redundant if METRICS_DIR is the same, but safe
ensure_dir_exists(MODEL_SAVE_PATH) # Also ensures MODEL_DIR

# --- Logging Function ---
def write_log(message):
    """Writes a message to the console and the log file."""
    print(message) # Print to console
    try:
        with open(LOG_FILE_PATH, "a", encoding='utf-8') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n") # Write to file
    except Exception as e:
        print(f"ERROR: Could not write to log file {LOG_FILE_PATH}. Error: {e}")

# --- Safe CSV Loading Function ---
def load_csv_safe(path, name):
    """Loads a CSV file safely, checking for existence and 'Grade' column."""
    try:
        df = pd.read_csv(path)
        write_log(f"Successfully loaded {name} data from: {path}. Samples: {len(df)}")
        if 'Grade' not in df.columns:
            write_log(f"ERROR: 'Grade' column missing in {name} data at {path}")
            return None
        # Check for empty dataframe after loading
        if df.empty:
             write_log(f"WARNING: Loaded dataframe '{name}' from {path} is empty.")
             return df # Return empty dataframe to allow checks later
        return df
    except FileNotFoundError:
        write_log(f"ERROR: CSV file not found at {path}")
        return None
    except pd.errors.EmptyDataError:
        write_log(f"ERROR: CSV file at {path} is empty.")
        # Return an empty DataFrame consistent with other empty cases
        return pd.DataFrame()
    except Exception as e:
        write_log(f"ERROR: Failed to load CSV {path}. Error: {e}")
        return None

# --- Functions and Class Definitions ---

# <<< FUNCTION FOR RANDOM DISCRETE ROTATION TRANSFORM >>>
def random_discrete_rotation(img):
    """Applies a random rotation of 0, 90, 180, or 270 degrees."""
    angle = random.choice([0, 90, 180, 270])
    return TF.rotate(img, angle)

# --- Dataset Class ---
class TyphoonDataset(Dataset):
    def __init__(self, dataframe, transform, grade_map):
        """Initializes the Typhoon Dataset."""
        # Check if dataframe is None or not a DataFrame before proceeding
        if dataframe is None or not isinstance(dataframe, pd.DataFrame):
             raise ValueError("Input dataframe is None or not a pandas DataFrame.")

        self.dataframe = dataframe
        self.transform = transform
        self.grade_map = grade_map # Store grade_map

        # Only perform checks if the dataframe is not empty
        if not self.dataframe.empty:
            # Check essential columns exist
            if 'file_name' not in self.dataframe.columns:
                raise ValueError("DataFrame is missing the 'file_name' column.")
            if 'Grade' not in self.dataframe.columns:
                 raise ValueError("DataFrame is missing the 'Grade' column required for mapping.")
            self._check_mappings()
        else:
             write_log("WARNING: Initializing TyphoonDataset with an empty DataFrame.")


    def _check_mappings(self):
        """Checks if all grades in the dataframe exist in the grade_map."""
        # Filter out potential NaN values in 'Grade' before checking against the map
        valid_grades = self.dataframe.dropna(subset=['Grade'])
        missing_grades = valid_grades[~valid_grades['Grade'].isin(self.grade_map.keys())]['Grade'].unique()
        if len(missing_grades) > 0:
            # Log the problematic grades and map for easier debugging
            write_log(f"ERROR: Found grades in dataframe not present in GRADE_MAP!")
            write_log(f"Missing Grades: {missing_grades}")
            write_log(f"Available GRADE_MAP Keys: {list(self.grade_map.keys())}")
            raise ValueError(f"Found grades in dataframe not present in GRADE_MAP: {missing_grades}")

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.dataframe)

    def __getitem__(self, idx):
        """Gets a sample (image and label) from the dataset at the given index."""
        if self.dataframe.empty:
             raise IndexError("Cannot get item from an empty dataset")
        if not isinstance(idx, int):
             idx = idx.item() # Handle tensor indices if they occur
        if idx < 0 or idx >= len(self.dataframe):
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self.dataframe)}")

        try:
            row = self.dataframe.iloc[idx]
            img_path = row['file_name']
            grade_value = row['Grade']
        except IndexError:
             write_log(f"ERROR: Internal IndexError accessing iloc[{idx}] with dataframe length {len(self.dataframe)}. Data might be inconsistent.")
             raise # Re-raise to signal a problem
        except KeyError as e:
             write_log(f"ERROR: Missing expected column '{e}' at row index {idx}.")
             raise # Re-raise critical error
        except Exception as e:
             write_log(f"ERROR: Failed to access row {idx} in dataframe. Error: {e}")
             raise # Re-raise error

        # Load Image
        try:
            # Ensure img_path is a string
            if not isinstance(img_path, str) or not img_path:
                 raise ValueError(f"Invalid image path at index {idx}: {img_path}")
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            write_log(f"ERROR: Image file not found at {img_path} (index {idx}).")
            # Decide how to handle: raise error, return None, return dummy data?
            # Raising is often best during development to catch missing files.
            raise FileNotFoundError(f"Image file not found: {img_path}")
        except Exception as e:
             write_log(f"ERROR: Failed to load image {img_path} (index {idx}). Error: {e}.")
             raise # Re-raise error

        # Apply Transforms
        try:
            if self.transform: # Apply transform if provided
                image = self.transform(image)
            else:
                 # If no transform, ensure image is a tensor if needed downstream
                 # This might depend on your model's expected input
                 image = TF.to_tensor(image) # Basic conversion if no other transform
        except Exception as e:
             write_log(f"ERROR: Failed to transform image {img_path} (index {idx}). Error: {e}.")
             raise # Re-raise error

        # Process Label
        try:
            if pd.isna(grade_value):
                raise ValueError(f"Grade is NaN for image {img_path} (index {idx})")
            # Ensure grade_value is hashable (e.g., string, number) for dict lookup
            if not isinstance(grade_value, (str, int, float)):
                 grade_value = str(grade_value) # Attempt conversion if needed

            grade = torch.tensor(self.grade_map[grade_value], dtype=torch.long)
        except KeyError:
            write_log(f"ERROR: Grade '{grade_value}' from image {img_path} (index {idx}) not found in GRADE_MAP: {self.grade_map}.")
            raise KeyError(f"Grade '{grade_value}' not in GRADE_MAP.") # Re-raise critical error
        except Exception as e:
            write_log(f"ERROR: Failed to process label for image {img_path} (index {idx}). Error: {e}")
            raise # Re-raise error

        return {
            'image': image,
            'labels': {'grade': grade}
        }


# --- Model Class ---
class ResNetGradeOnly(nn.Module):
    """ResNet-18 model modified for Grade classification only."""
    def __init__(self, num_grades):
        super().__init__()
        # Load ResNet18 with pre-trained weights
        base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Reuse layers from the base model
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool
        self.flatten = nn.Flatten()

        # Get the number of features output by the ResNet base
        final_feature_dim = base_model.fc.in_features

        # Define the classification head for Grade prediction
        self.grade_head = nn.Sequential(
            nn.Dropout(0.5), # Add dropout for regularization
            nn.Linear(final_feature_dim, num_grades) # Output layer for grades
        )

    def forward(self, x):
        """Defines the forward pass of the model."""
        # Pass input through the base model layers
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.flatten(x) # Flatten the features

        # Pass features through the grade classification head
        out_grade = self.grade_head(x)

        # Return predictions in a dictionary format
        outputs = {'grade': out_grade}
        return outputs


# --- Loss Calculation Function ---
def calculate_simplified_loss(outputs, labels, criterion_clf):
    """Calculates the loss only for the grade task using the provided criterion."""
    try:
        # Ensure labels['grade'] is on the correct device and has the right dtype
        target_grades = labels['grade'].long() # Ensure long type for CrossEntropy

        # Check if outputs dict contains the 'grade' key
        if 'grade' not in outputs:
            write_log("ERROR: 'grade' key missing in model outputs during loss calculation.")
            return torch.tensor(float('inf')), {'total': float('inf'), 'grade': float('inf')}

        output_grades = outputs['grade']

        # Basic shape check (optional but helpful)
        if output_grades.shape[0] != target_grades.shape[0]:
             write_log(f"ERROR: Mismatch in batch size between outputs ({output_grades.shape[0]}) and labels ({target_grades.shape[0]})")
             return torch.tensor(float('inf')), {'total': float('inf'), 'grade': float('inf')}


        loss_grade = criterion_clf(output_grades, target_grades)
        total_loss = loss_grade # Only one task

        # Check for NaN/Inf loss right after calculation
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            write_log("WARNING: NaN or Inf detected in loss calculation.")
            # Optionally add more debugging info here:
            # write_log(f"Outputs grade shape: {output_grades.shape}")
            # write_log(f"Labels grade shape: {target_grades.shape}")
            # write_log(f"Sample Outputs grade: {output_grades[0]}") # Log sample output
            # write_log(f"Sample Labels grade: {target_grades[0]}") # Log sample label
            # Return a large but finite loss to avoid breaking loops immediately, or NaN if preferred
            return torch.tensor(float('nan')), {'total': float('nan'), 'grade': float('nan')}

        individual_losses = {
            'total': total_loss.item(), # .item() gets scalar value
            'grade': loss_grade.item(),
        }
        return total_loss, individual_losses
    except KeyError as e:
        write_log(f"ERROR: Missing key '{e}' in labels dictionary during loss calculation.")
        return torch.tensor(float('nan')), {'total': float('nan'), 'grade': float('nan')}
    except Exception as e:
        write_log(f"ERROR: Unexpected error during loss calculation: {e}")
        return torch.tensor(float('nan')), {'total': float('nan'), 'grade': float('nan')}


# --- Evaluation Function ---
def evaluate_grade_only(model, data_loader, device, criterion_clf, dataset_name="Evaluation"):
    """Evaluates the model on a given dataset, returning metrics and predictions."""
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    correct_grades = 0
    total_samples = 0
    eval_indiv_losses = {'total': 0.0, 'grade': 0.0}
    all_labels_list = [] # To store all true labels
    all_preds_list = []  # To store all predictions
    num_batches_processed = 0 # Keep track of valid batches processed

    # --- Initial Checks ---
    if not isinstance(model, nn.Module):
        write_log(f"ERROR ({dataset_name}): Invalid model object passed.")
        return float('nan'), 0.0, {'grade': float('nan')}, [], []
    if not isinstance(data_loader, DataLoader):
        write_log(f"ERROR ({dataset_name}): Invalid DataLoader object passed.")
        return float('nan'), 0.0, {'grade': float('nan')}, [], []
    if not hasattr(data_loader, 'dataset') or data_loader.dataset is None:
        write_log(f"ERROR ({dataset_name}): DataLoader has no associated dataset.")
        return float('nan'), 0.0, {'grade': float('nan')}, [], []

    try:
        num_samples_in_dataset = len(data_loader.dataset)
    except Exception as e:
        write_log(f"ERROR ({dataset_name}): Could not get length of dataset. Error: {e}")
        return float('nan'), 0.0, {'grade': float('nan')}, [], []

    if num_samples_in_dataset == 0:
        write_log(f"\n--- Evaluating on {dataset_name} set ---")
        write_log(f"Skipping evaluation: {dataset_name} dataset is empty.")
        return float('nan'), 0.0, {'grade': float('nan')}, [], []

    write_log(f"\n--- Evaluating on {dataset_name} set ({num_samples_in_dataset} samples) ---")

    # --- Evaluation Loop ---
    with torch.no_grad(): # Disable gradient calculations
        for i, batch in enumerate(data_loader):
            # --- Batch Data Handling ---
            try:
                # Basic check for batch structure
                if not isinstance(batch, dict) or 'image' not in batch or 'labels' not in batch or 'grade' not in batch['labels']:
                     write_log(f"WARNING ({dataset_name}): Invalid batch structure at index {i}. Skipping batch.")
                     continue

                images = batch['image'].to(device, non_blocking=True)
                # Ensure labels are correctly extracted and moved
                labels_grade = batch['labels']['grade'].to(device, non_blocking=True, dtype=torch.long)
                labels = {'grade': labels_grade} # Reconstruct labels dict for consistency

                # Check for empty tensors
                if images.nelement() == 0 or labels['grade'].nelement() == 0:
                    write_log(f"WARNING ({dataset_name}): Empty image or label tensor in batch {i}. Skipping.")
                    continue

            except KeyError as e:
                 write_log(f"ERROR ({dataset_name}) processing batch {i}: Missing key {e}. Check DataLoader output format. Skipping batch.")
                 continue
            except AttributeError as e:
                 write_log(f"ERROR ({dataset_name}) processing batch {i}: Attribute error (maybe '.to(device)' failed?). Error: {e}. Skipping batch.")
                 continue
            except Exception as e:
                write_log(f"ERROR ({dataset_name}) processing batch {i} (loading/moving data): {e}. Skipping batch.")
                continue

            # --- Forward Pass and Loss Calculation ---
            try:
                outputs = model(images)
                loss, indiv_losses = calculate_simplified_loss(outputs, labels, criterion_clf)

                # Check for NaN/Inf loss from calculation function
                if torch.isnan(loss) or torch.isinf(loss):
                    write_log(f"WARNING ({dataset_name}): NaN or Inf loss detected during evaluation batch {i}. Skipping loss accumulation for this batch.")
                    # Still process predictions if possible, but don't update loss metrics
                    loss = None # Mark loss as invalid for accumulation
                else:
                    # Accumulate valid losses and counts
                    total_loss += loss.item()
                    eval_indiv_losses['grade'] += indiv_losses.get('grade', 0.0) # Use .get for safety
                    eval_indiv_losses['total'] += indiv_losses.get('total', 0.0)

                # --- Get Predictions and Update Metrics ---
                # Ensure 'grade' output exists before trying to use it
                if 'grade' in outputs:
                    _, predicted_grades = torch.max(outputs['grade'], 1)
                    batch_size = labels['grade'].size(0)
                    total_samples += batch_size # Count samples processed for accuracy calculation
                    correct_grades += (predicted_grades == labels['grade']).sum().item()

                    # Store labels and predictions for detailed metrics later
                    all_labels_list.extend(labels['grade'].cpu().numpy())
                    all_preds_list.extend(predicted_grades.cpu().numpy())
                else:
                    write_log(f"WARNING ({dataset_name}): 'grade' key missing in model outputs for batch {i}. Cannot calculate accuracy for this batch.")
                    # Decide if you should skip the batch entirely or just the accuracy part

                # Increment processed batch count only if loss calculation was attempted (even if NaN)
                # If predictions were also processed successfully
                if 'grade' in outputs:
                    num_batches_processed += 1

            except Exception as e:
                write_log(f"ERROR ({dataset_name}) during model forward pass or metric calculation for batch {i}: {e}. Skipping batch.")
                continue # Skip to next batch

    # --- Calculate Average Metrics ---
    avg_loss = float('nan'); accuracy = 0.0
    avg_indiv_losses = {'grade': float('nan')}

    if num_batches_processed > 0 and total_samples > 0:
        avg_loss = total_loss / num_batches_processed # Average over batches where loss was calculated (and not NaN/Inf)
        accuracy = 100 * correct_grades / total_samples # Accuracy over samples where prediction was possible
        # Calculate average individual losses based on valid batches
        avg_indiv_losses = {k: v / num_batches_processed for k, v in eval_indiv_losses.items() if k == 'grade'}

        write_log(f'{dataset_name} Evaluation Summary:')
        write_log(f'  Processed Batches: {num_batches_processed}/{len(data_loader)}')
        write_log(f'  Successfully Processed Samples (for accuracy): {total_samples}')
        write_log(f'  Average Loss (over processed batches): {avg_loss:.4f}')
        write_log(f'  Grade Accuracy: {accuracy:.2f}% ({correct_grades}/{total_samples})')
        grade_loss_str = f'{avg_indiv_losses.get("grade", float("nan")):.4f}'
        write_log(f'  Avg Individual Losses: {{grade: {grade_loss_str}}}')

    elif num_samples_in_dataset > 0: # Dataset had samples, but none were processed successfully
         write_log(f"{dataset_name} Evaluation Summary: No samples successfully processed (0 batches completed).")
         # Ensure lists are empty if nothing was processed
         all_labels_list = []
         all_preds_list = []
    # else case (empty dataset) handled at the beginning

    # Return calculated metrics and the collected labels/predictions
    return avg_loss, accuracy, avg_indiv_losses, all_labels_list, all_preds_list

# --- Function to Plot Confusion Matrix ---
def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', filename=None):
    """Plots and saves a confusion matrix heatmap."""
    if not isinstance(cm, np.ndarray):
        write_log("ERROR: Confusion matrix is not a numpy array.")
        return
    if not class_names or not isinstance(class_names, list):
        write_log("ERROR: Invalid class names provided for confusion matrix.")
        # Use generic labels if class names are bad
        class_names = [str(i) for i in range(cm.shape[0])]

    try:
        plt.figure(figsize=(max(8, len(class_names) * 0.8), max(6, len(class_names) * 0.6))) # Adjust size based on num classes
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={"size": 8}) # Adjust annotation font size if needed
        plt.title(title, fontsize=12)
        plt.ylabel('True Label', fontsize=10)
        plt.xlabel('Predicted Label', fontsize=10)
        plt.xticks(rotation=45, ha='right', fontsize=8) # Rotate x-labels for readability
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout() # Adjust layout to prevent overlap

        if filename:
            try:
                ensure_dir_exists(filename) # Ensure directory exists before saving
                plt.savefig(filename, dpi=150) # Save with decent resolution
                write_log(f"Confusion matrix saved to: {filename}")
            except Exception as e:
                write_log(f"ERROR: Could not save confusion matrix plot to {filename}. Error: {e}")
        else:
            plt.show() # Show plot if no filename is given

        plt.close() # Close the plot figure to free memory
    except Exception as e:
        write_log(f"ERROR: Failed to generate confusion matrix plot. Error: {e}")


# --- Main Execution Block ---
if __name__ == '__main__':
    # Optional: freeze_support() for Windows multiprocessing when packaging
    # freeze_support() # Uncomment if needed when creating an executable

    # Set up device (GPU or CPU)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_log(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        write_log(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # --- 1. Load and Preprocess Data ---
    write_log("\n--- Loading Data ---")
    df_train = load_csv_safe(TRAIN_CSV_PATH, "train")
    df_val = load_csv_safe(VAL_CSV_PATH, "validation")
    df_test = load_csv_safe(TEST_CSV_PATH, "test")

    # --- Critical Data Checks ---
    if df_train is None:
        write_log("CRITICAL ERROR: Training data CSV failed to load or is invalid. Exiting.")
        exit(1)
    if df_train.empty:
        write_log("CRITICAL ERROR: Training dataframe is empty. Cannot train. Exiting.")
        exit(1)
    if df_val is None:
        write_log("WARNING: Validation data CSV failed to load. Proceeding without validation.")
        # Set df_val to empty DataFrame to simplify downstream checks
        df_val = pd.DataFrame()
    elif df_val.empty:
         write_log("WARNING: Validation dataframe is empty. Validation will be skipped.")
    if df_test is None:
         write_log("WARNING: Test data CSV failed to load. Proceeding without final testing.")
         df_test = pd.DataFrame() # Set df_test to empty DataFrame
    elif df_test.empty:
         write_log("WARNING: Test dataframe is empty. Final testing will be skipped.")

    # Optional: Duplicate training data
    if not df_train.empty:
        original_train_size = len(df_train)
        # Consider if duplication is truly necessary or if augmentations suffice
        df_train = pd.concat([df_train, df_train], ignore_index=True)
        # write_log(f"Duplicated training data. Original size: {original_train_size}, New size (doubled): {len(df_train)}")
        write_log(f"Using original training data size: {original_train_size}") # Modify log if not duplicating
    write_log("Data loading and initial checks successful.")

    # --- Create Label Mapping ---
    write_log("\n--- Creating Label Mappings ---")
    # Combine grades from all available, non-empty dataframes that have the 'Grade' column
    all_dfs_for_map = [df for df in [df_train, df_val, df_test]
                       if isinstance(df, pd.DataFrame) and not df.empty and 'Grade' in df.columns]

    if not all_dfs_for_map:
        write_log("CRITICAL ERROR: No valid data found in any CSV with 'Grade' column to create GRADE_MAP. Exiting.")
        exit(1)

    try:
        # Concatenate, drop NaNs from the 'Grade' column specifically, get unique values
        all_grades = pd.concat([df['Grade'] for df in all_dfs_for_map]).dropna().unique()
    except Exception as e:
        write_log(f"CRITICAL ERROR: Failed during grade aggregation for mapping. Error: {e}. Exiting.")
        exit(1)

    if len(all_grades) == 0:
         write_log("CRITICAL ERROR: No unique non-NaN grades found in the data. Check 'Grade' column content. Exiting.")
         exit(1)

    # Sort grades before mapping for consistent class indices
    GRADE_MAP = {grade: i for i, grade in enumerate(sorted(all_grades))}
    NUM_GRADES = len(GRADE_MAP)
    # Get class names in the order corresponding to the mapped indices (0, 1, 2, ...)
    CLASS_NAMES = [name for name, idx in sorted(GRADE_MAP.items(), key=lambda item: item[1])]

    if NUM_GRADES <= 1:
         write_log(f"WARNING: Only {NUM_GRADES} unique grade(s) found: {CLASS_NAMES}. Classification might not be meaningful.")
         # Decide whether to exit or proceed based on requirements
         # if NUM_GRADES <= 1: exit(1)


    write_log(f"GRADE_MAP: {GRADE_MAP}")
    write_log(f"Number of Grades (Classes): {NUM_GRADES}")
    write_log(f"Class Names (Order): {CLASS_NAMES}")

    loss_weights = {'grade': W_GRADE} # Simple weight for single task
    write_log(f"\nUsing Loss Weight: {loss_weights}")

    # --- Define Data Transforms ---
    write_log("\n--- Defining Data Transforms ---")
    # <<<< CORRECTED TRANSFORM USING DIRECT FUNCTION CALL >>>>
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        # Use the top-level function directly - Fixes pickling error on Windows
        random_discrete_rotation,
        transforms.RandomAffine(
            degrees=0,              # Rotation handled by random_discrete_rotation
            translate=(0.05, 0.05), # Small random shift
            scale=(0.9, 1.1),       # Random zoom in/out
            # shear=5               # Optional: small shear transformation
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05), # Moderate color augmentation
        transforms.ToTensor(),      # Convert PIL Image to PyTorch Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization
    ])

    # Transforms for validation and testing (no augmentation, just resize, tensor conversion, normalization)
    val_test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    write_log(f"Train Transforms: {train_transform}")
    write_log(f"Validation/Test Transforms: {val_test_transform}")

    # --- 2. Create Datasets & DataLoaders ---
    write_log("\n--- Creating Datasets and DataLoaders ---")
    train_dataset, val_dataset, test_dataset = None, None, None
    train_loader, val_loader, test_loader = None, None, None
    num_workers = min(os.cpu_count(), 4) # Use a reasonable number of workers
    write_log(f"Using num_workers = {num_workers} for DataLoaders.")

    try:
        # Create datasets only if the corresponding dataframe is valid and not empty
        if isinstance(df_train, pd.DataFrame) and not df_train.empty:
            train_dataset = TyphoonDataset(df_train, transform=train_transform, grade_map=GRADE_MAP)
        if isinstance(df_val, pd.DataFrame) and not df_val.empty:
            val_dataset = TyphoonDataset(df_val, transform=val_test_transform, grade_map=GRADE_MAP)
        if isinstance(df_test, pd.DataFrame) and not df_test.empty:
            test_dataset = TyphoonDataset(df_test, transform=val_test_transform, grade_map=GRADE_MAP)

        # Create DataLoaders only if the datasets were successfully created
        if train_dataset:
            # persistent_workers=True can speed up epoch start after the first one
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                      num_workers=num_workers, pin_memory=True,
                                      persistent_workers=(num_workers > 0)) # Enable if workers > 0
        if val_dataset:
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                    num_workers=num_workers, pin_memory=True,
                                    persistent_workers=(num_workers > 0))
        if test_dataset:
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                     num_workers=num_workers, pin_memory=True,
                                     persistent_workers=(num_workers > 0))

        write_log(f"Train dataset size: {len(train_dataset) if train_dataset else 'N/A'}")
        write_log(f"Validation dataset size: {len(val_dataset) if val_dataset else 'N/A'}")
        write_log(f"Test dataset size: {len(test_dataset) if test_dataset else 'N/A'}")

        # Check if train_loader is essential and usable
        if not train_loader:
             write_log("CRITICAL ERROR: Train DataLoader could not be created. Cannot proceed with training. Exiting.")
             exit(1)

    except ValueError as e: # Catch specific errors from Dataset creation
        write_log(f"CRITICAL ERROR during Dataset creation: {e}")
        exit(1)
    except Exception as e:
         write_log(f"CRITICAL ERROR during Dataset or DataLoader creation: {e}")
         exit(1)

    # --- Initialize Model, Loss, Optimizer, Scheduler ---
    write_log("\n--- Initializing Model and Training Components ---")
    if NUM_GRADES <= 0:
        write_log(f"CRITICAL ERROR: Number of grades is {NUM_GRADES}. Cannot initialize model. Check GRADE_MAP.")
        exit(1)

    model = ResNetGradeOnly(NUM_GRADES).to(DEVICE)

    # Define criterion (Loss function) with label smoothing
    criterion_clf = nn.CrossEntropyLoss(label_smoothing=0.1).to(DEVICE) # Move loss to device if it has params

    # Optimizer (AdamW is often a good default)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Learning Rate Scheduler (ReduceLROnPlateau monitors validation accuracy)
    # Use 'max' mode because we want to maximize accuracy
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=SCHEDULER_FACTOR,
                                  patience=SCHEDULER_PATIENCE, verbose=False) # Manual logging of LR changes

    best_val_acc = 0.0          # Track the best validation accuracy achieved
    epochs_no_improve = 0       # Counter for early stopping
    training_history = []       # List to store metrics for each epoch

    # Log training setup details
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    write_log(f"\nModel: ResNet18 (Grade Prediction Only)")
    write_log(f"Total Parameters: {total_params:,}")
    write_log(f"Trainable Parameters: {trainable_params:,}")
    try:
         # Access the Linear layer within the Sequential head
         grade_head_linear = model.grade_head[-1] # Assumes Linear is the last layer
         if isinstance(grade_head_linear, nn.Linear):
             write_log(f"Grade head input dimension: {grade_head_linear.in_features}")
             write_log(f"Grade head output dimension: {grade_head_linear.out_features}")
         else:
             write_log("Could not confirm grade head structure.")
    except (AttributeError, IndexError, TypeError):
         write_log("Could not determine grade head dimensions.")

    write_log(f"\n--- Starting Training Configuration ---")
    write_log(f"Device: {DEVICE}")
    write_log(f"Epochs: {EPOCHS}")
    write_log(f"Batch Size: {BATCH_SIZE}")
    write_log(f"Initial Learning Rate: {LEARNING_RATE}")
    write_log(f"Weight Decay: {WEIGHT_DECAY}")
    write_log(f"Loss Function: CrossEntropyLoss (Label Smoothing=0.1)")
    write_log(f"Optimizer: AdamW")
    write_log(f"Scheduler: ReduceLROnPlateau (Mode=max, Patience={SCHEDULER_PATIENCE}, Factor={SCHEDULER_FACTOR})")
    write_log(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
    write_log(f"Model Save Path: {MODEL_SAVE_PATH}")
    write_log(f"Log File Path: {LOG_FILE_PATH}")
    write_log(f"Metrics File Path: {METRICS_SAVE_PATH}")
    write_log("-" * 60)

    # --- Training & Validation Loop ---
    start_time_training = time.time()
    write_log(f"\n--- Starting Training Loop ---")
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        write_log(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")

        # --- Training Phase ---
        model.train() # Set model to training mode
        running_loss = 0.0
        train_correct_grades = 0
        train_total_samples = 0
        train_batches_processed = 0
        epoch_train_indiv_losses = {'total': 0.0, 'grade': 0.0} # Reset per epoch

        for i, batch in enumerate(train_loader):
            batch_start_time = time.time()
            try:
                # --- Data Loading and Moving ---
                if not isinstance(batch, dict) or 'image' not in batch or 'labels' not in batch or 'grade' not in batch['labels']:
                     write_log(f"WARNING (Train): Invalid batch structure at index {i}. Skipping.")
                     continue
                images = batch['image'].to(DEVICE, non_blocking=True)
                labels_grade = batch['labels']['grade'].to(DEVICE, non_blocking=True, dtype=torch.long)
                labels = {'grade': labels_grade}
                if images.nelement() == 0 or labels['grade'].nelement() == 0:
                    write_log(f"WARNING (Train): Empty tensor in batch {i}. Skipping.")
                    continue

                # --- Forward Pass ---
                optimizer.zero_grad(set_to_none=True) # More efficient zeroing
                outputs = model(images)

                # --- Loss Calculation ---
                loss, indiv_losses = calculate_simplified_loss(outputs, labels, criterion_clf)

                # --- Backward Pass and Optimization ---
                # Handle potential NaN/Inf loss before backward pass
                if torch.isnan(loss) or torch.isinf(loss):
                    write_log(f"WARNING (Train): NaN or Inf loss detected epoch {epoch+1}, batch {i}. Skipping backward/step.")
                    # Skip gradient calculation and optimizer step for this batch
                    continue # Go to the next batch

                loss.backward() # Calculate gradients
                # Optional: Gradient Clipping (uncomment if needed)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step() # Update model weights

                # --- Accumulate Metrics for Valid Steps ---
                running_loss += loss.item()
                epoch_train_indiv_losses['grade'] += indiv_losses.get('grade', 0.0)
                epoch_train_indiv_losses['total'] += indiv_losses.get('total', 0.0)

                # Calculate accuracy for the batch
                if 'grade' in outputs:
                    _, predicted_grades = torch.max(outputs['grade'], 1)
                    current_batch_size = labels['grade'].size(0)
                    train_total_samples += current_batch_size
                    train_correct_grades += (predicted_grades == labels['grade']).sum().item()
                else:
                     write_log(f"WARNING (Train): 'grade' output missing batch {i}, cannot update accuracy.")

                train_batches_processed += 1 # Increment count of successfully processed batches

                # Optional: Log progress every N batches
                # if (i + 1) % 50 == 0:
                #    batch_time = time.time() - batch_start_time
                #    write_log(f"  Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}, Time: {batch_time:.2f}s")

            except Exception as e:
                 # Catch unexpected errors during the train batch loop
                 write_log(f"ERROR (Train) during processing batch {i}: {e}")
                 # Optionally add more details like traceback:
                 # import traceback
                 # write_log(traceback.format_exc())
                 # Decide whether to continue or stop training on such errors
                 continue # Attempt to continue with the next batch


        # --- End of Training Epoch ---
        epoch_train_loss = running_loss / train_batches_processed if train_batches_processed > 0 else float('nan')
        epoch_train_acc = 100 * train_correct_grades / train_total_samples if train_total_samples > 0 else 0.0
        avg_train_indiv_losses = {k: v / train_batches_processed for k, v in epoch_train_indiv_losses.items()} if train_batches_processed > 0 else {'grade': float('nan')}

        write_log(f'--- Training Summary (Epoch {epoch+1}) ---')
        if train_batches_processed > 0:
            write_log(f'  Batches Processed: {train_batches_processed}/{len(train_loader)}')
            write_log(f'  Avg Train Loss: {epoch_train_loss:.4f}')
            write_log(f'  Train Grade Accuracy: {epoch_train_acc:.2f}% ({train_correct_grades}/{train_total_samples})')
            train_grade_loss_str = f'{avg_train_indiv_losses.get("grade", float("nan")):.4f}'
            write_log(f'  Avg Train Individual Losses: {{grade: {train_grade_loss_str}}}')
        else:
            write_log("  No training batches were successfully processed in this epoch.")
            # If no training batches worked, maybe stop?
            # write_log("CRITICAL: Training failed for an entire epoch. Aborting.")
            # break

        # --- Validation Phase ---
        epoch_val_loss = float('nan') # Default values if validation skipped/failed
        epoch_val_acc = 0.0
        avg_val_indiv_losses = {'grade': float('nan')}
        validation_performed = False # Flag to track if validation ran

        if val_loader:
            validation_performed = True
            # Pass criterion_clf to the evaluation function
            val_loss, val_acc, val_indiv_loss_dict, _, _ = evaluate_grade_only(
                model, val_loader, DEVICE, criterion_clf, dataset_name=f"Validation (Epoch {epoch+1})"
            ) # Ignore labels/preds here, only need metrics

            # Check if validation produced valid results
            if not np.isnan(val_loss) and not np.isnan(val_acc):
                 epoch_val_loss = val_loss
                 epoch_val_acc = val_acc
                 avg_val_indiv_losses = val_indiv_loss_dict # Already averaged in evaluate func

                 # --- Scheduler Step & Early Stopping Logic ---
                 current_lr = optimizer.param_groups[0]['lr']
                 scheduler.step(epoch_val_acc) # Step scheduler based on validation accuracy
                 new_lr = optimizer.param_groups[0]['lr']
                 if new_lr < current_lr:
                     write_log(f"  Learning rate reduced by scheduler to {new_lr:.2e}")

                 if epoch_val_acc > best_val_acc:
                     best_val_acc = epoch_val_acc
                     write_log(f'  ** New Best Validation Accuracy: {best_val_acc:.2f}% **')
                     # Save the model state dictionary
                     try:
                         ensure_dir_exists(MODEL_SAVE_PATH) # Ensure directory exists again (safe)
                         torch.save(model.state_dict(), MODEL_SAVE_PATH)
                         write_log(f'  ** Best model saved to {MODEL_SAVE_PATH} **')
                         epochs_no_improve = 0 # Reset counter
                     except Exception as e:
                          write_log(f"ERROR: Failed to save best model to {MODEL_SAVE_PATH}. Error: {e}")
                 else:
                     epochs_no_improve += 1
                     write_log(f'  Validation accuracy ({epoch_val_acc:.2f}%) did not improve. Best: {best_val_acc:.2f}%. ({epochs_no_improve}/{EARLY_STOPPING_PATIENCE})')
                     if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                         write_log(f"\n--- Early stopping triggered after {epoch + 1} epochs. ---")
                         break # Exit the training loop

            else:
                 # Validation evaluation didn't return valid numeric results
                 write_log(f"WARNING: Validation evaluation results invalid for epoch {epoch+1} (Loss: {val_loss}, Acc: {val_acc}).")
                 # Treat as no improvement for early stopping purposes
                 epochs_no_improve += 1
                 write_log(f'  Epochs without valid validation improvement: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}')
                 if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                      write_log(f"\n--- Early stopping triggered after {epoch + 1} epochs (due to consecutive validation issues/no improvement). ---")
                      break # Exit the training loop
        else:
             # val_loader was None or empty
             write_log(f"--- Validation Summary (Epoch {epoch+1}) ---")
             write_log("  Skipping validation phase: No validation data/loader available.")
             # If no validation, early stopping based on val_acc cannot occur.
             # Training will continue until EPOCHS limit unless manually stopped.

        # --- Store Epoch History ---
        history_entry = {
            'epoch': epoch + 1,
            'train_loss': epoch_train_loss,
            'train_acc': epoch_train_acc,
            'val_loss': epoch_val_loss if validation_performed else 'N/A', # Store NaN or N/A if skipped
            'val_acc': epoch_val_acc if validation_performed else 'N/A',
            'lr': optimizer.param_groups[0]['lr'] # Log learning rate for the epoch
        }
        training_history.append(history_entry)

        # --- Log Epoch Time ---
        epoch_duration = time.time() - epoch_start_time
        write_log(f"--- Epoch {epoch+1} Time: {epoch_duration:.2f}s ---")
        write_log("-" * 60) # Separator for readability

    # --- End of Training Loop ---
    total_training_time = time.time() - start_time_training
    write_log("\n--- Training Finished ---")
    write_log(f"Total Training Time: {total_training_time:.2f}s ({total_training_time/60:.2f} minutes)")
    if best_val_acc > 0:
        write_log(f"Best Validation Grade Accuracy achieved: {best_val_acc:.2f}%")
    else:
        write_log("No best validation accuracy recorded (validation might have been skipped or failed).")

    # --- Print Training History Table ---
    write_log("\n--- Training History Summary ---")
    if training_history: # Check if history is not empty
        header = "| Epoch | Train Loss | Train Acc (%) | Val Loss   | Val Acc (%) | Learning Rate |"
        write_log(header)
        write_log("-" * len(header))
        for entry in training_history:
            # Format N/A or float values appropriately
            val_loss_str = f"{entry['val_loss']:.4f}" if isinstance(entry['val_loss'], (int, float)) else str(entry['val_loss'])
            val_acc_str = f"{entry['val_acc']:.2f}" if isinstance(entry['val_acc'], (int, float)) else str(entry['val_acc'])

            log_line = (f"| {entry['epoch']:<5} | "
                        f"{entry['train_loss']:<10.4f} | "
                        f"{entry['train_acc']:<13.2f} | "
                        f"{val_loss_str:<10} | "
                        f"{val_acc_str:<11} | "
                        f"{entry['lr']:.2e}       |") # Format LR in scientific notation
            write_log(log_line)
        write_log("-" * len(header))
    else:
        write_log("No training history was recorded (e.g., training aborted before first epoch finished).")


    # --- Final Evaluation Phase using the Best Saved Model ---
    write_log(f"\n--- Final Evaluation using Best Model ---")
    write_log(f"Attempting to load best model state_dict from: {MODEL_SAVE_PATH}")

    final_evaluation_summary = [] # Store lines for metrics file

    if os.path.exists(MODEL_SAVE_PATH):
        # Re-initialize model architecture
        model_to_test = ResNetGradeOnly(NUM_GRADES).to(DEVICE)
        try:
            # Load the saved state dictionary
            model_to_test.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
            write_log("Best model state_dict loaded successfully.")
            model_to_test.eval() # Set model to evaluation mode

            # --- Evaluate on Validation Set (using best model) ---
            if val_loader:
                write_log("\n--- Evaluating Best Model on Validation Set ---")
                # Pass criterion for loss calculation if needed, otherwise can be None if only metrics needed
                val_loss_final, val_acc_final, _, val_labels_final, val_preds_final = evaluate_grade_only(
                    model_to_test, val_loader, DEVICE, criterion_clf, dataset_name="Validation (Best Model)"
                )

                # Calculate and Log/Save Detailed Validation Metrics
                if val_labels_final is not None and val_preds_final is not None and len(val_labels_final) > 0:
                     try:
                        report_val = classification_report(
                            val_labels_final, val_preds_final,
                            target_names=CLASS_NAMES, digits=4, zero_division=0
                        )
                        cm_val = confusion_matrix(val_labels_final, val_preds_final, labels=range(NUM_GRADES)) # Ensure CM has all classes

                        write_log("\n--- Validation Set Performance Metrics (Best Model) ---")
                        write_log(f"Overall Accuracy: {val_acc_final:.2f}%")
                        write_log("Classification Report:")
                        write_log(report_val)
                        write_log("Confusion Matrix:")
                        write_log(str(cm_val)) # Log CM as string

                        # Prepare lines for metrics file
                        final_evaluation_summary.append(f"\n\n--- Validation Set Performance Metrics (Best Model) ---")
                        final_evaluation_summary.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                        final_evaluation_summary.append(f"Model Path: {MODEL_SAVE_PATH}")
                        final_evaluation_summary.append(f"Overall Accuracy: {val_acc_final:.2f}%")
                        final_evaluation_summary.append("\nClassification Report:\n" + report_val)
                        final_evaluation_summary.append("\nConfusion Matrix:\n" + np.array2string(cm_val, separator=', '))
                        final_evaluation_summary.append("-" * 60)

                        # Plot and save confusion matrix
                        plot_confusion_matrix(cm_val, CLASS_NAMES,
                                            title=f'Validation CM (Best Model - Acc: {val_acc_final:.2f}%)',
                                            filename=CONFUSION_MATRIX_VAL_PATH)

                     except Exception as e:
                          write_log(f"ERROR: Could not calculate or save detailed validation metrics. Error: {e}")
                          final_evaluation_summary.append(f"\nERROR calculating/saving validation metrics: {e}")
                else:
                     write_log("Skipping detailed validation metrics calculation (no labels/predictions returned from evaluation).")
                     final_evaluation_summary.append("\n--- Validation Set Metrics Skipped (No Results) ---")
            else:
                write_log("Skipping final validation set evaluation: Validation loader not available.")
                final_evaluation_summary.append("\n--- Validation Set Evaluation Skipped (No Loader) ---")


            # --- Evaluate on Test Set (using best model) ---
            if test_loader:
                write_log("\n--- Evaluating Best Model on Test Set ---")
                test_loss, test_acc, _, test_labels, test_preds = evaluate_grade_only(
                    model_to_test, test_loader, DEVICE, criterion_clf, dataset_name="Test (Best Model)"
                )

                # Calculate and Log/Save Detailed Test Metrics
                if test_labels is not None and test_preds is not None and len(test_labels) > 0:
                     try:
                        report_test = classification_report(
                            test_labels, test_preds,
                            target_names=CLASS_NAMES, digits=4, zero_division=0
                        )
                        cm_test = confusion_matrix(test_labels, test_preds, labels=range(NUM_GRADES)) # Ensure CM has all classes

                        write_log("\n--- Test Set Performance Metrics (Best Model) ---")
                        write_log(f"Overall Accuracy: {test_acc:.2f}%")
                        write_log("Classification Report:")
                        write_log(report_test)
                        write_log("Confusion Matrix:")
                        write_log(str(cm_test)) # Log CM as string

                        # Prepare lines for metrics file
                        final_evaluation_summary.append(f"\n\n--- Test Set Performance Metrics (Best Model) ---")
                        final_evaluation_summary.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                        final_evaluation_summary.append(f"Model Path: {MODEL_SAVE_PATH}")
                        final_evaluation_summary.append(f"Overall Accuracy: {test_acc:.2f}%")
                        final_evaluation_summary.append("\nClassification Report:\n" + report_test)
                        final_evaluation_summary.append("\nConfusion Matrix:\n" + np.array2string(cm_test, separator=', '))
                        final_evaluation_summary.append("-" * 60)


                        # Plot and save confusion matrix
                        plot_confusion_matrix(cm_test, CLASS_NAMES,
                                            title=f'Test CM (Best Model - Acc: {test_acc:.2f}%)',
                                            filename=CONFUSION_MATRIX_TEST_PATH)

                     except Exception as e:
                          write_log(f"ERROR: Could not calculate or save detailed test metrics. Error: {e}")
                          final_evaluation_summary.append(f"\nERROR calculating/saving test metrics: {e}")
                else:
                     write_log("Skipping detailed test metrics calculation (no labels/predictions returned from evaluation).")
                     final_evaluation_summary.append("\n--- Test Set Metrics Skipped (No Results) ---")

            else:
                write_log("Skipping final test set evaluation: Test loader not available.")
                final_evaluation_summary.append("\n--- Test Set Evaluation Skipped (No Loader) ---")

        except FileNotFoundError:
             write_log(f"ERROR: Model state_dict file not found at {MODEL_SAVE_PATH} during final evaluation load attempt.")
             final_evaluation_summary.append(f"\n\n--- Final Evaluation Skipped ---")
             final_evaluation_summary.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
             final_evaluation_summary.append(f"Reason: Model file not found at {MODEL_SAVE_PATH}")
        except Exception as e:
            write_log(f"ERROR: Failed to load model state_dict or run final evaluation. Error: {e}")
            final_evaluation_summary.append(f"\n\n--- Final Evaluation Failed ---")
            final_evaluation_summary.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            final_evaluation_summary.append(f"Reason: Error during load/eval - {e}")


    else: # Best model path doesn't exist
        write_log(f"ERROR: Best model file not found at {MODEL_SAVE_PATH}. Cannot perform final evaluation.")
        final_evaluation_summary.append(f"\n\n--- Final Evaluation Skipped ---")
        final_evaluation_summary.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        final_evaluation_summary.append(f"Reason: Best model file not found at {MODEL_SAVE_PATH}")

    # --- Append Final Evaluation Summary to Metrics File ---
    if final_evaluation_summary:
        try:
            ensure_dir_exists(METRICS_SAVE_PATH)
            with open(METRICS_SAVE_PATH, "a", encoding='utf-8') as f:
                for line in final_evaluation_summary:
                    f.write(line + "\n")
            write_log(f"\nFinal evaluation summary appended to: {METRICS_SAVE_PATH}")
        except Exception as e:
            write_log(f"ERROR: Could not append final evaluation summary to {METRICS_SAVE_PATH}. Error: {e}")


    write_log("\n--- Script Finished ---")

# --- END OF FILE ResNet18_DvorakNoPattern.py ---