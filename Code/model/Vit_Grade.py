# --- START OF FILE Vit_Grade.py ---

import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, default_collate # Import default_collate
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF # Import functional API for transforms
import pandas as pd
from PIL import Image
import os
import time
import random
from multiprocessing import freeze_support # Required for Windows multiprocessing
import numpy as np # Added for metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score # Added for metrics
import matplotlib.pyplot as plt # Added for plotting
import seaborn as sns # Added for confusion matrix visualization
import traceback # For detailed error logging

# --- Logging Setup ---
# <<< Ensure this path is correct for your system >>>
log_dir = "LLog" # Directory for logs and reports
log_file = os.path.join(log_dir, "training_log_ViT_S_Dvorak_GradeOnly.txt")
# CSV Paths (kept for data record)
conf_matrix_val_csv_file = os.path.join(log_dir, "confusion_matrix_val.csv")
conf_matrix_test_csv_file = os.path.join(log_dir, "confusion_matrix_test.csv")
# PNG Paths for Plots
loss_plot_file = os.path.join(log_dir, "loss_plot.png")
accuracy_plot_file = os.path.join(log_dir, "accuracy_plot.png")
conf_matrix_val_png_file = os.path.join(log_dir, "confusion_matrix_val.png")
conf_matrix_test_png_file = os.path.join(log_dir, "confusion_matrix_test.png")

# Create directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

def write_log(message):
    """Logs a message to both console and the log file."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    full_message = f"{timestamp} - {message}"
    try:
        with open(log_file, "a", encoding='utf-8') as f: # Specify encoding
            f.write(full_message + "\n")
    except Exception as log_err:
        print(f"{timestamp} - ERROR: Could not write to log file '{log_file}'. Reason: {log_err}")
        # Optionally print traceback for logging errors
        # traceback.print_exc()
    print(full_message) # Print to console regardless of file writing success


# --- Configuration & Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# <<< Ensure these paths are correct for your system >>>
MODEL_SAVE_PATH = 'model/best_typhoon_model_vit_s_grade_only.pth'
TRAIN_CSV_PATH = 'dataset/train_csv_4.csv'
VAL_CSV_PATH = 'dataset/val_csv_4.csv'
TEST_CSV_PATH = 'dataset/test_csv_4.csv'


# --- Utility Functions ---
def load_csv_safe(path, name):
    """Safely loads a CSV file into a Pandas DataFrame."""
    try:
        df = pd.read_csv(path)
        write_log(f"Successfully loaded {name} CSV from {path} ({len(df)} rows)")
        return df
    except FileNotFoundError:
        write_log(f"ERROR: {name} CSV file not found at {path}")
        return None
    except pd.errors.EmptyDataError:
        write_log(f"ERROR: {name} CSV file at {path} is empty.")
        return None
    except Exception as e:
        write_log(f"ERROR: Could not load {name} CSV from {path}. Reason: {e}")
        write_log(traceback.format_exc())
        return None

# --- Plotting Function ---
def plot_metrics(epochs_ran, train_losses, val_losses, train_accs, val_accs, loss_path, acc_path):
    """Plots training & validation loss and accuracy and saves them."""
    if not epochs_ran:
        write_log("Skipping plot generation: No epochs were completed.")
        return

    epochs = range(1, epochs_ran + 1)

    plt.style.use('seaborn-v0_8-darkgrid') # Use a visually appealing style

    # --- Plot Loss ---
    try:
        plt.figure(figsize=(10, 5))
        if train_losses:
            plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
        if val_losses:
            plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        # Only show legend if there's something to label
        if train_losses or val_losses:
            plt.legend()
        plt.xticks(epochs) # Ensure integer ticks for epochs if feasible
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(loss_path)
        write_log(f"Loss plot saved to {loss_path}")
    except Exception as e:
        write_log(f"ERROR: Could not generate or save loss plot to {loss_path}. Reason: {e}")
        write_log(traceback.format_exc())
    finally:
        plt.close() # Close the figure to free memory

    # --- Plot Accuracy ---
    try:
        plt.figure(figsize=(10, 5))
        if train_accs:
            plt.plot(epochs, train_accs, 'bo-', label='Training Accuracy')
        if val_accs:
            plt.plot(epochs, val_accs, 'ro-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        # Only show legend if there's something to label
        if train_accs or val_accs:
            plt.legend()
        plt.xticks(epochs) # Ensure integer ticks for epochs if feasible
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(acc_path)
        write_log(f"Accuracy plot saved to {acc_path}")
    except Exception as e:
        write_log(f"ERROR: Could not generate or save accuracy plot to {acc_path}. Reason: {e}")
        write_log(traceback.format_exc())
    finally:
        plt.close() # Close the figure


# --- Collate Function (Moved to Top Level) ---
def collate_fn_skip_none(batch):
   """
   Filters out None items from a batch and then uses the default collate function.
   Returns None if the entire batch is filtered out.
   """
   # Filter out None items first
   filtered_batch = [item for item in batch if item is not None]
   # If the filtered batch is empty, return None (dataloader loop should handle this)
   if not filtered_batch:
       return None
   # Otherwise, use the default collate function on the filtered batch
   try:
       return default_collate(filtered_batch)
   except RuntimeError as e:
       # Catch specific errors, e.g., stack error if tensors have different sizes unexpectedly
       write_log(f"ERROR during default_collate after filtering Nones: {e}")
       # Log details about the batch that failed collation if possible (be careful with large data)
       # for i, item in enumerate(filtered_batch):
       #     if isinstance(item, dict) and 'image' in item:
       #         write_log(f" Item {i} image shape: {item['image'].shape}")
       # Returning None might be safer to avoid crashing the training loop entirely
       return None
   except Exception as e:
        write_log(f"UNEXPECTED ERROR during default_collate: {e}")
        write_log(traceback.format_exc())
        return None


# --- Model Hyperparameters ---
IMG_SIZE = 224
# NUM_GRADES defined inside `if __name__ == '__main__':`

# --- Training Hyperparameters ---
LEARNING_RATE = 8e-5
BATCH_SIZE = 32 # Adjust based on your GPU memory
EPOCHS = 50
WEIGHT_DECAY = 0.05
SCHEDULER_PATIENCE = 20 # Reduced patience for potentially faster LR adjustment
SCHEDULER_FACTOR = 0.2
EARLY_STOPPING_PATIENCE = 30 # Reduced patience for faster stopping if no improvement
NUM_WORKERS = 4 # Set to 0 if encountering multiprocessing issues, especially on Windows first


# --- Data Augmentation and Transforms ---
# Define the custom transform class HERE (outside the main block)
class RandomRotationTransform:
    """Rotates the image by a randomly chosen angle from a given list."""
    def __init__(self, angles=[0, 90, 180, 270]): # Include 0 degrees explicitly
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        # Use torchvision.transforms.functional.rotate
        return TF.rotate(img, angle)

# Define the transforms using the custom class
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    RandomRotationTransform(angles=[0, 90, 180, 270]), # Use instance of the custom class
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), # Added slight color jitter
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Dataset Definition ---
class TyphoonGradeDataset(Dataset):
    def __init__(self, dataframe, grade_map, transform):
        if dataframe is None:
            raise ValueError("Input DataFrame cannot be None for TyphoonGradeDataset.")
        self.dataframe = dataframe.copy() # Work on a copy
        self.transform = transform
        self.grade_map = grade_map
        # Pre-filter dataframe for valid grades to avoid errors in __getitem__
        self.dataframe = self.dataframe[self.dataframe['Grade'].notna()]
        self.dataframe = self.dataframe[self.dataframe['Grade'].isin(self.grade_map.keys())]
        if len(self.dataframe) == 0:
             write_log("WARNING: DataFrame is empty after filtering for valid and mappable grades.")
        # self._check_mappings() # Check is implicitly done by filtering above

    def _check_mappings(self):
        """DEPRECATED (filtering done in __init__): Check necessary mappings."""
        try:
            # Check if all 'Grade' values exist in the map AFTER filtering NaNs
            valid_grades = self.dataframe['Grade'].dropna()
            if not valid_grades.isin(self.grade_map.keys()).all():
                 problem_vals = valid_grades[~valid_grades.isin(self.grade_map.keys())].unique()
                 raise KeyError(f"Value(s) {list(problem_vals)} not found in GRADE_MAP.")
        except KeyError as e:
            write_log(f"\n\nMAPPING ERROR during Dataset Initialization!")
            problem_col = 'Grade'
            all_df_grades = self.dataframe[problem_col].unique()
            missing_in_map = [g for g in all_df_grades if g not in self.grade_map and pd.notna(g)]
            write_log(f"Column '{problem_col}' contains value(s) not found in GRADE_MAP: {missing_in_map}")
            write_log(f"Specific error message: {e}")
            write_log(f"Provided GRADE_MAP: {self.grade_map}")
            raise ValueError(f"Values {missing_in_map} not found in provided GRADE_MAP.") from e
        except Exception as general_e:
             write_log(f"\n\nUNEXPECTED ERROR during mapping check: {general_e}")
             write_log(traceback.format_exc())
             raise general_e

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
             idx = idx.tolist()

        # Use try-except for the whole process for a single item
        try:
            # Use .iloc for integer-based indexing, reset_index might be needed if df index isn't sequential
            row = self.dataframe.iloc[idx]
            img_path = row['file_name']
            grade_val = row['Grade'] # Already checked for NaN and map validity in __init__

            try:
                # Check if path exists before trying to open
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"Image file not found at {img_path}")
                # Use context manager for file opening
                with Image.open(img_path) as img:
                    image = img.convert('RGB')
            except FileNotFoundError as fnf_err:
                 write_log(f"ERROR [Dataset Index {idx}]: {fnf_err}. Returning None.")
                 return None # Signal to collate_fn_skip_none to skip this item
            except Exception as img_err:
                 write_log(f"ERROR [Dataset Index {idx}]: Could not open/process image {img_path}. Error: {img_err}. Returning None.")
                 # Consider logging traceback for unexpected image errors
                 # write_log(traceback.format_exc())
                 return None # Skip this item

            # Apply transformations
            if self.transform:
                 image = self.transform(image)

            # Get and encode the 'Grade' label (should be safe due to pre-filtering)
            grade = torch.tensor(self.grade_map[grade_val], dtype=torch.long)

            return {
                'image': image,
                'labels': {
                    'grade': grade,
                }
            }
        # Catch errors specific to this item fetching process
        except IndexError:
            write_log(f"ERROR: Index {idx} out of bounds for DataFrame length {len(self.dataframe)}.")
            return None # Skip item if index is bad
        except KeyError as ke:
            # This might happen if the DataFrame structure is unexpected
            write_log(f"ERROR [Dataset Index {idx}]: KeyError accessing row data. Error: {ke}. Row Data (may be partial): {row if 'row' in locals() else 'N/A'}. Returning None.")
            return None
        except Exception as getitem_err:
            # Log unexpected errors during item fetching
            write_log(f"ERROR in __getitem__ for index {idx}: {getitem_err}")
            write_log(traceback.format_exc())
            return None # Skip item on unexpected errors


# --- Model Definition ---
class ViTGradeClassifier(nn.Module):
    def __init__(self, num_grades):
        super().__init__()
        # Load pre-trained model, handle potential download issues
        try:
            self.vit = timm.create_model('vit_small_patch16_224', pretrained=True)
        except Exception as e:
            write_log(f"FATAL ERROR: Could not create timm model 'vit_small_patch16_224'. Check internet connection and timm installation. Error: {e}")
            raise e # Stop execution if model can't be loaded

        vit_feature_dim = self.vit.num_features
        # Replace the classifier head
        self.vit.head = nn.Identity() # Remove original head
        # Define new head for grade classification
        self.grade_head = nn.Sequential(
            nn.LayerNorm(vit_feature_dim), # Added LayerNorm before Linear
            nn.Dropout(0.5),
            nn.Linear(vit_feature_dim, num_grades)
        )

    def forward(self, x):
        # Pass input through ViT base
        features = self.vit(x)
        # Pass features through the grade classification head
        out_grade = self.grade_head(features)
        # Return output in a dictionary
        outputs = {'grade': out_grade}
        return outputs

# --- Loss Function ---
criterion_clf = nn.CrossEntropyLoss(label_smoothing=0.1) # Added label smoothing

# --- Loss Calculation ---
def calculate_loss(outputs, labels, batch_idx, epoch_num):
    """Calculates the loss, adding context for logging."""
    try:
        # Ensure labels are on the correct device and have the right shape/type
        target_labels = labels['grade'].to(outputs['grade'].device, dtype=torch.long)
        # Calculate loss
        loss_grade = criterion_clf(outputs['grade'], target_labels)

        # Check for NaN/Inf loss immediately
        if torch.isnan(loss_grade) or torch.isinf(loss_grade):
            write_log(f"WARNING: NaN/Inf loss detected at Epoch {epoch_num}, Batch {batch_idx}. Loss: {loss_grade.item()}")
            # Log details for debugging
            write_log(f"  Output Grade (sample): {outputs['grade'][0]}")
            write_log(f"  Target Labels (sample): {target_labels[0]}")
            # Return a tensor representation of NaN to be handled in the training loop
            return torch.tensor(float('nan'), device=outputs['grade'].device)

    except KeyError as e:
         write_log(f"ERROR [Loss Calc Epoch {epoch_num}, Batch {batch_idx}]: KeyError accessing key: {e}. Check model output/label keys.")
         raise e # Re-raise to stop if keys are wrong
    except TypeError as e:
        write_log(f"ERROR [Loss Calc Epoch {epoch_num}, Batch {batch_idx}]: TypeError during loss calculation: {e}")
        write_log(f"  Output Grade type: {outputs.get('grade').dtype if outputs.get('grade') is not None else 'N/A'}")
        write_log(f"  Target Labels type: {labels.get('grade').dtype if labels.get('grade') is not None else 'N/A'}")
        raise e
    except Exception as e:
        write_log(f"ERROR [Loss Calc Epoch {epoch_num}, Batch {batch_idx}]: Error calculating loss: {e}")
        write_log("Shapes:")
        write_log(f"  outputs['grade']: {outputs.get('grade').shape if outputs.get('grade') is not None else 'N/A'}")
        write_log(f"  labels['grade']: {labels.get('grade').shape if labels.get('grade') is not None else 'N/A'}")
        write_log(traceback.format_exc())
        raise e # Re-raise unexpected errors
    return loss_grade

# --- Evaluation Function (Modified) ---
def evaluate(model, data_loader, device, criterion_eval, dataset_name="Validation", epoch_num=0):
    """Evaluates the model and returns loss, accuracy, and lists of labels/predictions."""
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    all_true_labels = []
    all_predicted_labels = []
    batches_processed = 0

    num_samples_dataset = len(data_loader.dataset) if data_loader.dataset else 0
    num_batches = len(data_loader)
    # Log start of evaluation for clarity
    # write_log(f"--- Starting Evaluation on {dataset_name} set (Epoch {epoch_num}, {num_samples_dataset} samples, {num_batches} batches) ---")

    if num_samples_dataset == 0 or num_batches == 0:
        write_log(f"WARNING: {dataset_name} evaluation skipped (Epoch {epoch_num}): data loader is empty or dataset has zero length.")
        return float('nan'), 0.0, [], []

    with torch.no_grad(): # Disable gradient calculations
        for i, batch in enumerate(data_loader):
            # Handle None batches potentially returned by collate_fn
            if batch is None:
                write_log(f"Warning [Eval Epoch {epoch_num}, {dataset_name}]: Skipping None batch at index {i}.")
                continue
            try:
                images = batch['image'].to(device, non_blocking=True)
                labels_grade = batch['labels']['grade'].to(device, non_blocking=True)

                # Forward pass
                outputs = model(images)

                # Calculate loss (using a separate criterion if needed, e.g., without label smoothing)
                loss = criterion_eval(outputs['grade'], labels_grade)

                # Check for NaN/Inf loss
                if torch.isnan(loss) or torch.isinf(loss):
                    write_log(f"WARNING [Eval Epoch {epoch_num}, Batch {i+1}, {dataset_name}]: NaN or Inf loss detected ({loss.item()}). Skipping batch in loss calculation.")
                    continue # Skip this batch for average loss calculation

                total_loss += loss.item()

                # Get predictions
                _, predicted_grades = torch.max(outputs['grade'], 1)

                # Collect labels and predictions (move to CPU before converting to numpy)
                all_true_labels.extend(labels_grade.cpu().numpy())
                all_predicted_labels.extend(predicted_grades.cpu().numpy())
                batches_processed += 1

            # Handle potential errors during a batch evaluation
            except Exception as e:
                write_log(f"\nERROR during {dataset_name} evaluation batch {i+1}/{num_batches} (Epoch {epoch_num}): {e}")
                write_log(traceback.format_exc())
                # Decide if you want to continue or raise the error
                continue # Continue evaluation if one batch fails

    # Calculate average loss and accuracy
    avg_loss = total_loss / batches_processed if batches_processed > 0 else float('nan')
    # Calculate accuracy using collected labels/predictions only if lists are not empty
    accuracy = accuracy_score(all_true_labels, all_predicted_labels) * 100 if all_true_labels and all_predicted_labels else 0.0

    # Moved logging summary to where evaluate is called
    return avg_loss, accuracy, all_true_labels, all_predicted_labels


# --- Helper Function to Calculate and Log Metrics (Modified) ---
def calculate_and_log_metrics(true_labels, predicted_labels, class_names, dataset_name, cm_csv_path, cm_png_path):
    """Calculates classification report and confusion matrix, logs them, saves CSV and PNG."""
    write_log(f"\n--- Detailed Metrics for {dataset_name} Set ---")
    if not class_names:
        write_log(f"ERROR: Cannot generate metrics for {dataset_name}. Class names list is empty.")
        return None, None
    if not isinstance(true_labels, (list, np.ndarray)) or not isinstance(predicted_labels, (list, np.ndarray)):
        write_log(f"ERROR: Invalid input types for labels/predictions in {dataset_name}. Must be lists or numpy arrays.")
        return None, None
    if len(true_labels) == 0 or len(predicted_labels) == 0:
        write_log(f"No labels/predictions found for {dataset_name} (length is zero). Skipping detailed metrics.")
        return None, None
    if len(true_labels) != len(predicted_labels):
         write_log(f"ERROR: Mismatch in number of true labels ({len(true_labels)}) and predicted labels ({len(predicted_labels)}) for {dataset_name}.")
         return None, None

    # Ensure labels are numpy arrays
    true_labels_np = np.array(true_labels)
    predicted_labels_np = np.array(predicted_labels)

    try:
        # --- Classification Report ---
        # Generate report using sklearn
        report = classification_report(
            true_labels_np,
            predicted_labels_np,
            target_names=class_names,
            digits=4,
            zero_division=0 # Avoid warnings for classes with no predictions/support
        )
        write_log(f"Classification Report ({dataset_name}):\n{report}")

        # --- Confusion Matrix ---
        # Ensure labels range from 0 to num_classes-1 for confusion_matrix indices
        labels_for_cm = list(range(len(class_names)))
        cm = confusion_matrix(true_labels_np, predicted_labels_np, labels=labels_for_cm)
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

        # --- Save Confusion Matrix CSV ---
        try:
            cm_df.to_csv(cm_csv_path)
            write_log(f"Confusion Matrix ({dataset_name}) saved to CSV: {cm_csv_path}")
        except Exception as e:
            write_log(f"ERROR: Could not save confusion matrix CSV for {dataset_name} to {cm_csv_path}. Reason: {e}")
            write_log(traceback.format_exc())

        # --- Save Confusion Matrix PNG ---
        try:
            plt.style.use('default') # Use default style for heatmap clarity
            fig_height = max(6, len(class_names) * 0.6)
            fig_width = max(8, len(class_names) * 0.8)
            plt.figure(figsize=(fig_width, fig_height))

            # Create heatmap
            sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True, annot_kws={"size": 10}) # 'd' for integer format

            plt.title(f'Confusion Matrix - {dataset_name} Set', fontsize=14, pad=20)
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.xticks(rotation=45, ha='right', fontsize=10) # Adjust rotation and alignment
            plt.yticks(rotation=0, fontsize=10)
            plt.tight_layout() # Adjust layout to prevent labels overlapping
            plt.savefig(cm_png_path, dpi=150) # Save with decent resolution
            write_log(f"Confusion Matrix ({dataset_name}) saved to PNG: {cm_png_path}")
        except Exception as e:
            write_log(f"ERROR: Could not generate or save confusion matrix PNG for {dataset_name} to {cm_png_path}. Reason: {e}")
            write_log(traceback.format_exc())
        finally:
            plt.close() # Close the plot figure to free memory

        return report, cm_df

    except ValueError as ve:
         write_log(f"ERROR calculating metrics for {dataset_name}: {ve}. This might happen if predicted labels contain classes not present in true labels (or vice versa in some cases).")
         write_log(f"Unique True Labels: {np.unique(true_labels_np)}")
         write_log(f"Unique Predicted Labels: {np.unique(predicted_labels_np)}")
         write_log(f"Expected Class Indices: {labels_for_cm}")
         write_log(traceback.format_exc())
         return None, None
    except Exception as e:
        write_log(f"UNEXPECTED ERROR calculating or logging metrics for {dataset_name}: {e}")
        write_log(traceback.format_exc())
        return None, None


# --- Main Execution Guard ---
if __name__ == '__main__':
    # This needs to be the first thing in the main block for Windows multiprocessing
    freeze_support()

    start_time_script = time.time()
    write_log(f"Script started. Log directory: {log_dir}")
    write_log(f"Using device: {DEVICE}")
    write_log(f"Number of CPU workers for DataLoader: {NUM_WORKERS}")

    # --- Load Dataframes ---
    write_log("\n--- Loading Data ---")
    df_train = load_csv_safe(TRAIN_CSV_PATH, "train")
    df_val = load_csv_safe(VAL_CSV_PATH, "validation")
    df_test = load_csv_safe(TEST_CSV_PATH, "test")

    # Exit if essential dataframes are missing
    if df_train is None or df_val is None or df_test is None:
        write_log("FATAL ERROR: One or more essential data CSV files failed to load or are empty. Exiting.")
        exit()

    # Optional: Duplicate training data (Uncomment if needed)
    # original_train_size = len(df_train)
    # df_train = pd.concat([df_train, df_train], ignore_index=True)
    # write_log(f"Duplicated training data. Original size: {original_train_size}, New size (doubled): {len(df_train)}")

    write_log("Data loading successful.")

    # --- Label Mapping ---
    GRADE_MAP = {}
    NUM_GRADES = 0
    CLASS_NAMES = []
    try:
        write_log("\n--- Defining Label Mapping ---")
        # Combine unique grades from all splits, handling potential NaN values explicitly
        all_grades_series = pd.concat([df_train['Grade'], df_val['Grade'], df_test['Grade']]).dropna().astype(str)
        # Get unique grades and sort them
        all_grades_unique = all_grades_series.unique()
        # Sort grades for consistent mapping (important for confusion matrix labels)
        # Attempt numeric sort first, treat non-numeric as strings placed last
        try:
             sorted_grades = sorted(all_grades_unique, key=lambda x: (0, int(x)) if x.isdigit() else (1, x))
        except ValueError:
             # Fallback to simple string sort if complex types exist
             write_log("Warning: Could not perform numeric sort on grades, falling back to string sort.")
             sorted_grades = sorted(all_grades_unique)

        if not sorted_grades:
             write_log("FATAL ERROR: No unique, non-NaN grades found in 'Grade' column across all datasets. Exiting.")
             exit()

        GRADE_MAP = {grade: i for i, grade in enumerate(sorted_grades)}
        NUM_GRADES = len(GRADE_MAP)
        CLASS_NAMES = sorted_grades # Use the sorted list directly

        write_log(f"GRADE_MAP: {GRADE_MAP}")
        write_log(f"Class Names (for reports/plots): {CLASS_NAMES}")
        write_log(f"Number of unique grades (NUM_GRADES): {NUM_GRADES}")

    except KeyError:
        write_log("FATAL ERROR: 'Grade' column not found in one of the CSV files. Please check CSV headers. Exiting.")
        exit()
    except Exception as map_err:
        write_log(f"FATAL ERROR during Grade mapping: {map_err}. Exiting.")
        write_log(traceback.format_exc())
        exit()


    # --- Create Datasets and DataLoaders ---
    try:
        write_log("\n--- Creating Datasets and DataLoaders ---")
        train_dataset = TyphoonGradeDataset(df_train, GRADE_MAP, transform=train_transform)
        val_dataset = TyphoonGradeDataset(df_val, GRADE_MAP, transform=val_test_transform)
        test_dataset = TyphoonGradeDataset(df_test, GRADE_MAP, transform=val_test_transform)

        # Check if datasets are empty after filtering
        if len(train_dataset) == 0:
            write_log("FATAL ERROR: Training dataset is empty after filtering for valid grades. Check input data and GRADE_MAP. Exiting.")
            exit()
        if len(val_dataset) == 0:
            write_log("WARNING: Validation dataset is empty after filtering. Validation steps will be skipped.")
        if len(test_dataset) == 0:
            write_log("WARNING: Test dataset is empty after filtering. Final testing will be skipped.")

        pin_memory_flag = torch.cuda.is_available() # Automatically enable if CUDA is available
        # Use persistent workers only if num_workers > 0
        persistent_workers_flag = NUM_WORKERS > 0

        # Create DataLoaders using the globally defined collate function
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=pin_memory_flag,
                                  collate_fn=collate_fn_skip_none,
                                  persistent_workers=persistent_workers_flag,
                                  drop_last=True) # Drop last incomplete batch in training
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=pin_memory_flag,
                                collate_fn=collate_fn_skip_none,
                                persistent_workers=persistent_workers_flag)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=pin_memory_flag,
                                 collate_fn=collate_fn_skip_none,
                                 persistent_workers=persistent_workers_flag)

        write_log("Datasets and DataLoaders created successfully.")
        write_log(f"  Train samples: {len(train_dataset)}, Train batches: {len(train_loader)} (drop_last=True)")
        write_log(f"  Validation samples: {len(val_dataset)}, Validation batches: {len(val_loader)}")
        write_log(f"  Test samples: {len(test_dataset)}, Test batches: {len(test_loader)}")
        write_log(f"  Using num_workers = {NUM_WORKERS}, persistent_workers = {persistent_workers_flag}")
        write_log(f"  Pin memory enabled: {pin_memory_flag}")

    except ValueError as dataset_init_err:
         # Errors from Dataset __init__ (like empty dataframe)
         write_log(f"FATAL ERROR during Dataset Initialization: {dataset_init_err}")
         write_log("Check GRADE_MAP, 'Grade' column values (NaNs?), image file paths, and filtering logic.")
         exit()
    except Exception as dataset_err:
        write_log(f"FATAL ERROR creating Datasets or DataLoaders: {dataset_err}")
        write_log("Check file paths, permissions, data integrity, and system memory.")
        write_log(traceback.format_exc())
        exit()

    # --- Model, Optimizer, Scheduler ---
    try:
        write_log("\n--- Initializing Model, Optimizer, and Scheduler ---")
        model = ViTGradeClassifier(NUM_GRADES).to(DEVICE)
        write_log(f"Model '{type(model).__name__}' initialized and moved to {DEVICE}.")

        # Optimizer
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        write_log(f"Optimizer: AdamW (LR={LEARNING_RATE}, Weight Decay={WEIGHT_DECAY})")

        # Scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, verbose=False)
        write_log(f"Scheduler: ReduceLROnPlateau (mode=max, factor={SCHEDULER_FACTOR}, patience={SCHEDULER_PATIENCE})")

        # Separate criterion for evaluation (without label smoothing)
        criterion_eval = nn.CrossEntropyLoss().to(DEVICE)

    except Exception as model_setup_err:
        write_log(f"FATAL ERROR setting up Model, Optimizer, or Scheduler: {model_setup_err}")
        write_log(traceback.format_exc())
        exit()

    # --- Training Loop ---
    best_val_acc = 0.0
    epochs_no_improve = 0
    # Lists to store metrics for plotting
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    epochs_completed = 0 # Track actual epochs run

    write_log(f"\n--- Starting Training ---")
    write_log(f"Model: ViT Small (vit_small_patch16_224)")
    write_log(f"Task: Grade Classification Only ({NUM_GRADES} classes)")
    write_log(f"Total Epochs: {EPOCHS}")
    write_log(f"Batch Size: {BATCH_SIZE}")
    write_log(f"Initial Learning Rate: {LEARNING_RATE}")
    write_log(f"Weight Decay: {WEIGHT_DECAY}")
    write_log(f"Label Smoothing: {criterion_clf.label_smoothing}")
    write_log(f"Scheduler Patience: {SCHEDULER_PATIENCE}, Factor: {SCHEDULER_FACTOR}")
    write_log(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
    write_log(f"Model Save Path: {MODEL_SAVE_PATH}")
    write_log("-" * 80)
    # Log header for the epoch table
    header = f"{'Epoch':<7} | {'Train Loss':<12} | {'Train Acc (%)':<15} | {'Val Loss':<10} | {'Val Acc (%)':<13} | {'LR':<10} | {'Time (s)':<8}"
    write_log(header)
    write_log("-" * len(header))

    # Ensure model save directory exists
    try:
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    except OSError as e:
        write_log(f"WARNING: Could not create model save directory '{os.path.dirname(MODEL_SAVE_PATH)}'. Error: {e}")
        # Continue, but saving might fail later


    for epoch in range(EPOCHS):
        epoch_num = epoch + 1
        epochs_completed += 1 # Increment counter for epochs actually started
        model.train() # Set model to training mode
        running_loss = 0.0
        train_correct_grades = 0
        train_total_samples = 0
        batches_processed_train = 0
        start_time_epoch = time.time()

        # --- Training Batch Loop ---
        for i, batch in enumerate(train_loader):
            batch_idx = i + 1
            # Handle None batches potentially returned by collate_fn
            if batch is None:
                 write_log(f"Warning [Train Epoch {epoch_num}]: Skipping None batch returned by dataloader at index {i}.")
                 continue
            try:
                images = batch['image'].to(DEVICE, non_blocking=pin_memory_flag)
                labels_grade = batch['labels']['grade'].to(DEVICE, non_blocking=pin_memory_flag)

                # Check batch size validity
                current_batch_size = images.size(0)
                if current_batch_size == 0:
                    write_log(f"Warning [Train Epoch {epoch_num}, Batch {batch_idx}]: Skipping batch with size 0.")
                    continue

                # Zero gradients
                optimizer.zero_grad(set_to_none=True) # More memory efficient

                # Forward pass
                outputs = model(images)

                # Calculate loss
                loss = calculate_loss(outputs, {'grade': labels_grade}, batch_idx, epoch_num)

                # Handle NaN/Inf loss returned from calculate_loss
                if torch.isnan(loss):
                    write_log(f"ERROR [Train Epoch {epoch_num}, Batch {batch_idx}]: NaN loss encountered. Stopping training.")
                    # Consider alternative strategies like skipping update or reducing LR drastically
                    exit() # Stop training if NaN loss occurs

                # Backward pass and optimization
                loss.backward()
                # Optional: Gradient Clipping (uncomment if needed)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # --- Statistics Update ---
                running_loss += loss.item() * current_batch_size # Weighted by batch size
                _, predicted_grades = torch.max(outputs['grade'], 1)
                train_total_samples += current_batch_size
                train_correct_grades += (predicted_grades == labels_grade).sum().item()
                batches_processed_train += 1

                # Optional: Log progress within epoch
                # if batch_idx % 50 == 0: # Log every 50 batches
                #     write_log(f"  Epoch {epoch_num}, Batch {batch_idx}/{len(train_loader)}: Current Avg Loss: {running_loss / train_total_samples:.4f}")

            # Handle critical errors like file not found during training iteration
            except FileNotFoundError as fnf_err:
                write_log(f"\nFATAL ERROR during Training Batch {batch_idx}/{len(train_loader)} in Epoch {epoch_num}: {fnf_err}")
                write_log("Please check dataset integrity and file paths. Stopping training.")
                exit() # Stop training immediately if a file is missing mid-epoch
            # Handle other potential errors during the batch
            except Exception as batch_err:
                 write_log(f"\nERROR during Training Batch {batch_idx}/{len(train_loader)} in Epoch {epoch_num}: {batch_err}")
                 write_log(traceback.format_exc())
                 # Decide whether to continue or stop based on the error severity
                 # For now, let's try to continue to the next batch
                 write_log("Attempting to continue with the next batch...")
                 continue

        # --- End of Training Epoch ---
        epoch_time = time.time() - start_time_epoch

        # Calculate average training loss and accuracy for the epoch
        epoch_train_loss = running_loss / train_total_samples if train_total_samples > 0 else 0
        epoch_train_acc = 100 * train_correct_grades / train_total_samples if train_total_samples > 0 else 0
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)

        # --- Validation Phase ---
        # Only run validation if the loader is not empty
        if val_loader and len(val_dataset) > 0:
            # Use the evaluation function
            val_loss, val_acc, _, _ = evaluate(model, val_loader, DEVICE, criterion_eval, dataset_name="Validation", epoch_num=epoch_num)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            current_val_loss = val_loss if not np.isnan(val_loss) else float('inf') # Handle potential NaN from eval
            current_val_acc = val_acc if not np.isnan(val_acc) else 0.0
        else:
            # Append placeholder values if validation is skipped
            history['val_loss'].append(float('nan'))
            history['val_acc'].append(0.0)
            current_val_loss = float('inf')
            current_val_acc = 0.0
            write_log(f"Epoch {epoch_num}: Validation skipped (empty dataset/loader).")

        current_lr = optimizer.param_groups[0]['lr']

        # Log epoch results in table format
        log_row = f"{epoch_num:<7} | {epoch_train_loss:<12.4f} | {epoch_train_acc:<15.2f} | {current_val_loss:<10.4f} | {current_val_acc:<13.2f} | {current_lr:<10.2e} | {epoch_time:<8.2f}"
        write_log(log_row)

        # --- Scheduler Step and Model Saving Logic ---
        if not np.isnan(current_val_acc) and current_val_acc > 0: # Only step/save if validation ran and produced valid acc
            # Scheduler steps based on validation accuracy
            scheduler.step(current_val_acc)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < current_lr:
                write_log(f"        Learning rate reduced by scheduler to {new_lr:.2e}")

            # Check if validation accuracy improved
            if current_val_acc > best_val_acc:
                best_val_acc = current_val_acc
                try:
                    # Save the model state dict
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
                    write_log(f'        ** New best model saved to {MODEL_SAVE_PATH} (Val Grade Acc: {best_val_acc:.2f}%) **')
                    epochs_no_improve = 0 # Reset counter
                except Exception as save_err:
                    write_log(f"        ERROR: Could not save model! Reason: {save_err}")
                    write_log(traceback.format_exc())
            else:
                epochs_no_improve += 1
                write_log(f'        Validation accuracy did not improve. ({epochs_no_improve}/{EARLY_STOPPING_PATIENCE})')
                # Check for early stopping
                if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                    write_log(f"\n--- Early stopping triggered after {epoch_num} epochs due to no improvement in validation accuracy. ---")
                    break # Exit training loop
        else:
            # Log if skipping checks due to no validation data or NaN results
            if not val_loader or len(val_dataset) == 0:
                write_log("        Skipping scheduler/save checks (no validation data).")
            else:
                write_log("        Warning: Validation results were NaN or zero. Skipping scheduler/save/early stopping checks for this epoch.")


    # --- End of Training Loop ---
    write_log("-" * len(header)) # Footer for the table
    write_log("\n--- Training Finished ---")
    if epochs_completed < EPOCHS and epochs_no_improve < EARLY_STOPPING_PATIENCE:
         write_log(f"Training stopped before reaching max epochs due to an error after {epochs_completed} epochs.")
    else:
        write_log(f"Training completed after {epochs_completed} epochs.")
    write_log(f"Best Validation Grade Accuracy achieved during training: {best_val_acc:.2f}%")

    # --- Generate Performance Plots ---
    write_log("\n--- Generating Performance Plots ---")
    plot_metrics(epochs_completed, history['train_loss'], history['val_loss'], history['train_acc'], history['val_acc'], loss_plot_file, accuracy_plot_file)


    # --- Final Evaluation on Validation and Test Set using Best Model ---
    write_log(f"\n--- Final Evaluation Phase using Best Model ---")
    write_log(f"Loading best model state from: {MODEL_SAVE_PATH}")

    if os.path.exists(MODEL_SAVE_PATH):
        try:
            # Initialize a new model instance
            model_to_test = ViTGradeClassifier(NUM_GRADES).to(DEVICE)
            # Load the saved state dict
            model_to_test.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
            write_log("Best model state loaded successfully.")

            final_results = {}

            # --- Evaluate on Validation Set (Final) ---
            if val_loader and len(val_dataset) > 0:
                write_log(f"\n--- Evaluating Best Model on Validation Set ---")
                val_loss_final, val_acc_final, val_true_final, val_pred_final = evaluate(model_to_test, val_loader, DEVICE, criterion_eval, dataset_name="Validation (Final)", epoch_num=-1) # Use epoch -1 for final
                if not np.isnan(val_loss_final):
                    write_log(f'Final Validation Summary: Avg Loss={val_loss_final:.4f}, Accuracy={val_acc_final:.2f}% ({int(accuracy_score(val_true_final, val_pred_final) * len(val_true_final))}/{len(val_true_final)})')
                    # Calculate and Log Validation Metrics (CSV and PNG)
                    calculate_and_log_metrics(val_true_final, val_pred_final, CLASS_NAMES, "Validation", conf_matrix_val_csv_file, conf_matrix_val_png_file)
                    final_results['val_loss'] = val_loss_final
                    final_results['val_acc'] = val_acc_final
                else:
                    write_log("Final Validation Evaluation resulted in NaN loss. Metrics skipped.")
            else:
                write_log("Skipping final evaluation on Validation Set (empty dataset/loader).")


            # --- Evaluate on Test Set ---
            if test_loader and len(test_dataset) > 0:
                write_log(f"\n--- Evaluating Best Model on Test Set ---")
                test_loss_final, test_acc_final, test_true_final, test_pred_final = evaluate(model_to_test, test_loader, DEVICE, criterion_eval, dataset_name="Test", epoch_num=-1) # Use epoch -1 for final
                if not np.isnan(test_loss_final):
                    write_log(f'Final Test Summary: Avg Loss={test_loss_final:.4f}, Accuracy={test_acc_final:.2f}% ({int(accuracy_score(test_true_final, test_pred_final) * len(test_true_final))}/{len(test_true_final)})')
                    # Calculate and Log Test Metrics (CSV and PNG)
                    calculate_and_log_metrics(test_true_final, test_pred_final, CLASS_NAMES, "Test", conf_matrix_test_csv_file, conf_matrix_test_png_file)
                    final_results['test_loss'] = test_loss_final
                    final_results['test_acc'] = test_acc_final
                else:
                     write_log("Final Test Evaluation resulted in NaN loss. Metrics skipped.")
            else:
                write_log("Skipping final evaluation on Test Set (empty dataset/loader).")

            # --- Final Performance Summary Log ---
            write_log(f"\n--- Final Performance Summary (Best Model) ---")
            if 'val_acc' in final_results:
                write_log(f"  Validation Set: Loss={final_results.get('val_loss', 'N/A'):.4f}, Accuracy={final_results.get('val_acc', 'N/A'):.2f}%")
            if 'test_acc' in final_results:
                write_log(f"  Test Set:       Loss={final_results.get('test_loss', 'N/A'):.4f}, Accuracy={final_results.get('test_acc', 'N/A'):.2f}%")
            write_log(f"Detailed reports, plots, and confusion matrices saved in: {log_dir}")

        except FileNotFoundError:
             write_log(f"ERROR: Model file disappeared before final evaluation: {MODEL_SAVE_PATH}")
        except Exception as load_eval_err:
           write_log(f"ERROR during model loading or final evaluation: {load_eval_err}")
           write_log(traceback.format_exc())
    else:
        write_log(f"ERROR: Best model file not found at {MODEL_SAVE_PATH}. Skipping final evaluation.")

    end_time_script = time.time()
    total_time_script = end_time_script - start_time_script
    write_log(f"\n--- Script Finished ---")
    write_log(f"Total execution time: {total_time_script:.2f} seconds ({time.strftime('%H:%M:%S', time.gmtime(total_time_script))}).")

# --- END OF FILE Vit_Grade.py ---