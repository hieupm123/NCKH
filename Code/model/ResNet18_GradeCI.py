# --- START OF FILE ResNet18_GradeCI.py ---

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd
from PIL import Image, UnidentifiedImageError # Added UnidentifiedImageError
import os
import numpy as np
import io
import time
import random
from collections import OrderedDict # For ordered table headers
import multiprocessing # Import multiprocessing

# --- New Imports ---
from tabulate import tabulate # For live table
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Global Constants / Configurations ---
# Define constants and paths that might be needed globally or are fundamental configs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = 'model/best_typhoon_model_resnet18_grade_cinumber_only.pth'
TRAIN_CSV_PATH = 'dataset/train_csv_4.csv'
VAL_CSV_PATH = 'dataset/val_csv_4.csv'
TEST_CSV_PATH = 'dataset/test_csv_4.csv'
IMG_SIZE = 224
SCALE_FACTOR = 10.0 # CInumber scaling

# --- Logging Setup ---
# Define directories globally, but create them inside the main block
log_dir = "LLog"
results_dir = "ResNet18_CI" # Directory for metrics and plots
log_file = os.path.join(log_dir, "training_log_Resnet18_GradeCINumberOnly_WithEval.txt")

def write_log(message):
    """Writes a message to the log file and prints it."""
    # Ensure log directory exists before writing
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"{timestamp} - {message}"
    try:
        with open(log_file, "a", encoding='utf-8') as f:
            f.write(log_message + "\n")
        print(log_message)
    except Exception as e:
        print(f"{timestamp} - ERROR writing to log file: {e}")
        print(f"{timestamp} - Original message: {message}") # Still print original message

# --- Helper Function Definitions ---

def load_csv_safe(path, name):
    """Safely loads a CSV file, logs status, and checks required columns."""
    try:
        df = pd.read_csv(path)
        write_log(f"Successfully loaded {name} data from: {path}. Samples: {len(df)}")
        required_cols = ['file_name', 'Grade', 'CInumber']
        if not all(col in df.columns for col in required_cols):
             missing = [col for col in required_cols if col not in df.columns]
             raise ValueError(f"Missing required columns in {path}: {missing}")
        # Basic check for CInumber data type (optional but good)
        if not pd.api.types.is_numeric_dtype(df['CInumber']):
             write_log(f"Warning: CInumber column in {path} might not be purely numeric.")
        return df
    except FileNotFoundError:
        write_log(f"ERROR: CSV file not found at {path}")
        return None
    except Exception as e:
        write_log(f"ERROR: Failed to load {name} data from {path}. Error: {e}")
        return None

def random_rotate(img):
    """Applies a random 0, 90, 180, or 270 degree rotation. (Top-level function)"""
    angle = random.choice([0, 90, 180, 270])
    # Ensure transforms.functional is available, assuming transforms is torchvision.transforms
    return transforms.functional.rotate(img, angle)

def collate_fn_skip_none(batch):
    """Collate function that filters out None results (e.g., from missing files)."""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None # Return None if the whole batch was invalid
    try:
        return torch.utils.data.dataloader.default_collate(batch)
    except RuntimeError as e:
        # More specific error catching can be added if needed (e.g., size mismatch)
        write_log(f"ERROR during collate: {e}. Skipping batch.")
        return None # Return None if collation fails

def plot_confusion_matrix(cm, class_names, save_path, title='Confusion Matrix'):
    """Plots and saves a confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=300)
        write_log(f"Confusion matrix saved to: {save_path}")
    except Exception as e:
        write_log(f"ERROR saving confusion matrix to {save_path}: {e}")
    plt.close() # Close the plot to free memory

def evaluate_and_report(model, data_loader, device, loss_weights, class_names, dataset_name="Test"):
    """Evaluates the model, calculates metrics, and generates reports/plots."""
    model.eval()
    total_loss = 0.0; total_samples = 0
    eval_indiv_losses = {k: 0.0 for k in loss_weights.keys()}
    eval_indiv_losses['total'] = 0.0
    batches_processed_eval = 0

    all_preds_grades = []
    all_labels_grades = []

    # Use criterion instances directly here (or pass them as arguments if preferred)
    criterion_clf_eval = nn.CrossEntropyLoss() # No smoothing for eval
    criterion_reg_eval = nn.MSELoss()

    if not data_loader or not hasattr(data_loader, 'dataset') or len(data_loader.dataset) == 0:
        write_log(f"\n--- Evaluating on {dataset_name} set ---")
        write_log(f"Skipping evaluation: {dataset_name} loader is empty or invalid.")
        return None # Indicate evaluation could not be performed

    num_samples = len(data_loader.dataset)
    write_log(f"\n--- Evaluating on {dataset_name} set ({num_samples} samples) ---")

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if batch is None:
                # write_log(f"Skipping None batch {i} during {dataset_name} evaluation.") # Can be verbose
                continue
            try:
                images = batch['image'].to(device, non_blocking=True)
                labels = {k: v.to(device, non_blocking=True) for k, v in batch['labels'].items()}

                outputs = model(images)

                # Calculate loss for this batch (using eval criteria)
                loss_grade = criterion_clf_eval(outputs['grade'], labels['grade'])
                loss_cinumber = criterion_reg_eval(outputs['cinumber'], labels['cinumber'].view_as(outputs['cinumber']))
                batch_total_loss = (loss_weights['grade'] * loss_grade +
                                    loss_weights['cinumber'] * loss_cinumber)
                indiv_losses = {
                    'total': batch_total_loss.item(),
                    'grade': loss_grade.item(),
                    'cinumber': loss_cinumber.item(),
                }

                if torch.isnan(batch_total_loss):
                    write_log(f"WARNING: NaN loss detected during {dataset_name} evaluation batch {i}. Skipping.")
                    continue

                total_loss += batch_total_loss.item()
                for k, v in indiv_losses.items(): eval_indiv_losses[k] += v

                _, predicted_grades = torch.max(outputs['grade'], 1)
                batch_size = labels['grade'].size(0)
                total_samples += batch_size

                all_preds_grades.extend(predicted_grades.cpu().numpy())
                all_labels_grades.extend(labels['grade'].cpu().numpy())
                batches_processed_eval += 1

            except Exception as e:
                write_log(f"ERROR processing batch {i} during {dataset_name} evaluation: {e}. Skipping batch.")
                continue # Skip batch on error

    # --- Calculate and Report Metrics ---
    if total_samples == 0 or batches_processed_eval == 0:
        write_log(f"{dataset_name} Evaluation Summary: No samples successfully processed.")
        return None

    avg_loss = total_loss / batches_processed_eval
    avg_indiv_losses = {k: v / batches_processed_eval for k, v in eval_indiv_losses.items()}

    # Calculate sklearn metrics
    accuracy = accuracy_score(all_labels_grades, all_preds_grades) * 100
    # Ensure class_names correspond to the indices 0..N-1 used in cm/report
    unique_labels_in_split = sorted(np.unique(all_labels_grades + all_preds_grades))
    # Create target names only for labels that actually appear in this split's predictions/true labels
    target_names_eval = [cn for i, cn in enumerate(class_names) if i in unique_labels_in_split] if class_names else None

    # Ensure labels argument in classification_report matches the unique labels present
    report = classification_report(
        all_labels_grades,
        all_preds_grades,
        labels=unique_labels_in_split, # Use only labels present in this dataset split
        target_names=target_names_eval, # Corresponding names
        digits=4,
        zero_division=0
    )
    # Confusion matrix uses all potential classes (0 to N-1) for consistent axes across reports
    cm = confusion_matrix(all_labels_grades, all_preds_grades, labels=range(len(class_names)))

    write_log(f'{dataset_name} Evaluation Summary:')
    write_log(f'  Average Total Loss: {avg_loss:.4f}')
    write_log(f'  Overall Grade Accuracy: {accuracy:.2f}% ({accuracy_score(all_labels_grades, all_preds_grades, normalize=True):.4f})')
    write_log(f'  Avg Individual Losses: { {k: f"{v:.4f}" for k,v in avg_indiv_losses.items()} }')

    # Save classification report
    report_path = os.path.join(results_dir, f"{dataset_name.lower().replace(' ', '_')}_classification_report.txt")
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"--- {dataset_name} Set Classification Report ---\n\n")
            f.write(f"Overall Accuracy: {accuracy:.4f}%\n\n")
            f.write(f"Labels present in this evaluation: {unique_labels_in_split}\n")
            f.write(f"Corresponding target names: {target_names_eval}\n\n")
            f.write(report)
        write_log(f"Classification report saved to: {report_path}")
    except Exception as e:
        write_log(f"ERROR saving classification report to {report_path}: {e}")

    # Plot and save confusion matrix
    # Use all class names for consistent plot axes
    cm_path = os.path.join(results_dir, f"{dataset_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_path, title=f'{dataset_name} Set Confusion Matrix (Acc: {accuracy:.2f}%)')

    return {"accuracy": accuracy, "report": report, "cm": cm, "loss": avg_loss}


# --- Dataset Class Definition ---
class TyphoonDataset(Dataset):
    def __init__(self, dataframe, transform, grade_map):
        if dataframe is None or dataframe.empty:
             # Log error and raise value error immediately
             log_msg = "CRITICAL: Input dataframe to TyphoonDataset is None or empty."
             write_log(log_msg)
             raise ValueError(log_msg)

        self.dataframe = dataframe.copy() # Use copy to avoid modifying original df
        self.transform = transform
        self.grade_map = grade_map

        initial_len_total = len(self.dataframe)
        num_filtered_grade = 0
        num_filtered_cinum = 0
        num_filtered_path = 0

        # 1. Filter rows with grades not in the map
        unknown_grades_mask = ~self.dataframe['Grade'].isin(self.grade_map.keys())
        if unknown_grades_mask.any():
            unknown_grades = self.dataframe.loc[unknown_grades_mask, 'Grade'].unique()
            write_log(f"INFO: Filtering out {unknown_grades_mask.sum()} samples with unknown grades: {list(unknown_grades)}")
            self.dataframe = self.dataframe[~unknown_grades_mask]
            num_filtered_grade = unknown_grades_mask.sum()

        # 2. Filter rows with missing CInumber (NaN or None)
        nan_cinumber_mask = self.dataframe['CInumber'].isna()
        if nan_cinumber_mask.any():
             num_filtered_cinum = nan_cinumber_mask.sum()
             write_log(f"INFO: Filtering out {num_filtered_cinum} samples with missing CInumber.")
             self.dataframe = self.dataframe[~nan_cinumber_mask]

        # 3. (Optional but recommended) Check file paths *during initialization*
        # This is slower at startup but prevents many errors during training iteration
        if not self.dataframe.empty: # Only check if there are rows left
            exists_mask = self.dataframe['file_name'].apply(os.path.isfile)
            if not exists_mask.all():
                num_filtered_path = (~exists_mask).sum()
                missing_files = self.dataframe.loc[~exists_mask, 'file_name'].tolist()
                write_log(f"WARNING: Filtering out {num_filtered_path} samples with non-existent or non-file image paths during dataset init.")
                # Optionally log first few missing files for debugging
                # if missing_files: write_log(f"  Examples: {missing_files[:5]}")
                self.dataframe = self.dataframe[exists_mask]

        # Reset index after all filtering
        self.dataframe.reset_index(drop=True, inplace=True)

        final_len = len(self.dataframe)
        total_filtered = initial_len_total - final_len
        if total_filtered > 0:
             write_log(f"Dataset Init Summary: Started with {initial_len_total}, "
                       f"Filtered: {num_filtered_grade} (Grade), {num_filtered_cinum} (CInum), {num_filtered_path} (Path). "
                       f"Final size: {final_len}")
        elif initial_len_total > 0 :
             write_log(f"Dataset Init: All {initial_len_total} samples passed initial filtering.")
        # else: initial_len_total was 0

        # Raise error if dataset becomes empty after filtering
        if final_len == 0 and initial_len_total > 0 :
             log_msg = "CRITICAL: Dataset is empty after filtering steps. Cannot proceed with training."
             write_log(log_msg)
             raise ValueError(log_msg)


    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Basic check - should be caught by DataLoader if index is truly out of bounds
        if not (0 <= idx < len(self.dataframe)):
             write_log(f"ERROR: Attempted to access invalid index {idx} in dataset of size {len(self.dataframe)}")
             # Return None, Dataloader's collate_fn should handle this, although it indicates a deeper issue
             return None
        row = self.dataframe.iloc[idx]
        img_path = row['file_name']

        try:
            # File existence was checked in __init__, but opening can still fail
            image = Image.open(img_path).convert('RGB')

        # Specific exceptions during image open/convert
        except FileNotFoundError: # Should be rare if init check worked, but handle defensively
             write_log(f"ERROR: FileNotFoundError (post-init check): {img_path} (index {idx}). Skipping.")
             return None
        except IsADirectoryError:
            write_log(f"ERROR: Path is a directory: {img_path} (index {idx}). Skipping.")
            return None
        except UnidentifiedImageError:
             write_log(f"ERROR: Cannot identify image file (corrupted?): {img_path} (index {idx}). Skipping.")
             return None
        except Exception as e:
             # Catch other potential PIL/OS errors during open
             write_log(f"ERROR opening/reading image {img_path} (index {idx}): {e}. Skipping.")
             return None

        # Apply transforms only if image loaded successfully
        try:
            image = self.transform(image)
        except Exception as e:
            write_log(f"ERROR applying transforms to image {img_path} (index {idx}): {e}. Skipping.")
            return None

        # Process labels (should be valid after initial filtering)
        try:
            grade_label = row['Grade']
            # Grade map check is slightly redundant if init worked, but safe
            if grade_label not in self.grade_map:
                write_log(f"ERROR: Grade '{grade_label}' from file {img_path} (idx {idx}) not in GRADE_MAP post-init. Skipping sample.")
                return None
            grade = torch.tensor(self.grade_map[grade_label], dtype=torch.long)

            # CInumber should be valid float after dropna/type check in init
            cinumber_val = row['CInumber']
            if cinumber_val is None or np.isnan(cinumber_val): # Should not happen post-init
                 write_log(f"ERROR: Invalid CInumber '{cinumber_val}' from file {img_path} (idx {idx}) post-init. Skipping sample.")
                 return None
            cinumber = torch.tensor(cinumber_val / SCALE_FACTOR, dtype=torch.float32)

        except KeyError as e:
             write_log(f"ERROR: Missing expected column '{e}' for row index {idx}, file {img_path}. Skipping.")
             return None
        except Exception as e:
             write_log(f"ERROR processing labels for row index {idx}, file {img_path}: {e}. Skipping.")
             return None


        return {
            'image': image,
            'labels': {
                'grade': grade,
                'cinumber': cinumber.unsqueeze(0), # Ensure [1] shape for MSELoss compatibility
            }
        }


# --- Model Class Definitions ---
class AuxiliaryHead(nn.Module):
    """Auxiliary head for intermediate features."""
    def __init__(self, in_channels, num_outputs, is_regression=False):
        super().__init__()
        self.is_regression = is_regression
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.4) # Consider tuning dropout
        self.fc = nn.Linear(in_channels, num_outputs)

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class ResNetMultiTask(nn.Module):
    """ResNet18-based model for Grade classification and CInumber regression."""
    def __init__(self, num_grades):
        super().__init__()
        # Load ResNet18 with pre-trained weights
        base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Extract ResNet layers up to layer 4
        self.conv1 = base_model.conv1; self.bn1 = base_model.bn1
        self.relu = base_model.relu; self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1; self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3; self.layer4 = base_model.layer4

        # Global Average Pooling after layer 4
        self.avgpool = base_model.avgpool
        self.flatten = nn.Flatten()

        # Feature dimensions
        aux_feature_dim = 512 # Output channels of ResNet18 layer4 blocks
        final_feature_dim = base_model.fc.in_features # 512 for ResNet18

        # Auxiliary Head for CINumber (regression) - takes features from layer 4
        self.cinumber_head = AuxiliaryHead(aux_feature_dim, 1, is_regression=True)

        # Main Head for Grade classification
        # Concatenates globally pooled final features and the CInumber prediction
        concatenated_feature_dim = final_feature_dim + 1 # 512 (pooled) + 1 (CInum)
        self.grade_head = nn.Sequential(
            nn.BatchNorm1d(concatenated_feature_dim), # Batch norm on concatenated features
            nn.Dropout(0.5),
            nn.Linear(concatenated_feature_dim, num_grades) # Final classification layer
        )

    def forward(self, x):
        # Backbone feature extraction
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        features_l4 = self.layer4(x) # Output: [B, 512, H/32, W/32]

        # Auxiliary Output (from Layer 4 features before pooling)
        out_cinumber = self.cinumber_head(features_l4) # Output: [B, 1]

        # Main Path (Global Average Pooling on Layer 4 features)
        final_features_pooled = self.avgpool(features_l4) # Output: [B, 512, 1, 1]
        final_features_flat = self.flatten(final_features_pooled) # Output: [B, 512]

        # Concatenate main features and auxiliary output
        concatenated_features = torch.cat([final_features_flat, out_cinumber], dim=1) # Output: [B, 513]

        # Main Output (Grade Prediction)
        out_grade = self.grade_head(concatenated_features) # Output: [B, num_grades]

        outputs = {'grade': out_grade, 'cinumber': out_cinumber}
        return outputs

# --- Loss Function Definition ---
# Define criteria globally or pass them to the function
criterion_clf_train = nn.CrossEntropyLoss(label_smoothing=0.1)
criterion_reg_train = nn.MSELoss()

def calculate_combined_loss(outputs, labels, weights):
    """Calculates the combined weighted loss for training."""
    loss_grade = criterion_clf_train(outputs['grade'], labels['grade'])

    # Ensure cinumber label has the right shape [batch_size, 1] for MSELoss compatibility
    loss_cinumber = criterion_reg_train(outputs['cinumber'], labels['cinumber'].view_as(outputs['cinumber']))

    total_loss = (weights['grade'] * loss_grade +
                  weights['cinumber'] * loss_cinumber)

    individual_losses = {
        'total': total_loss.item() if not torch.isnan(total_loss) else float('nan'),
        'grade': loss_grade.item() if not torch.isnan(loss_grade) else float('nan'),
        'cinumber': loss_cinumber.item() if not torch.isnan(loss_cinumber) else float('nan'),
    }
    return total_loss, individual_losses


# ==============================================================================
# --- Main Execution Block ---
# ==============================================================================
if __name__ == '__main__':
    # Add freeze_support() right at the beginning for Windows/macOS compatibility with multiprocessing
    multiprocessing.freeze_support()

    # --- Create Directories ---
    # Create necessary directories defined globally
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True) # Ensure model directory exists

    # --- Start Logging ---
    write_log("--- Starting Script Execution ---")
    write_log(f"PyTorch Version: {torch.__version__}")
    write_log(f"Using device: {DEVICE}")
    if DEVICE.type == 'cuda':
        write_log(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        write_log(f"CUDA Available: {torch.cuda.is_available()}")

    # --- 1. Configuration & Setup ---
    # Hyperparameters
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 64
    EPOCHS = 64
    WEIGHT_DECAY = 1e-5
    SCHEDULER_PATIENCE = 50
    SCHEDULER_FACTOR = 0.2
    EARLY_STOPPING_PATIENCE = 50
    W_GRADE = 2.0
    W_CINUMBER = 10.0
    loss_weights = {'grade': W_GRADE, 'cinumber': W_CINUMBER}

    # Load dataframes
    df_train = load_csv_safe(TRAIN_CSV_PATH, "train")
    df_val = load_csv_safe(VAL_CSV_PATH, "validation")
    df_test = load_csv_safe(TEST_CSV_PATH, "test")

    # Check if data loading failed critically
    if df_train is None or df_val is None or df_test is None:
        write_log("CRITICAL ERROR: One or more required data CSV files failed to load or had missing columns. Please check paths and file contents. Exiting.")
        raise SystemExit("Could not load necessary CSV data files or required columns are missing.")

    # --- *** Training Data Duplication *** ---
    write_log("\n--- Processing Training Data ---")
    original_train_size = len(df_train)
    df_train = pd.concat([df_train, df_train], ignore_index=True) # DUPLICATION ACTIVE
    new_train_size = len(df_train)
    write_log(f"DUPLICATED training data. Original size: {original_train_size}, New size: {new_train_size}")


    # --- Label Mapping ---
    # Consolidate all grades from loaded dataframes to ensure map covers everything
    all_grades_list = []
    if df_train is not None: all_grades_list.append(df_train['Grade'])
    if df_val is not None: all_grades_list.append(df_val['Grade'])
    if df_test is not None: all_grades_list.append(df_test['Grade'])

    if not all_grades_list:
        raise SystemExit("CRITICAL ERROR: No grade data found in any loaded CSV file. Cannot proceed.")

    all_grades = sorted(pd.concat(all_grades_list).unique())
    GRADE_MAP = {grade: i for i, grade in enumerate(all_grades)}
    CLASS_NAMES = list(GRADE_MAP.keys()) # Get class names in the mapped order
    NUM_GRADES = len(GRADE_MAP) # Define NUM_GRADES based on the actual data
    write_log("\n--- Label Mappings ---")
    write_log(f"GRADE_MAP: {GRADE_MAP}")
    write_log(f"CLASS_NAMES (for metrics/plots): {CLASS_NAMES}")
    write_log(f"Number of Grade classes (NUM_GRADES): {NUM_GRADES}")


    # --- 2. Dataset & DataLoader Instantiation ---
    write_log("\n--- Defining Data Transforms ---")
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Lambda(random_rotate), # Use the named function here
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    write_log(f"Train Transforms: {train_transform}")

    val_test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    write_log(f"Validation/Test Transforms: {val_test_transform}")

    write_log("\n--- Creating Datasets ---")
    try:
        # Datasets will now perform filtering during __init__
        train_dataset = TyphoonDataset(df_train, transform=train_transform, grade_map=GRADE_MAP)
        val_dataset = TyphoonDataset(df_val, transform=val_test_transform, grade_map=GRADE_MAP)
        test_dataset = TyphoonDataset(df_test, transform=val_test_transform, grade_map=GRADE_MAP)

        # Log sizes *after* potential filtering in __init__
        write_log(f"Train dataset size (after filtering): {len(train_dataset)}")
        write_log(f"Validation dataset size (after filtering): {len(val_dataset)}")
        write_log(f"Test dataset size (after filtering): {len(test_dataset)}")

        # Check if datasets became empty *after* init filtering
        if len(train_dataset) == 0:
             write_log("CRITICAL ERROR: Training dataset is empty after initialization filtering. Check data paths and filters. Exiting.")
             raise SystemExit("Empty training dataset.")
        if len(val_dataset) == 0:
             # Depending on workflow, maybe allow training without validation? For now, treat as error.
             write_log("CRITICAL ERROR: Validation dataset is empty after initialization filtering. Cannot monitor performance or use early stopping. Exiting.")
             raise SystemExit("Empty validation dataset.")
             # Alternatively: write_log("WARNING: Validation dataset empty. Proceeding without validation.")

        # Set num_workers based on CPU cores, but cap it for stability if needed
        num_cpu_cores = os.cpu_count() or 1 # Default to 1 if cpu_count fails
        # Rule of thumb: <= num_cpu_cores. Start lower (e.g., 4 or 8) and increase if I/O is bottleneck.
        # Cap based on system memory and potential overhead. 0 disables multiprocessing.
        NUM_WORKERS = min(4, num_cpu_cores) if DEVICE.type == 'cuda' else 0 # Use 0 workers if only CPU to avoid overhead
        if DEVICE.type == 'cuda' and num_cpu_cores <= 2: NUM_WORKERS = min(2, num_cpu_cores) # Reduce workers on low-core systems

        # Ensure persistent_workers=False if NUM_WORKERS is 0
        persistent_workers_flag = (NUM_WORKERS > 0) and (DEVICE.type == 'cuda') # Recommended for CUDA > 0 workers
        write_log(f"Using num_workers = {NUM_WORKERS} for DataLoaders (persistent_workers={persistent_workers_flag}).")

        # DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True, # pin_memory is useful with CUDA
            persistent_workers=persistent_workers_flag,
            collate_fn=collate_fn_skip_none, drop_last=True # drop_last can help with BatchNorm stability
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
            persistent_workers=persistent_workers_flag,
            collate_fn=collate_fn_skip_none
        )
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
            persistent_workers=persistent_workers_flag,
            collate_fn=collate_fn_skip_none
        )

        write_log("\n--- DataLoaders Created Successfully ---")

    except ValueError as e: # Catch potential errors from TyphoonDataset init
        write_log(f"CRITICAL ERROR during Dataset creation: {e}")
        raise SystemExit(f"Dataset creation failed: {e}")
    except Exception as e:
        write_log(f"CRITICAL UNEXPECTED ERROR during Dataset/DataLoader creation: {e}")
        # Optionally log traceback for debugging unexpected errors
        # import traceback
        # write_log(traceback.format_exc())
        raise SystemExit(f"Unexpected error during data setup: {e}")


    # --- 3. Model Definition (Instantiation) ---
    write_log("\n--- Initializing Model ---")
    model = ResNetMultiTask(NUM_GRADES).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    write_log(f"Model: ResNetMultiTask (ResNet18 Base)")
    write_log(f"Tasks: Grade (Classifier, {NUM_GRADES} classes), CINumber (Regressor)")
    write_log(f"Model Parameters: Total={total_params:,}, Trainable={trainable_params:,}")


    # --- 4. Optimizer, Scheduler, Loss (Instantiation/Setup) ---
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, verbose=False, min_lr=1e-7) # Add min_lr
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7) # Alternative: Cosine Annealing

    write_log("\n--- Training Setup ---")
    write_log(f"Optimizer: AdamW (LR={LEARNING_RATE}, WeightDecay={WEIGHT_DECAY})")
    write_log(f"Scheduler: ReduceLROnPlateau (Mode=max, Factor={SCHEDULER_FACTOR}, Patience={SCHEDULER_PATIENCE}, MinLR=1e-7)")
    # write_log(f"Scheduler: CosineAnnealingLR (T_max={EPOCHS}, MinLR=1e-7)") # If using Cosine
    write_log(f"Loss Weights: {loss_weights}")
    write_log(f"Label Smoothing (Train): 0.1")
    write_log(f"Epochs: {EPOCHS}")
    write_log(f"Batch Size: {BATCH_SIZE}")
    write_log(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")


    # --- 5. Training & Validation Loop ---
    best_val_acc = 0.0
    epochs_no_improve = 0
    history = [] # Store epoch history for table
    training_start_time = time.time()

    # Table Headers - Use OrderedDict for guaranteed order
    table_headers = OrderedDict([
        ("Epoch", ""),
        ("Train Loss", ""),
        ("Tr Grade Acc (%)", ""), # Shorter name
        ("Val Loss", ""),
        ("Val Grade Acc (%)", ""), # Shorter name
        ("LR", ""),
        ("Time (s)", "")
    ])

    write_log(f"\n--- Starting Training for {EPOCHS} Epochs ---")

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()

        # --- Training Phase ---
        model.train()
        running_loss = 0.0; train_correct_grades = 0; train_total_samples = 0
        epoch_train_indiv_losses = {k: 0.0 for k in loss_weights.keys()}
        epoch_train_indiv_losses['total'] = 0.0
        batches_processed_train = 0
        skipped_batches_train = 0

        # Training loop with batch skipping
        for i, batch in enumerate(train_loader):
            if batch is None:
                skipped_batches_train += 1
                continue # Skip bad batches returned by collate_fn or dataset __getitem__

            try:
                images = batch['image'].to(DEVICE, non_blocking=True)
                labels = {k: v.to(DEVICE, non_blocking=True) for k, v in batch['labels'].items()}

                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                loss, indiv_losses = calculate_combined_loss(outputs, labels, loss_weights)

                # Check for invalid loss BEFORE backward()
                if torch.isnan(loss) or torch.isinf(loss):
                    write_log(f"WARNING: NaN or Inf loss detected during training epoch {epoch+1}, batch {i}. Skipping batch update.")
                    # No optimizer.step() or loss accumulation if loss is invalid
                    skipped_batches_train += 1
                    continue

                loss.backward()
                # Optional: Gradient Clipping (uncomment if exploding gradients are suspected)
                # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # if grad_norm > 1.0: write_log(f"  Gradient norm clipped: {grad_norm:.2f}")
                optimizer.step()

                # Accumulate metrics only for successful steps
                running_loss += loss.item() # Use .item() for scalar loss
                for k, v in indiv_losses.items():
                     # Check if v is valid number (not NaN/inf) before adding
                     if not np.isnan(v) and not np.isinf(v):
                         epoch_train_indiv_losses[k] += v
                     else: # Log if individual loss component was invalid, even if total loss was okay
                         write_log(f"Warning: Invalid value ({v}) for individual loss '{k}' in train batch {i}, epoch {epoch+1}")

                _, predicted_grades = torch.max(outputs['grade'], 1)
                train_total_samples += labels['grade'].size(0)
                train_correct_grades += (predicted_grades == labels['grade']).sum().item()
                batches_processed_train += 1

            except Exception as e:
                write_log(f"ERROR during training batch {i}, epoch {epoch+1}: {e}")
                # Consider adding traceback logging for debugging difficult errors
                # import traceback
                # write_log(traceback.format_exc())
                optimizer.zero_grad() # Ensure grads are zeroed if error occurs mid-batch
                skipped_batches_train += 1
                continue # Skip to next batch

        # --- End of Training Epoch ---
        if skipped_batches_train > 0:
            write_log(f"INFO: Skipped {skipped_batches_train} training batches in epoch {epoch+1} due to errors or None data.")

        if batches_processed_train == 0:
             write_log(f"CRITICAL ERROR: Epoch {epoch+1} had 0 training batches processed successfully. Training cannot continue. Check data loading and processing.")
             # Decide whether to continue (e.g., hoping next epoch works) or stop
             break # Stop training

        epoch_train_loss = running_loss / batches_processed_train
        epoch_train_acc = 100 * train_correct_grades / train_total_samples if train_total_samples > 0 else 0
        # Calculate average individual losses safely
        avg_train_indiv_losses = {k: (v / batches_processed_train) if batches_processed_train > 0 else 0.0
                                   for k, v in epoch_train_indiv_losses.items()}


        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0; val_correct_grades = 0; val_total_samples = 0
        epoch_val_indiv_losses = {k: 0.0 for k in loss_weights.keys()}
        epoch_val_indiv_losses['total'] = 0.0
        batches_processed_val = 0
        skipped_batches_val = 0
        can_validate = True # Assume validation is possible initially

        # Check if val_loader is usable (redundant if checked at init, but safe)
        if not val_loader or not hasattr(val_loader, 'dataset') or len(val_loader.dataset) == 0:
            write_log(f"Epoch [{epoch+1}/{EPOCHS}] - WARNING: Skipping validation phase: Loader invalid or validation dataset empty.")
            can_validate = False
        else:
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    if batch is None:
                        skipped_batches_val += 1
                        continue

                    try:
                        images = batch['image'].to(DEVICE, non_blocking=True)
                        labels = {k: v.to(DEVICE, non_blocking=True) for k, v in batch['labels'].items()}

                        outputs = model(images)
                        # Calculate loss using the same combined function for consistent reporting scale
                        # (Even though it uses training criteria like label smoothing inside)
                        loss, indiv_losses = calculate_combined_loss(outputs, labels, loss_weights)

                        # Check for invalid loss
                        if torch.isnan(loss) or torch.isinf(loss):
                            write_log(f"WARNING: NaN or Inf loss detected during validation epoch {epoch+1}, batch {i}. Skipping batch.")
                            skipped_batches_val += 1
                            continue

                        # Accumulate metrics for successful batches
                        val_loss += loss.item()
                        for k, v in indiv_losses.items():
                           if not np.isnan(v) and not np.isinf(v):
                               epoch_val_indiv_losses[k] += v
                           else:
                               write_log(f"Warning: Invalid value ({v}) for individual loss '{k}' in val batch {i}, epoch {epoch+1}")

                        _, predicted_grades = torch.max(outputs['grade'], 1)
                        val_total_samples += labels['grade'].size(0)
                        val_correct_grades += (predicted_grades == labels['grade']).sum().item()
                        batches_processed_val += 1
                    except Exception as e:
                        write_log(f"ERROR during validation batch {i}, epoch {epoch+1}: {e}")
                        skipped_batches_val += 1
                        continue # Skip batch

            # --- End of Validation Loop for Epoch ---
            if skipped_batches_val > 0:
                write_log(f"INFO: Skipped {skipped_batches_val} validation batches in epoch {epoch+1} due to errors or None data.")

            # Calculate validation metrics
            if batches_processed_val == 0 and val_total_samples == 0: # Check if any samples processed
                if skipped_batches_val == len(val_loader): # All batches were skipped
                    write_log(f"WARNING: Epoch {epoch+1} had 0 validation batches processed successfully (all skipped). Cannot determine validation performance.")
                else: # Loader might be empty or other issue
                    write_log(f"WARNING: Epoch {epoch+1} had 0 validation batches processed successfully. Validation loader might be empty or issues occurred.")
                can_validate = False # Cannot use results if no batches processed
            else:
                # Calculate averages safely, avoid division by zero
                epoch_val_loss = val_loss / batches_processed_val if batches_processed_val > 0 else float('nan')
                epoch_val_acc = 100 * val_correct_grades / val_total_samples if val_total_samples > 0 else 0.0
                avg_val_indiv_losses = {k: (v / batches_processed_val) if batches_processed_val > 0 else 0.0
                                         for k, v in epoch_val_indiv_losses.items()}
                # If loss is NaN even after averaging, validation failed
                if np.isnan(epoch_val_loss):
                     write_log(f"WARNING: Average validation loss is NaN for Epoch {epoch+1}. Treating validation as failed for this epoch.")
                     can_validate = False

        # --- Process Epoch Results (Handle Case Where Validation Failed) ---
        if not can_validate:
            epoch_val_loss = float('nan')
            epoch_val_acc = float('nan')
            avg_val_indiv_losses = {k: float('nan') for k in loss_weights.keys()}
            avg_val_indiv_losses['total'] = float('nan')


        # --- Epoch Summary & Table Update ---
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']

        # Store history for table (use "N/A" for display if validation failed)
        epoch_data = OrderedDict([
            ("Epoch", f"{epoch+1}/{EPOCHS}"),
            ("Train Loss", f"{epoch_train_loss:.4f}"),
            ("Tr Grade Acc (%)", f"{epoch_train_acc:.2f}"),
            ("Val Loss", f"{epoch_val_loss:.4f}" if can_validate else "N/A"),
            ("Val Grade Acc (%)", f"{epoch_val_acc:.2f}" if can_validate else "N/A"),
            ("LR", f"{current_lr:.2e}"),
            ("Time (s)", f"{epoch_time:.1f}")
        ])
        history.append(epoch_data)

        # Log epoch summary
        summary_log = (f"Epoch {epoch+1}/{EPOCHS} | "
                       f"Tr Loss: {epoch_train_loss:.4f}, Tr Acc: {epoch_train_acc:.2f}% | ")
        if can_validate:
            summary_log += (f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}% | ")
        else:
            summary_log += ("Val Loss: N/A, Val Acc: N/A | ")
        summary_log += (f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
        write_log(summary_log)

        # Optional: Log individual losses if needed (e.g., every few epochs)
        # if (epoch + 1) % 10 == 0:
        #      log_str_indiv = f"  Avg Indiv Losses Ep {epoch+1}: Train=" + str({k: f'{v:.3f}' for k, v in avg_train_indiv_losses.items()})
        #      if can_validate:
        #          log_str_indiv += " Val=" + str({k: f'{v:.3f}' for k, v in avg_val_indiv_losses.items()})
        #      write_log(log_str_indiv)


        # --- LR Scheduling and Early Stopping ---
        if can_validate: # Only step/check if validation was successful for this epoch
            # Scheduler Step (based on validation accuracy)
            # If using ReduceLROnPlateau
            scheduler.step(epoch_val_acc)
            # If using CosineAnnealingLR - step happens regardless of metric, usually once per epoch
            # scheduler.step()

            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < current_lr: # Log if LR was reduced by ReduceLROnPlateau
                write_log(f"  Learning rate reduced to {new_lr:.2e}.")

            # Check for improvement and save best model
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                try:
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
                    write_log(f'  ---> New best model saved to {MODEL_SAVE_PATH} (Val Grade Acc: {best_val_acc:.2f}%) <---')
                    epochs_no_improve = 0 # Reset counter
                except Exception as e:
                     write_log(f"ERROR saving model checkpoint: {e}")
            else:
                epochs_no_improve += 1
                write_log(f'  Val Acc ({epoch_val_acc:.2f}%) did not improve vs best ({best_val_acc:.2f}%). Plateau: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}.')
                if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                    write_log(f"\n--- Early stopping triggered after {epoch + 1} epochs (patience {EARLY_STOPPING_PATIENCE}). Best Val Acc: {best_val_acc:.2f}% ---")
                    break # Exit training loop
        else:
            # What to do if validation failed?
            # Option 1: Continue training, hoping validation works next time.
            write_log("  Skipping LR schedule step and early stopping check (validation failed or not possible this epoch).")
            # Option 2: Increment no-improvement counter anyway? (Could stop early if validation consistently fails)
            # epochs_no_improve += 1
            # if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            #     write_log(f"\n--- Early stopping triggered after {epoch + 1} epochs due to lack of valid validation for {EARLY_STOPPING_PATIENCE} epochs. ---")
            #     break


    # --- End of Training ---
    training_duration = time.time() - training_start_time
    write_log("\n--- Training Finished ---")
    write_log(f"Total training time: {training_duration:.2f} seconds ({training_duration/3600:.2f} hours)")
    if history:
        write_log("\n--- Training History Summary ---")
        try:
            history_table = tabulate([list(h.values()) for h in history], headers=table_headers.keys(), tablefmt="plain")
            write_log(history_table)
        except Exception as e:
            write_log(f"Error generating history table: {e}")
    write_log(f"Best Validation Grade Accuracy achieved during training: {best_val_acc:.2f}%")
    write_log(f"Model corresponding to best val acc saved at: {MODEL_SAVE_PATH}")


    # --- 6. Final Evaluation & Metrics ---
    write_log(f"\n--- Starting Final Evaluation ---")
    write_log(f"Loading best model state_dict from: {MODEL_SAVE_PATH}")

    # Check if the best model file exists
    if os.path.exists(MODEL_SAVE_PATH):
        # Instantiate a fresh model instance for testing to ensure no residual state
        model_to_test = ResNetMultiTask(NUM_GRADES)
        model_to_test.to(DEVICE)
        try:
            # Load the saved state dict
            state_dict = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
            # Adapt state_dict if model architecture changed slightly (e.g., removing 'module.' prefix if saved from DataParallel)
            # new_state_dict = OrderedDict()
            # for k, v in state_dict.items():
            #     name = k[7:] if k.startswith('module.') else k
            #     new_state_dict[name] = v
            # model_to_test.load_state_dict(new_state_dict, strict=True)
            model_to_test.load_state_dict(state_dict, strict=True) # Use strict=True initially

            write_log("Best model state_dict loaded successfully for final evaluation.")

            # Evaluate on Validation Set (using the loaded best model)
            val_results = evaluate_and_report(
                model=model_to_test,
                data_loader=val_loader,
                device=DEVICE,
                loss_weights=loss_weights,
                class_names=CLASS_NAMES, # Use the globally defined class names
                dataset_name="Validation (Best Model)"
            )

            # Evaluate on Test Set
            test_results = evaluate_and_report(
                model=model_to_test,
                data_loader=test_loader,
                device=DEVICE,
                loss_weights=loss_weights,
                class_names=CLASS_NAMES,
                dataset_name="Test"
            )

        except FileNotFoundError: # Safety check
             write_log(f"ERROR: Model file disappeared before loading: {MODEL_SAVE_PATH}")
             write_log("Cannot perform final evaluation.")
        except RuntimeError as e:
             write_log(f"ERROR loading state_dict: {e}.")
             write_log("This usually means the saved model's architecture differs from the current `ResNetMultiTask` definition, or the file is corrupt.")
             write_log("Try setting strict=False if the difference is expected (e.g., missing/extra keys), but investigate the cause.")
             write_log("Cannot perform final evaluation with strict=True.")
        except Exception as e:
             write_log(f"An unexpected error occurred during model loading or final evaluation: {e}")
             # import traceback
             # write_log(traceback.format_exc())
             write_log("Cannot perform final evaluation.")
    else:
        write_log(f"ERROR: Best model file not found at {MODEL_SAVE_PATH}. Cannot perform final evaluation.")
        write_log("Ensure training ran long enough to save a model and the path is correct, or if training failed.")

    write_log("\n--- Script Finished ---")

# --- END OF FILE ResNet18_GradeCI.py ---