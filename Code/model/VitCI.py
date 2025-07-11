# --- START OF FILE ViT_Small_GradeCI.py ---

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
# import torchvision.models as models # No longer needed for base ResNet
import pandas as pd
from PIL import Image, UnidentifiedImageError
import os
import numpy as np
import io
import time
import random
from collections import OrderedDict
import multiprocessing

# --- New Imports ---
from tabulate import tabulate
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import timm # Import timm

# --- Global Constants / Configurations ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Adjust model save path name
MODEL_SAVE_PATH = 'model/best_typhoon_model_vitsmall_grade_cinumber_only.pth'
TRAIN_CSV_PATH = 'dataset/train_csv_4.csv'
VAL_CSV_PATH = 'dataset/val_csv_4.csv'
TEST_CSV_PATH = 'dataset/test_csv_4.csv'
IMG_SIZE = 224 # ViT models often use 224x224
SCALE_FACTOR = 10.0 # CInumber scaling

# --- Logging Setup ---
log_dir = "LLog"
results_dir = "ViTSmall_CI" # Directory for metrics and plots for ViT model
log_file = os.path.join(log_dir, "training_log_ViTSmall_GradeCINumberOnly_WithEval.txt")

def write_log(message):
    """Writes a message to the log file and prints it."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"{timestamp} - {message}"
    try:
        with open(log_file, "a", encoding='utf-8') as f:
            f.write(log_message + "\n")
        print(log_message)
    except Exception as e:
        print(f"{timestamp} - ERROR writing to log file: {e}")
        print(f"{timestamp} - Original message: {message}")

# --- Helper Function Definitions --- (Keep load_csv_safe, random_rotate, collate_fn_skip_none, plot_confusion_matrix, evaluate_and_report as they are)

def load_csv_safe(path, name):
    """Safely loads a CSV file, logs status, and checks required columns."""
    try:
        df = pd.read_csv(path)
        write_log(f"Successfully loaded {name} data from: {path}. Samples: {len(df)}")
        required_cols = ['file_name', 'Grade', 'CInumber']
        if not all(col in df.columns for col in required_cols):
             missing = [col for col in required_cols if col not in df.columns]
             raise ValueError(f"Missing required columns in {path}: {missing}")
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
    return transforms.functional.rotate(img, angle)

def collate_fn_skip_none(batch):
    """Collate function that filters out None results (e.g., from missing files)."""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None # Return None if the whole batch was invalid
    try:
        return torch.utils.data.dataloader.default_collate(batch)
    except RuntimeError as e:
        write_log(f"ERROR during collate: {e}. Skipping batch.")
        return None # Return None if collation fails

def plot_confusion_matrix(cm, class_names, save_path, title='Confusion Matrix'):
    """Plots and saves a confusion matrix."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True) # Ensure dir exists
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

    criterion_clf_eval = nn.CrossEntropyLoss()
    criterion_reg_eval = nn.MSELoss()

    if not data_loader or not hasattr(data_loader, 'dataset') or len(data_loader.dataset) == 0:
        write_log(f"\n--- Evaluating on {dataset_name} set ---")
        write_log(f"Skipping evaluation: {dataset_name} loader is empty or invalid.")
        return None

    num_samples = len(data_loader.dataset)
    write_log(f"\n--- Evaluating on {dataset_name} set ({num_samples} samples) ---")

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if batch is None:
                continue
            try:
                images = batch['image'].to(device, non_blocking=True)
                labels = {k: v.to(device, non_blocking=True) for k, v in batch['labels'].items()}

                outputs = model(images)

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
                continue

    if total_samples == 0 or batches_processed_eval == 0:
        write_log(f"{dataset_name} Evaluation Summary: No samples successfully processed.")
        return None

    avg_loss = total_loss / batches_processed_eval
    avg_indiv_losses = {k: v / batches_processed_eval for k, v in eval_indiv_losses.items()}

    accuracy = accuracy_score(all_labels_grades, all_preds_grades) * 100
    unique_labels_in_split = sorted(np.unique(all_labels_grades + all_preds_grades))
    target_names_eval = [cn for i, cn in enumerate(class_names) if i in unique_labels_in_split] if class_names else None

    report = classification_report(
        all_labels_grades,
        all_preds_grades,
        labels=unique_labels_in_split,
        target_names=target_names_eval,
        digits=4,
        zero_division=0
    )
    cm = confusion_matrix(all_labels_grades, all_preds_grades, labels=range(len(class_names)))

    write_log(f'{dataset_name} Evaluation Summary:')
    write_log(f'  Average Total Loss: {avg_loss:.4f}')
    write_log(f'  Overall Grade Accuracy: {accuracy:.2f}% ({accuracy_score(all_labels_grades, all_preds_grades, normalize=True):.4f})')
    write_log(f'  Avg Individual Losses: { {k: f"{v:.4f}" for k,v in avg_indiv_losses.items()} }')

    report_path = os.path.join(results_dir, f"{dataset_name.lower().replace(' ', '_')}_classification_report.txt")
    try:
        os.makedirs(os.path.dirname(report_path), exist_ok=True) # Ensure dir exists
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"--- {dataset_name} Set Classification Report ---\n\n")
            f.write(f"Overall Accuracy: {accuracy:.4f}%\n\n")
            f.write(f"Labels present in this evaluation: {unique_labels_in_split}\n")
            f.write(f"Corresponding target names: {target_names_eval}\n\n")
            f.write(report)
        write_log(f"Classification report saved to: {report_path}")
    except Exception as e:
        write_log(f"ERROR saving classification report to {report_path}: {e}")

    cm_path = os.path.join(results_dir, f"{dataset_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_path, title=f'{dataset_name} Set Confusion Matrix (Acc: {accuracy:.2f}%)')

    return {"accuracy": accuracy, "report": report, "cm": cm, "loss": avg_loss}


# --- Dataset Class Definition --- (Keep TyphoonDataset as it is)
class TyphoonDataset(Dataset):
    def __init__(self, dataframe, transform, grade_map):
        if dataframe is None or dataframe.empty:
             log_msg = "CRITICAL: Input dataframe to TyphoonDataset is None or empty."
             write_log(log_msg)
             raise ValueError(log_msg)

        self.dataframe = dataframe.copy()
        self.transform = transform
        self.grade_map = grade_map

        initial_len_total = len(self.dataframe)
        num_filtered_grade = 0; num_filtered_cinum = 0; num_filtered_path = 0

        # Filtering steps... (identical to original)
        unknown_grades_mask = ~self.dataframe['Grade'].isin(self.grade_map.keys())
        if unknown_grades_mask.any():
            unknown_grades = self.dataframe.loc[unknown_grades_mask, 'Grade'].unique()
            write_log(f"INFO: Filtering out {unknown_grades_mask.sum()} samples with unknown grades: {list(unknown_grades)}")
            self.dataframe = self.dataframe[~unknown_grades_mask]
            num_filtered_grade = unknown_grades_mask.sum()

        nan_cinumber_mask = self.dataframe['CInumber'].isna()
        if nan_cinumber_mask.any():
             num_filtered_cinum = nan_cinumber_mask.sum()
             write_log(f"INFO: Filtering out {num_filtered_cinum} samples with missing CInumber.")
             self.dataframe = self.dataframe[~nan_cinumber_mask]

        if not self.dataframe.empty:
            exists_mask = self.dataframe['file_name'].apply(os.path.isfile)
            if not exists_mask.all():
                num_filtered_path = (~exists_mask).sum()
                missing_files = self.dataframe.loc[~exists_mask, 'file_name'].tolist()
                write_log(f"WARNING: Filtering out {num_filtered_path} samples with non-existent image paths during dataset init.")
                self.dataframe = self.dataframe[exists_mask]

        self.dataframe.reset_index(drop=True, inplace=True)
        final_len = len(self.dataframe)
        total_filtered = initial_len_total - final_len
        if total_filtered > 0:
             write_log(f"Dataset Init Summary: Started with {initial_len_total}, "
                       f"Filtered: {num_filtered_grade} (Grade), {num_filtered_cinum} (CInum), {num_filtered_path} (Path). "
                       f"Final size: {final_len}")
        elif initial_len_total > 0 :
             write_log(f"Dataset Init: All {initial_len_total} samples passed initial filtering.")

        if final_len == 0 and initial_len_total > 0 :
             log_msg = "CRITICAL: Dataset is empty after filtering steps. Cannot proceed."
             write_log(log_msg)
             raise ValueError(log_msg)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if not (0 <= idx < len(self.dataframe)):
             write_log(f"ERROR: Attempted to access invalid index {idx} in dataset of size {len(self.dataframe)}")
             return None
        row = self.dataframe.iloc[idx]
        img_path = row['file_name']

        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
             write_log(f"ERROR: FileNotFoundError (post-init check): {img_path} (index {idx}). Skipping.")
             return None
        except IsADirectoryError:
            write_log(f"ERROR: Path is a directory: {img_path} (index {idx}). Skipping.")
            return None
        except UnidentifiedImageError:
             write_log(f"ERROR: Cannot identify image file (corrupted?): {img_path} (index {idx}). Skipping.")
             return None
        except Exception as e:
             write_log(f"ERROR opening/reading image {img_path} (index {idx}): {e}. Skipping.")
             return None

        try:
            image = self.transform(image)
        except Exception as e:
            write_log(f"ERROR applying transforms to image {img_path} (index {idx}): {e}. Skipping.")
            return None

        try:
            grade_label = row['Grade']
            if grade_label not in self.grade_map:
                write_log(f"ERROR: Grade '{grade_label}' from file {img_path} (idx {idx}) not in GRADE_MAP post-init. Skipping.")
                return None
            grade = torch.tensor(self.grade_map[grade_label], dtype=torch.long)

            cinumber_val = row['CInumber']
            if cinumber_val is None or np.isnan(cinumber_val):
                 write_log(f"ERROR: Invalid CInumber '{cinumber_val}' from file {img_path} (idx {idx}) post-init. Skipping.")
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
                'cinumber': cinumber.unsqueeze(0), # Ensure [1] shape
            }
        }

# --- Model Class Definitions ---

# Modify AuxiliaryHead: Simplify for flat feature input from ViT
class CINumberHead(nn.Module):
    """Regression head for CInumber, taking flat features."""
    def __init__(self, in_features, num_outputs=1):
        super().__init__()
        # ViT output before head is typically [B, embed_dim], no pooling needed
        # self.pool = nn.AdaptiveAvgPool2d((1, 1)) # Removed
        # self.flatten = nn.Flatten() # Removed
        self.dropout = nn.Dropout(0.4) # Keep dropout
        self.fc = nn.Linear(in_features, num_outputs)

    def forward(self, x):
        # Input x is assumed to be [B, in_features]
        # x = self.pool(x) # Removed
        # x = self.flatten(x) # Removed
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Create ViTMultiTask model using timm
class ViTMultiTask(nn.Module):
    """ViT-Small based model for Grade classification and CInumber regression."""
    def __init__(self, num_grades, vit_model_name='vit_small_patch16_224'):
        super().__init__()
        # Load ViT Small model with pre-trained weights using timm
        # We want the features *before* the final classification head
        self.backbone = timm.create_model(vit_model_name, pretrained=True)

        # Get the feature dimension (embedding size) from the ViT model
        # For 'vit_small_patch16_224', this is typically 384
        feature_dim = self.backbone.head.in_features # Access the input feature dim of the original head

        # Remove the original classification head of the ViT model
        # We'll replace it with our custom heads
        self.backbone.head = nn.Identity()

        # Head for CINumber (regression) - takes ViT features directly
        self.cinumber_head = CINumberHead(feature_dim, 1) # Use the simplified head

        # Head for Grade classification
        # Concatenates ViT features and the CInumber prediction
        concatenated_feature_dim = feature_dim + 1 # ViT features + 1 (CInum prediction)
        self.grade_head = nn.Sequential(
            nn.LayerNorm(concatenated_feature_dim), # Use LayerNorm for transformer features? Or BatchNorm1d? Let's try LayerNorm first.
            # nn.BatchNorm1d(concatenated_feature_dim), # Alternative
            nn.Dropout(0.5),
            nn.Linear(concatenated_feature_dim, num_grades) # Final classification layer
        )

    def forward(self, x):
        # Backbone feature extraction - output is typically [B, feature_dim] (e.g., CLS token embedding)
        features = self.backbone(x)

        # CInumber Prediction (from ViT features)
        out_cinumber = self.cinumber_head(features) # Output: [B, 1]

        # Concatenate main features and auxiliary output
        concatenated_features = torch.cat([features, out_cinumber], dim=1) # Output: [B, feature_dim + 1]

        # Grade Prediction (from concatenated features)
        out_grade = self.grade_head(concatenated_features) # Output: [B, num_grades]

        outputs = {'grade': out_grade, 'cinumber': out_cinumber}
        return outputs

# --- Loss Function Definition --- (Keep calculate_combined_loss as it is)
criterion_clf_train = nn.CrossEntropyLoss(label_smoothing=0.1)
criterion_reg_train = nn.MSELoss()

def calculate_combined_loss(outputs, labels, weights):
    """Calculates the combined weighted loss for training."""
    loss_grade = criterion_clf_train(outputs['grade'], labels['grade'])
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
    multiprocessing.freeze_support()

    # --- Create Directories ---
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True) # Use updated results_dir
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True) # Use updated model path

    # --- Start Logging ---
    write_log("--- Starting Script Execution (ViT Small Version) ---")
    write_log(f"PyTorch Version: {torch.__version__}")
    write_log(f"Timm Version: {timm.__version__}") # Log timm version
    write_log(f"Using device: {DEVICE}")
    if DEVICE.type == 'cuda':
        write_log(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        write_log(f"CUDA Available: {torch.cuda.is_available()}")

    # --- 1. Configuration & Setup ---
    # Hyperparameters (ViT might benefit from different LR/WD, keep existing for now, but note for tuning)
    LEARNING_RATE = 5e-5 # Often lower LR for ViT fine-tuning
    BATCH_SIZE = 32 # ViTs can be memory intensive, may need smaller batch size
    EPOCHS = 80
    WEIGHT_DECAY = 1e-5 # AdamW default is often okay, but sometimes higher (0.05) is used for ViT
    SCHEDULER_PATIENCE = 20
    SCHEDULER_FACTOR = 0.2
    EARLY_STOPPING_PATIENCE = 30
    W_GRADE = 2.0
    W_CINUMBER = 10.0 # Adjust weights based on task importance / loss scales if needed
    loss_weights = {'grade': W_GRADE, 'cinumber': W_CINUMBER}

    # Load dataframes (identical)
    df_train = load_csv_safe(TRAIN_CSV_PATH, "train")
    df_val = load_csv_safe(VAL_CSV_PATH, "validation")
    df_test = load_csv_safe(TEST_CSV_PATH, "test")

    if df_train is None or df_val is None or df_test is None:
        write_log("CRITICAL ERROR: Data CSV loading failed. Exiting.")
        raise SystemExit("Could not load necessary CSV data.")

    # --- *** Training Data Duplication *** --- (identical)
    write_log("\n--- Processing Training Data ---")
    original_train_size = len(df_train)
    df_train = pd.concat([df_train], ignore_index=True) # DUPLICATION ACTIVE
    new_train_size = len(df_train)
    write_log(f"DUPLICATED training data. Original size: {original_train_size}, New size: {new_train_size}")

    # --- Label Mapping --- (identical)
    all_grades_list = []
    if df_train is not None: all_grades_list.append(df_train['Grade'])
    if df_val is not None: all_grades_list.append(df_val['Grade'])
    if df_test is not None: all_grades_list.append(df_test['Grade'])
    if not all_grades_list:
        raise SystemExit("CRITICAL ERROR: No grade data found.")
    all_grades = sorted(pd.concat(all_grades_list).unique())
    GRADE_MAP = {grade: i for i, grade in enumerate(all_grades)}
    CLASS_NAMES = list(GRADE_MAP.keys())
    NUM_GRADES = len(GRADE_MAP)
    write_log("\n--- Label Mappings ---")
    write_log(f"GRADE_MAP: {GRADE_MAP}")
    write_log(f"CLASS_NAMES: {CLASS_NAMES}")
    write_log(f"Number of Grade classes (NUM_GRADES): {NUM_GRADES}")

    # --- 2. Dataset & DataLoader Instantiation ---
    write_log("\n--- Defining Data Transforms ---")
    # Use standard ImageNet means/stds which ViT models are typically trained with
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), # Ensure 224x224
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Lambda(random_rotate),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet norm
        # Could also use timm's recommended normalization:
        # transforms.Normalize(mean=timm.data.IMAGENET_DEFAULT_MEAN, std=timm.data.IMAGENET_DEFAULT_STD)
    ])
    write_log(f"Train Transforms: {train_transform}")

    val_test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # transforms.Normalize(mean=timm.data.IMAGENET_DEFAULT_MEAN, std=timm.data.IMAGENET_DEFAULT_STD)
    ])
    write_log(f"Validation/Test Transforms: {val_test_transform}")

    write_log("\n--- Creating Datasets ---")
    # Dataset creation and checks remain the same
    try:
        train_dataset = TyphoonDataset(df_train, transform=train_transform, grade_map=GRADE_MAP)
        val_dataset = TyphoonDataset(df_val, transform=val_test_transform, grade_map=GRADE_MAP)
        test_dataset = TyphoonDataset(df_test, transform=val_test_transform, grade_map=GRADE_MAP)

        write_log(f"Train dataset size (after filtering): {len(train_dataset)}")
        write_log(f"Validation dataset size (after filtering): {len(val_dataset)}")
        write_log(f"Test dataset size (after filtering): {len(test_dataset)}")

        if len(train_dataset) == 0:
             raise SystemExit("Empty training dataset.")
        if len(val_dataset) == 0:
             raise SystemExit("Empty validation dataset.")

        num_cpu_cores = os.cpu_count() or 1
        # ViT might be heavier, keep NUM_WORKERS reasonable
        NUM_WORKERS = min(4, num_cpu_cores) if DEVICE.type == 'cuda' else 0
        if DEVICE.type == 'cuda' and num_cpu_cores <= 2: NUM_WORKERS = min(2, num_cpu_cores)
        persistent_workers_flag = (NUM_WORKERS > 0) and (DEVICE.type == 'cuda')
        write_log(f"Using num_workers = {NUM_WORKERS} (persistent_workers={persistent_workers_flag}).")

        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True,
            persistent_workers=persistent_workers_flag,
            collate_fn=collate_fn_skip_none, drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
            persistent_workers=persistent_workers_flag,
            collate_fn=collate_fn_skip_none
        )
        # Only create test_loader if test_dataset is valid
        test_loader = None
        if len(test_dataset) > 0:
            test_loader = DataLoader(
                test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=True,
                persistent_workers=persistent_workers_flag,
                collate_fn=collate_fn_skip_none
            )
            write_log(f"Test dataset size (after filtering): {len(test_dataset)}")
        else:
             write_log("WARNING: Test dataset is empty after filtering. Test evaluation will be skipped.")


        write_log("\n--- DataLoaders Created Successfully ---")

    except ValueError as e:
        write_log(f"CRITICAL ERROR during Dataset creation: {e}")
        raise SystemExit(f"Dataset creation failed: {e}")
    except Exception as e:
        write_log(f"CRITICAL UNEXPECTED ERROR during Dataset/DataLoader creation: {e}")
        raise SystemExit(f"Unexpected error during data setup: {e}")


    # --- 3. Model Definition (Instantiation) ---
    write_log("\n--- Initializing Model ---")
    # Instantiate the ViT model
    model = ViTMultiTask(NUM_GRADES, vit_model_name='vit_small_patch16_224').to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    write_log(f"Model: ViTMultiTask (ViT Small Base: 'vit_small_patch16_224')") # Update model name
    write_log(f"Tasks: Grade (Classifier, {NUM_GRADES} classes), CINumber (Regressor)")
    write_log(f"Model Parameters: Total={total_params:,}, Trainable={trainable_params:,}")


    # --- 4. Optimizer, Scheduler, Loss (Instantiation/Setup) ---
    # AdamW is generally good for ViTs. LR and WD might need tuning.
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, verbose=False, min_lr=1e-7)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7) # Cosine annealing often works well too

    write_log("\n--- Training Setup ---")
    write_log(f"Optimizer: AdamW (LR={LEARNING_RATE}, WeightDecay={WEIGHT_DECAY})")
    write_log(f"Scheduler: ReduceLROnPlateau (Mode=max, Factor={SCHEDULER_FACTOR}, Patience={SCHEDULER_PATIENCE}, MinLR=1e-7)")
    write_log(f"Loss Weights: {loss_weights}")
    write_log(f"Label Smoothing (Train): 0.1")
    write_log(f"Epochs: {EPOCHS}")
    write_log(f"Batch Size: {BATCH_SIZE}") # Log potentially adjusted batch size
    write_log(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")


    # --- 5. Training & Validation Loop --- (Identical structure to original)
    best_val_acc = 0.0
    epochs_no_improve = 0
    history = []
    training_start_time = time.time()

    table_headers = OrderedDict([
        ("Epoch", ""), ("Train Loss", ""), ("Tr Grade Acc (%)", ""),
        ("Val Loss", ""), ("Val Grade Acc (%)", ""), ("LR", ""), ("Time (s)", "")
    ])

    write_log(f"\n--- Starting Training for {EPOCHS} Epochs ---")

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()

        # --- Training Phase ---
        model.train()
        running_loss = 0.0; train_correct_grades = 0; train_total_samples = 0
        epoch_train_indiv_losses = {k: 0.0 for k in loss_weights.keys()}; epoch_train_indiv_losses['total'] = 0.0
        batches_processed_train = 0; skipped_batches_train = 0

        for i, batch in enumerate(train_loader):
            if batch is None: skipped_batches_train += 1; continue
            try:
                images = batch['image'].to(DEVICE, non_blocking=True)
                labels = {k: v.to(DEVICE, non_blocking=True) for k, v in batch['labels'].items()}
                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                loss, indiv_losses = calculate_combined_loss(outputs, labels, loss_weights)

                if torch.isnan(loss) or torch.isinf(loss):
                    write_log(f"WARNING: Invalid loss (NaN/Inf) in train epoch {epoch+1}, batch {i}. Skipping update.")
                    skipped_batches_train += 1; continue

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Optional gradient clipping
                optimizer.step()

                running_loss += loss.item()
                for k, v in indiv_losses.items():
                     if not np.isnan(v) and not np.isinf(v): epoch_train_indiv_losses[k] += v
                     else: write_log(f"Warning: Invalid value ({v}) for indiv loss '{k}' train batch {i}, epoch {epoch+1}")

                _, predicted_grades = torch.max(outputs['grade'], 1)
                train_total_samples += labels['grade'].size(0)
                train_correct_grades += (predicted_grades == labels['grade']).sum().item()
                batches_processed_train += 1
            except Exception as e:
                write_log(f"ERROR during training batch {i}, epoch {epoch+1}: {e}")
                # import traceback; write_log(traceback.format_exc()) # Uncomment for detailed trace
                optimizer.zero_grad(); skipped_batches_train += 1; continue

        if skipped_batches_train > 0: write_log(f"INFO: Skipped {skipped_batches_train} train batches epoch {epoch+1}.")
        if batches_processed_train == 0:
             write_log(f"CRITICAL ERROR: Epoch {epoch+1} had 0 successful train batches. Stopping."); break

        epoch_train_loss = running_loss / batches_processed_train
        epoch_train_acc = 100 * train_correct_grades / train_total_samples if train_total_samples > 0 else 0
        avg_train_indiv_losses = {k: (v / batches_processed_train) if batches_processed_train > 0 else 0.0
                                   for k, v in epoch_train_indiv_losses.items()}

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0; val_correct_grades = 0; val_total_samples = 0
        epoch_val_indiv_losses = {k: 0.0 for k in loss_weights.keys()}; epoch_val_indiv_losses['total'] = 0.0
        batches_processed_val = 0; skipped_batches_val = 0; can_validate = True

        if not val_loader or not hasattr(val_loader, 'dataset') or len(val_loader.dataset) == 0:
            write_log(f"Epoch [{epoch+1}/{EPOCHS}] - WARNING: Skipping validation: Loader invalid/empty.")
            can_validate = False
        else:
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    if batch is None: skipped_batches_val += 1; continue
                    try:
                        images = batch['image'].to(DEVICE, non_blocking=True)
                        labels = {k: v.to(DEVICE, non_blocking=True) for k, v in batch['labels'].items()}
                        outputs = model(images)
                        loss, indiv_losses = calculate_combined_loss(outputs, labels, loss_weights)

                        if torch.isnan(loss) or torch.isinf(loss):
                            write_log(f"WARNING: Invalid loss (NaN/Inf) in val epoch {epoch+1}, batch {i}. Skipping batch.")
                            skipped_batches_val += 1; continue

                        val_loss += loss.item()
                        for k, v in indiv_losses.items():
                           if not np.isnan(v) and not np.isinf(v): epoch_val_indiv_losses[k] += v
                           else: write_log(f"Warning: Invalid value ({v}) for indiv loss '{k}' val batch {i}, epoch {epoch+1}")

                        _, predicted_grades = torch.max(outputs['grade'], 1)
                        val_total_samples += labels['grade'].size(0)
                        val_correct_grades += (predicted_grades == labels['grade']).sum().item()
                        batches_processed_val += 1
                    except Exception as e:
                        write_log(f"ERROR during validation batch {i}, epoch {epoch+1}: {e}")
                        skipped_batches_val += 1; continue

            if skipped_batches_val > 0: write_log(f"INFO: Skipped {skipped_batches_val} val batches epoch {epoch+1}.")
            if batches_processed_val == 0 and val_total_samples == 0:
                write_log(f"WARNING: Epoch {epoch+1} had 0 successful val batches. Cannot validate.")
                can_validate = False
            else:
                epoch_val_loss = val_loss / batches_processed_val if batches_processed_val > 0 else float('nan')
                epoch_val_acc = 100 * val_correct_grades / val_total_samples if val_total_samples > 0 else 0.0
                avg_val_indiv_losses = {k: (v / batches_processed_val) if batches_processed_val > 0 else 0.0
                                         for k, v in epoch_val_indiv_losses.items()}
                if np.isnan(epoch_val_loss):
                     write_log(f"WARNING: Avg val loss is NaN Epoch {epoch+1}. Treating validation as failed.")
                     can_validate = False

        if not can_validate:
            epoch_val_loss = float('nan'); epoch_val_acc = float('nan')
            avg_val_indiv_losses = {k: float('nan') for k in loss_weights.keys()}; avg_val_indiv_losses['total'] = float('nan')

        # --- Epoch Summary & Table Update ---
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        epoch_data = OrderedDict([
            ("Epoch", f"{epoch+1}/{EPOCHS}"), ("Train Loss", f"{epoch_train_loss:.4f}"),
            ("Tr Grade Acc (%)", f"{epoch_train_acc:.2f}"),
            ("Val Loss", f"{epoch_val_loss:.4f}" if can_validate else "N/A"),
            ("Val Grade Acc (%)", f"{epoch_val_acc:.2f}" if can_validate else "N/A"),
            ("LR", f"{current_lr:.2e}"), ("Time (s)", f"{epoch_time:.1f}")
        ])
        history.append(epoch_data)
        summary_log = (f"Epoch {epoch+1}/{EPOCHS} | Tr Loss: {epoch_train_loss:.4f}, Tr Acc: {epoch_train_acc:.2f}% | "
                       f"Val Loss: {epoch_data['Val Loss']}, Val Acc: {epoch_data['Val Grade Acc (%)']}% | "
                       f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
        write_log(summary_log)

        # --- LR Scheduling and Early Stopping ---
        if can_validate:
            scheduler.step(epoch_val_acc)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < current_lr: write_log(f"  Learning rate reduced to {new_lr:.2e}.")

            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                try:
                    torch.save(model.state_dict(), MODEL_SAVE_PATH) # Save to updated path
                    write_log(f'  ---> New best model saved to {MODEL_SAVE_PATH} (Val Grade Acc: {best_val_acc:.2f}%) <---')
                    epochs_no_improve = 0
                except Exception as e: write_log(f"ERROR saving model: {e}")
            else:
                epochs_no_improve += 1
                write_log(f'  Val Acc ({epoch_val_acc:.2f}%) did not improve vs best ({best_val_acc:.2f}%). Plateau: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}.')
                if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                    write_log(f"\n--- Early stopping triggered epoch {epoch + 1}. Best Val Acc: {best_val_acc:.2f}% ---")
                    break
        else:
            write_log("  Skipping LR schedule step/early stopping (validation failed/skipped).")


    # --- End of Training ---
    training_duration = time.time() - training_start_time
    write_log("\n--- Training Finished ---")
    write_log(f"Total training time: {training_duration:.2f} seconds ({training_duration/3600:.2f} hours)")
    if history:
        write_log("\n--- Training History Summary ---")
        try:
            history_table = tabulate([list(h.values()) for h in history], headers=table_headers.keys(), tablefmt="plain")
            write_log(history_table)
        except Exception as e: write_log(f"Error generating history table: {e}")
    write_log(f"Best Validation Grade Accuracy: {best_val_acc:.2f}%")
    write_log(f"Model saved at: {MODEL_SAVE_PATH}")


    # --- 6. Final Evaluation & Metrics ---
    write_log(f"\n--- Starting Final Evaluation ---")
    write_log(f"Loading best model state_dict from: {MODEL_SAVE_PATH}")

    if os.path.exists(MODEL_SAVE_PATH):
        # Instantiate a fresh ViT model for testing
        model_to_test = ViTMultiTask(NUM_GRADES, vit_model_name='vit_small_patch16_224')
        model_to_test.to(DEVICE)
        try:
            state_dict = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
            model_to_test.load_state_dict(state_dict, strict=True)
            write_log("Best model state_dict loaded successfully.")

            # Evaluate on Validation Set
            val_results = evaluate_and_report(
                model=model_to_test, data_loader=val_loader, device=DEVICE,
                loss_weights=loss_weights, class_names=CLASS_NAMES,
                dataset_name="Validation (Best Model)"
            )

            # Evaluate on Test Set (only if test_loader is valid)
            if test_loader:
                test_results = evaluate_and_report(
                    model=model_to_test, data_loader=test_loader, device=DEVICE,
                    loss_weights=loss_weights, class_names=CLASS_NAMES,
                    dataset_name="Test"
                )
            else:
                 write_log("Skipping Test Set evaluation as the test loader was not created (dataset likely empty).")


        except FileNotFoundError: # Safety
             write_log(f"ERROR: Model file disappeared: {MODEL_SAVE_PATH}")
        except RuntimeError as e:
             write_log(f"ERROR loading state_dict: {e}. Architecture mismatch or corrupt file?")
        except Exception as e:
             write_log(f"An unexpected error occurred during final evaluation: {e}")
    else:
        write_log(f"ERROR: Best model file not found at {MODEL_SAVE_PATH}. Cannot perform final evaluation.")

    write_log("\n--- Script Finished ---")

# --- END OF FILE ViT_Small_GradeCI.py ---