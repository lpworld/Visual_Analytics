#### 1. Load modules
import shap
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import time
import warnings
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
from facenet_pytorch import InceptionResnetV1
from PIL import Image

# New imports for evaluation
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, auc


# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
except ImportError:
    pass

# Check for and set mixed precision policy
mixed_precision_enabled = torch.cuda.is_available()
if mixed_precision_enabled: print("Mixed precision training is enabled.")

#### 2. PyTorch Model Definition (Custom Attention)
class AttentionVGGFace(nn.Module):
    def __init__(self):
        super(AttentionVGGFace, self).__init__()
        resnet = InceptionResnetV1(pretrained='vggface2')
        self.feature_extractor_layers = nn.Sequential(
            resnet.conv2d_1a, resnet.conv2d_2a, resnet.conv2d_2b, resnet.maxpool_3a,
            resnet.conv2d_3b, resnet.conv2d_4a, resnet.conv2d_4b, resnet.repeat_1,
            resnet.mixed_6a, resnet.repeat_2, resnet.mixed_7a, resnet.repeat_3, resnet.block8
        )
        for param in self.feature_extractor_layers.parameters():
            param.requires_grad = False
        pt_depth = 1792
        self.attention = nn.Conv2d(in_channels=pt_depth, out_channels=1, kernel_size=1)
        self.fan_out = nn.Conv2d(in_channels=1, out_channels=pt_depth, kernel_size=1, bias=False)
        self.fan_out.weight.data.fill_(1.0)
        self.fan_out.weight.requires_grad = False
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(pt_depth * 5 * 5, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2)
        )

    def forward(self, x):
        features = self.feature_extractor_layers(x)
        attn_map = torch.sigmoid(self.attention(features))
        attn_map_fanned = self.fan_out(attn_map)
        attended_features = features * attn_map_fanned
        flat = self.flatten(attended_features)
        output = self.classifier(flat)
        return output

#### 3. PyTorch Data Handling
class FaceDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.transform = transform
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        if self.transform: image = self.transform(image)
        return image, self.labels[idx]

#### 4. PyTorch Training & Prediction Utilities
def run_training(model, train_loader, val_loader, device):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(list(model.attention.parameters()) + list(model.classifier.parameters()), lr=1e-5)
    scaler = GradScaler(enabled=mixed_precision_enabled)
    best_val_loss, epochs_no_improve, patience = float('inf'), 0, 10
    for epoch in range(50):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast(enabled=mixed_precision_enabled):
                outputs = model(images); loss = criterion(outputs, labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast(enabled=mixed_precision_enabled):
                    outputs = model(images); val_loss += criterion(outputs, labels).item()
        val_loss /= len(val_loader)
        if (epoch + 1) % 10 == 0: print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss, epochs_no_improve = val_loss, 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}."); break
    return model

def get_model_accuracy(model, data_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(torch.sigmoid(outputs), dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(torch.argmax(labels, dim=1).numpy())
    return accuracy_score(all_labels, all_preds)

#### 5. Evaluation Metrics Implementation
def evaluate_feature_importance(trained_model, importance_weights, test_data, test_labels, full_data_parts, device):
    print(f"\n{'='*20} RUNNING EVALUATION METRICS {'='*20}")
    
    feature_keys = list(importance_weights.keys())
    feature_keys_map = {key: key.upper() + '_PARSE' if key not in ['eyes'] else key.upper() for key in feature_keys}
    if 'EYEBROWS_PARSE' not in full_data_parts: 
        feature_keys_map['eyebrows'] = 'EYEBROWS'
    
    test_transforms = transforms.Compose([transforms.ToTensor()])
    test_loader = DataLoader(FaceDataset(test_data, test_labels, transform=test_transforms), batch_size=32)

    print("\n--- (1) Model Parameter Randomization Check ---")
    random_model = deepcopy(trained_model)
    for layer in random_model.classifier.children():
        if hasattr(layer, 'reset_parameters'): layer.reset_parameters()
    
    background_batch = next(iter(test_loader))[0][:50].to(device)
    test_batch = next(iter(test_loader))[0][:50].to(device)
    
    explainer = shap.GradientExplainer(random_model, background_batch)
    shap_values = explainer.shap_values(test_batch)
    sv_positive = shap_values[:, :, :, :, 1]
    shap_transposed = np.transpose(sv_positive, (0, 2, 3, 1))
    
    part_masks = {key: (np.sum(full_data_parts[feature_keys_map[key]], axis=0) > 0).astype(np.float32) for key in feature_keys}
    random_importance = {key: np.mean(np.sum(np.abs(shap_transposed * part_masks[key]), axis=(1, 2, 3))) for key in feature_keys}
    
    original_scores = np.array([importance_weights[key] for key in feature_keys])
    random_scores = np.array([random_importance[key] for key in feature_keys])
    correlation, _ = spearmanr(original_scores, random_scores)
    print(f"Spearman Correlation between original and randomized explanations: {correlation:.4f}")
    print("Interpretation: A LOW correlation is GOOD.")

    print("\n--- (3 & 4) Single and Incremental Deletion Checks ---")
    original_accuracy = get_model_accuracy(trained_model, test_loader, device)
    print(f"Original model accuracy on test set: {original_accuracy:.4f}")

    sorted_features = sorted(importance_weights.items(), key=lambda item: item[1], reverse=True)
    feature_order = [item[0] for item in sorted_features]
    
    single_deletion_impact, incremental_accuracies = {}, [original_accuracy]
    deleted_mask = np.zeros_like(test_data[0], dtype=np.float32)

    for i, feature_to_delete in enumerate(feature_order):
        single_del_mask = part_masks[feature_to_delete]
        single_del_test_data = test_data * (1 - single_del_mask)
        single_del_loader = DataLoader(FaceDataset(single_del_test_data.astype(np.uint8), test_labels, transform=test_transforms), batch_size=32)
        acc_after_single_del = get_model_accuracy(trained_model, single_del_loader, device)
        single_deletion_impact[feature_to_delete] = original_accuracy - acc_after_single_del

        deleted_mask += part_masks[feature_to_delete]
        incremental_test_data = test_data * (1 - np.clip(deleted_mask, 0, 1))
        incremental_loader = DataLoader(FaceDataset(incremental_test_data.astype(np.uint8), test_labels, transform=test_transforms), batch_size=32)
        acc_after_incremental_del = get_model_accuracy(trained_model, incremental_loader, device)
        incremental_accuracies.append(acc_after_incremental_del)

    impact_scores = [single_deletion_impact[key] for key in feature_keys]
    correlation, _ = spearmanr(original_scores, impact_scores)
    print(f"\n(3) Single Deletion: Spearman Correlation between SHAP scores and impact: {correlation:.4f}")
    print("Interpretation: A HIGH POSITIVE correlation is GOOD.")

    auc_score = auc(x=np.arange(len(incremental_accuracies)), y=incremental_accuracies)
    print(f"\n(4) Incremental Deletion: Accuracies after removing features: {[f'{acc:.3f}' for acc in incremental_accuracies]}")
    print(f"Area Under the Deletion Curve (AUC): {auc_score:.4f}")
    print("Interpretation: A LOWER AUC is BETTER.")

#### 6. Main Script Execution
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"; print(f"Using device: {device}")
    print("Loading data..."); INPUT = "allfeatures_parse_224.pickle"
    with open(INPUT, 'rb') as f: data = pickle.load(f)
    feature_keys_map = {'mouth': 'MOUTH_PARSE', 'eyes': 'EYES', 'nose': 'NOSE_PARSE', 'eyebrows': 'EYEBROWS_PARSE', 'outline': 'OUTLINE_PARSE'}
    feature_keys = list(feature_keys_map.keys())
    data_name = pd.DataFrame(data['filename'], columns=["img_name"])
    
    print("Preparing labels..."); label = "trust"
    df_mean = pd.read_excel("label.xlsx"); df_mean.drop(columns=["Unnamed:0"], axis=1, inplace=True, errors='ignore')
    RA1 = pd.read_excel("label1.xlsx", sheet_name='RA1'); df_mean["img_name"] = RA1["img_name"]
    df = df_mean.copy(); df['img_name'] = pd.Series(df['img_name'].str.replace('%', '_'))
    results = pd.merge(data_name, df, on=['img_name'])

    def get_balanced_data_indices(all_results, label_col):
        y_scores, dict_filter = all_results[label_col].values, 3
        label_0_indices, label_1_indices = np.where(y_scores < dict_filter)[0], np.where(y_scores >= dict_filter)[0]
        num_to_choose = min(len(label_0_indices), len(label_1_indices))
        if num_to_choose == 0: raise ValueError("No samples found for one or both classes.")
        sorted_indices = np.argsort(y_scores)
        chosen_label_0, chosen_label_1 = sorted_indices[:num_to_choose], sorted_indices[-num_to_choose:]
        final_indices = np.concatenate([chosen_label_0, chosen_label_1]); final_labels = np.concatenate([np.zeros(num_to_choose), np.ones(num_to_choose)])
        shuffle_perm = np.random.permutation(len(final_indices)); return final_indices[shuffle_perm], final_labels[shuffle_perm]

    NUM_META_ITERATIONS = 8
    feature_weights = {key: 1.0 / len(feature_keys) for key in feature_keys}
    history_of_weights = []

    for meta_iter in range(NUM_META_ITERATIONS):
        print(f"\n{'='*20} META-ITERATION: {meta_iter + 1}/{NUM_META_ITERATIONS} {'='*20}")
        history_of_weights.append(feature_weights.copy())
        print(f"Current Feature Weights: {feature_weights}")

        print("Creating weighted input data...")
        part_shape = data[feature_keys_map[feature_keys[0]]].shape
        part_array = np.zeros(part_shape, dtype=np.float32)
        for key in feature_keys: part_array += data[feature_keys_map[key]] * feature_weights[key]
        part_array = np.clip(part_array, 0, 255).astype(np.uint8)

        index_data = [data_name['img_name'].tolist().index(name) for name in results['img_name']]
        x_all = part_array[index_data]
        data_indices, data_labels = get_balanced_data_indices(results, label)
        x_images, y_labels = x_all[data_indices], pd.get_dummies(data_labels).values
        
        x_main, x_test, y_main, y_test = train_test_split(x_images, y_labels, stratify=y_labels[:,0], test_size=0.15, random_state=567)
        train_data, val_data, train_label, val_label = train_test_split(x_main, y_main, stratify=y_main[:,0], test_size=0.18, random_state=567)
        
        train_transforms = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.2, contrast=0.2)])
        test_transforms = transforms.Compose([transforms.ToTensor()])
        train_dataset = FaceDataset(train_data, train_label, transform=train_transforms)
        val_dataset = FaceDataset(val_data, val_label, transform=test_transforms)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Create the test_loader so it's available for the SHAP step below
        test_loader = DataLoader(FaceDataset(x_test, y_test, transform=test_transforms), batch_size=32)
        
        model = AttentionVGGFace().to(device)
        trained_model = run_training(model, train_loader, val_loader, device)

        print("Analyzing with SHAP and calibrating to derive new weights...")
        trained_model.eval()
        background_batch = next(iter(train_loader))[0][:200].to(device)
        test_batch = next(iter(test_loader))[0][:200].to(device)
        explainer = shap.GradientExplainer(trained_model, background_batch)
        shap_values = explainer.shap_values(test_batch)
        sv_positive_class = shap_values[:, :, :, :, 1]
        shap_values_calibrated = np.transpose(sv_positive_class, (0, 2, 3, 1))
        
        part_masks_final = {key: (np.sum(data[feature_keys_map[key]], axis=0) > 0).astype(np.float32) for key in feature_keys}
        per_sample_importance = np.array([np.sum(np.abs(shap_values_calibrated * part_masks_final[key]), axis=(1, 2, 3)) for key in feature_keys]).T
        cov_matrix = np.cov(per_sample_importance, rowvar=False)
        cov_df = pd.DataFrame(cov_matrix, index=feature_keys, columns=feature_keys)
        local_importance = np.mean(per_sample_importance, axis=0)
        cov_abs_norm = cov_df.abs().div(cov_df.abs().sum(axis=1), axis=0)
        calibrated_importance = cov_abs_norm.values @ local_importance
        total_calibrated_importance = sum(calibrated_importance)
        if total_calibrated_importance > 0:
            feature_weights = {key: val / total_calibrated_importance for key, val in zip(feature_keys, calibrated_importance)}

    print(f"\n{'='*20} LOOP COMPLETE {'='*20}")
    final_weights_df = pd.DataFrame(history_of_weights)
    if not final_weights_df.empty:
        final_weights_df.loc[len(history_of_weights)] = feature_weights
        print("\nFinal Converged Feature Weights (Prior-Guided Custom Attention):")
        print(final_weights_df.iloc[-1])

    if feature_weights:
        evaluate_feature_importance(trained_model, feature_weights, x_test, y_test, data, device)

if __name__ == '__main__':
    main()
