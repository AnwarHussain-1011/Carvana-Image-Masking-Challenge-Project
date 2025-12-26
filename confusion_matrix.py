import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import torch

def generate_all_plots(csv_file, model, val_loader, device, output_dir):
    """Generates both the training metrics dashboard and the confusion matrix."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # --- 1. GENERATE TRAINING METRICS GRAPH ---
    try:
        df = pd.read_csv(csv_file)
        # Fix: Sort by epoch and keep only the latest entry if training was restarted
        df = df.sort_values('epoch').drop_duplicates('epoch', keep='last')

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Training Analysis', fontsize=20, fontweight='bold')

        # Training Loss Plot
        sns.lineplot(ax=axes[0, 0], data=df, x='epoch', y='train_loss', marker='o', color='#1f77b4')
        axes[0, 0].set_title('Training Loss (BCE + Dice)')

        # Dice Score Plot
        sns.lineplot(ax=axes[0, 1], data=df, x='epoch', y='val_dice', marker='o', color='#2ca02c')
        axes[0, 1].set_title('Validation Dice Score')

        # Learning Rate Plot (Log Scale)
        sns.lineplot(ax=axes[1, 0], data=df, x='epoch', y='learning_rate', marker='o', color='#d62728')
        axes[1, 0].set_yscale('log')
        axes[1, 0].set_title('Learning Rate Schedule')

        # Combined Loss vs Dice
        ax2 = axes[1, 1].twinx()
        sns.lineplot(ax=axes[1, 1], data=df, x='epoch', y='train_loss', color='#1f77b4', label='Loss')
        sns.lineplot(ax=ax2, data=df, x='epoch', y='val_dice', color='#2ca02c', label='Dice')
        axes[1, 1].set_title('Loss & Dice Convergence')
        axes[1, 1].legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_dir / 'training_dashboard.png', dpi=300)
        print(f"✅ Dashboard saved to {output_dir / 'training_dashboard.png'}")
    except Exception as e:
        print(f"❌ Metrics plotting failed: {e}")

    # --- 2. GENERATE CONFUSION MATRIX ---
    try:
        model.eval()
        all_preds, all_labels = [], []
        
        # We sample a few batches to avoid memory overflow
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i > 5: break # Sample 5 batches for the matrix
                images, targets = batch['image'].to(device), batch['mask'].to(device)
                outputs = model(images)
                preds = (torch.sigmoid(outputs) > 0.5).float() if model.n_classes == 1 else outputs.argmax(dim=1)
                
                all_preds.append(preds.cpu().numpy().flatten())
                all_labels.append(targets.cpu().numpy().flatten())

        cm = confusion_matrix(np.concatenate(all_labels), np.concatenate(all_preds))
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # Normalize

        plt.figure(figsize=(10, 8))
        class_names = ['Background', 'Object'] if model.n_classes <= 2 else [f'Class {i}' for i in range(model.n_classes)]
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Pixel-Wise Prediction Accuracy (Confusion Matrix)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=300)
        plt.close('all')
        print(f"✅ Confusion Matrix saved to {output_dir / 'confusion_matrix.png'}")
    except Exception as e:
        print(f"❌ Confusion Matrix failed: {e}")