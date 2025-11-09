"""
Training Script - Train GNN model cho vulnerability detection
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json
from pathlib import Path

from data_loader import VulnerabilityDataset, create_dataloaders
from model import create_model


class Trainer:
    """Training manager"""
    
    LABEL_NAMES = ['Safe', 'Buffer_Overflow', 'Command_Injection', 'Path_Traversal', 'SQL_Injection']
    
    def __init__(self, 
                 model,
                 train_loader,
                 val_loader,
                 test_loader,
                 device='cuda',
                 lr=1e-3,
                 num_epochs=50):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = num_epochs
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.5, 
            patience=5
        )
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        self.best_model_state = None
    
    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(batch)
            loss = self.criterion(logits, batch.y.squeeze())
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = batch.y.squeeze().cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def validate(self, data_loader, desc='Validation'):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=desc):
                batch = batch.to(self.device)
                
                # Forward pass
                logits = self.model(batch)
                loss = self.criterion(logits, batch.y.squeeze())
                
                # Track metrics
                total_loss += loss.item()
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy()
                labels = batch.y.squeeze().cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
                all_probs.extend(probs.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'labels': all_labels,
            'probs': all_probs
        }
        
        return metrics
    
    def train(self):
        """Full training loop"""
        
        print(f"\n{'='*70}")
        print(f"Starting Training - {self.num_epochs} epochs")
        print(f"{'='*70}\n")
        
        for epoch in range(self.num_epochs):
            print(f"\n{'─'*70}")
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"{'─'*70}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_metrics = self.validate(self.val_loader, desc='Validation')
            
            # Track
            self.train_losses.append(train_loss)
            self.val_losses.append(val_metrics['loss'])
            self.val_accuracies.append(val_metrics['accuracy'])
            
            # Learning rate scheduling
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_metrics['accuracy'])
            new_lr = self.optimizer.param_groups[0]['lr']
            
            if new_lr != old_lr:
                print(f"  📉 Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
            
            # Print epoch results
            print(f"\nResults:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f} | Val Acc:   {val_metrics['accuracy']:.4f}")
            print(f"  Val F1:     {val_metrics['f1']:.4f}")
            
            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.best_model_state = self.model.state_dict().copy()
                print(f"  ✓ New best model! (Val Acc: {self.best_val_acc:.4f})")
        
        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"Best Validation Accuracy: {self.best_val_acc:.4f}")
        print(f"{'='*70}\n")
    
    def test(self):
        """Test on test set"""
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Test
        test_metrics = self.validate(self.test_loader, desc='Testing')
        
        # Print results
        print(f"\n{'='*70}")
        print(f"Test Results:")
        print(f"{'='*70}")
        print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"Precision: {test_metrics['precision']:.4f}")
        print(f"Recall:    {test_metrics['recall']:.4f}")
        print(f"F1 Score:  {test_metrics['f1']:.4f}")
        
        # Confusion matrix
        print(f"\nConfusion Matrix:")
        print(f"{'─'*70}")
        cm = test_metrics['confusion_matrix']
        
        # Header
        print(f"{'':15s}", end='')
        for label in self.LABEL_NAMES:
            print(f"{label[:10]:>12s}", end='')
        print()
        
        # Rows
        for i, label in enumerate(self.LABEL_NAMES):
            print(f"{label[:15]:15s}", end='')
            for j in range(len(self.LABEL_NAMES)):
                print(f"{cm[i][j]:12d}", end='')
            print()
        
        print(f"{'='*70}\n")
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            test_metrics['labels'], 
            test_metrics['predictions'],
            labels=list(range(5)),
            zero_division=0
        )
        
        print(f"Per-Class Metrics:")
        print(f"{'─'*70}")
        print(f"{'Class':<20s} {'Precision':>12s} {'Recall':>12s} {'F1':>12s} {'Support':>12s}")
        print(f"{'─'*70}")
        for i, label in enumerate(self.LABEL_NAMES):
            print(f"{label:<20s} {precision[i]:>12.4f} {recall[i]:>12.4f} {f1[i]:>12.4f} {support[i]:>12d}")
        print(f"{'='*70}\n")
        
        return test_metrics
    
    def save_model(self, save_path):
        """Save model"""
        torch.save({
            'model_state_dict': self.best_model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }, save_path)
        print(f"Model saved to {save_path}")


def main():
    """Main training function"""
    
    # Config - Auto detect environment
    import os
    if os.path.exists('/content'):
        # Running on Google Colab
        ROOT_DIR = '/content/DoAn_2'
    else:
        # Running locally
        ROOT_DIR = r"c:\Users\abcdx\OneDrive\Máy tính\renew"
    
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Environment: {'Google Colab' if '/content' in ROOT_DIR else 'Local'}")
    print(f"Root directory: {ROOT_DIR}")
    print(f"Device: {DEVICE}")
    
    # Create dataloaders
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(
        ROOT_DIR,
        batch_size=BATCH_SIZE,
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    # Create model
    print("\nCreating model...")
    model = create_model('simplified', num_classes=5, device=DEVICE)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=DEVICE,
        lr=LEARNING_RATE,
        num_epochs=NUM_EPOCHS
    )
    
    # Train
    trainer.train()
    
    # Test
    test_metrics = trainer.test()
    
    # Save model
    save_path = Path(ROOT_DIR) / 'saved_models' / 'best_model.pth'
    save_path.parent.mkdir(exist_ok=True)
    trainer.save_model(save_path)
    
    # Save results
    results = {
        'test_accuracy': float(test_metrics['accuracy']),
        'test_f1': float(test_metrics['f1']),
        'confusion_matrix': test_metrics['confusion_matrix'].tolist()
    }
    
    results_path = Path(ROOT_DIR) / 'results' / 'training_results.json'
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
