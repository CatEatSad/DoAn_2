# 🚀 Google Colab Setup - Vulnerability Detection với GNN

## 1. Clone Repository

```python
!git clone https://github.com/CatEatSad/DoAn_2.git
%cd DoAn_2
```

## 2. Kiểm Tra Cấu Trúc Files

```python
# Kiểm tra xem có files JSON không
!ls -la output/Buffer_Overflow/ | head -10
!ls -la output_safe/Buffer_Overflow/ | head -10

# Đếm số files
!echo "Vulnerable files:"
!find output -name "*.json" -not -name "*prediction*" | wc -l

!echo "Safe files:"
!find output_safe -name "*.json" | wc -l
```

## 3. Cài Đặt Dependencies

```python
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q torch-geometric
!pip install -q transformers
!pip install -q scikit-learn pandas tqdm
```

## 4. Verify Installation

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

## 5. Test Data Loading

```python
import sys
sys.path.insert(0, '/content/DoAn_2/src')

from data_loader import VulnerabilityDataset

# Load dataset
dataset = VulnerabilityDataset('/content/DoAn_2', split='all')

print(f"Total samples: {len(dataset)}")

if len(dataset) > 0:
    # Get first sample
    sample = dataset[0]
    print(f"\nFirst sample:")
    print(f"  Nodes: {sample.num_nodes}")
    print(f"  Edges: {sample.edge_index.shape[1]}")
    print(f"  Label: {sample.y.item()}")
else:
    print("⚠️ No data found! Check file paths.")
```

## 6. Train Model

```python
%cd /content/DoAn_2/src
!python train.py
```

## 7. Monitor Training (Alternative - In Notebook)

```python
import sys
sys.path.insert(0, '/content/DoAn_2/src')

from data_loader import create_dataloaders
from model import create_model
from train import Trainer

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    '/content/DoAn_2',
    batch_size=8,
    train_ratio=0.7,
    val_ratio=0.15
)

# Create model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = create_model('simplified', num_classes=5, device=device)

# Train
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    device=device,
    lr=1e-3,
    num_epochs=10  # Start with fewer epochs for testing
)

trainer.train()
test_metrics = trainer.test()
```

## 8. Save Model to Google Drive (Optional)

```python
from google.colab import drive
drive.mount('/content/drive')

# Save model
import shutil
!mkdir -p /content/DoAn_2/saved_models
shutil.copy(
    '/content/DoAn_2/saved_models/best_model.pth',
    '/content/drive/MyDrive/vulnerability_model.pth'
)
print("Model saved to Google Drive!")
```

## 9. Visualize Results

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(trainer.train_losses, label='Train Loss')
plt.plot(trainer.val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training & Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(trainer.val_accuracies, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Validation Accuracy')

plt.tight_layout()
plt.show()

# Confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    test_metrics['confusion_matrix'],
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Safe', 'Buffer', 'Cmd', 'Path', 'SQL'],
    yticklabels=['Safe', 'Buffer', 'Cmd', 'Path', 'SQL']
)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
```

## 10. Predict New Files

```python
from predict import VulnerabilityPredictor

# Load model
predictor = VulnerabilityPredictor(
    '/content/DoAn_2/saved_models/best_model.pth',
    device='cuda'
)

# Analyze a file
example_file = '/content/DoAn_2/output/Command_Injection/Command_Injection_0001_vul.json'
analysis = predictor.analyze_code_patterns(example_file)

print("Patterns detected:")
for pattern in analysis['patterns_detected']:
    print(f"  - {pattern['type']}: {pattern['code'][:50]}")

print("\nRisk factors:")
for risk in analysis['risk_factors']:
    print(f"  - {risk['severity']}: {risk['type']}")
    print(f"    Reason: {risk['reason']}")
```

---

## 🔧 Troubleshooting

### Lỗi: No data found (0 samples)

```python
# Kiểm tra đường dẫn
import os
print("Current dir:", os.getcwd())
print("Files in output/:")
!ls output/

# Nếu không có thư mục output, cần upload hoặc tạo
```

### Lỗi: CUDA out of memory

```python
# Giảm batch size
BATCH_SIZE = 4  # hoặc 2

# Hoặc dùng CPU
DEVICE = 'cpu'
```

### Lỗi: Module not found

```python
# Cài lại dependencies
!pip install torch torch-geometric transformers scikit-learn --upgrade
```

---

## 📊 Expected Runtime on Colab

| Stage | Time (GPU T4) | Time (CPU) |
|-------|---------------|------------|
| Data Loading | 1-2 min | 2-3 min |
| Training (50 epochs) | 15-20 min | 2-3 hours |
| Testing | 30 sec | 2-3 min |

**Total:** ~20-25 minutes với GPU T4 (free tier)

---

## 💡 Tips for Colab

1. **Enable GPU:** Runtime → Change runtime type → GPU
2. **Keep session alive:** Run `!nvidia-smi` periodically
3. **Save to Drive:** Mount Google Drive để không mất model
4. **Use smaller epochs first:** Test với 5-10 epochs trước
5. **Monitor memory:** `!nvidia-smi` để check GPU usage
