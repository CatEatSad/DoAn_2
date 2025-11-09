# 🚀 Hướng Dẫn Chạy Trên Google Colab

## ⚠️ Lỗi Bạn Đang Gặp

```
Loaded 0 samples for all
ValueError: With n_samples=0, test_size=0.3...
```

**Nguyên nhân:** Không tìm thấy file JSON trong thư mục `output/` và `output_safe/`

---

## 🔧 Giải Quyết - 3 Bước

### **Bước 1: Kiểm Tra Files**

Chạy trong Colab:

```python
# Clone repo
!git clone https://github.com/CatEatSad/DoAn_2.git
%cd DoAn_2

# Kiểm tra có files không
!python debug_data.py
```

**Nếu output là:**
```
TOTAL: 0 files
⚠️ NO JSON FILES FOUND!
```

→ **Bạn chưa upload files JSON!**

---

### **Bước 2A: Nếu Có Files Trên Local**

Upload files lên repo GitHub:

```bash
# Trên máy local (Windows)
cd "c:\Users\abcdx\OneDrive\Máy tính\renew"

# Add files
git add output/ output_safe/
git commit -m "Add JSON files"
git push
```

Sau đó pull lại trong Colab:

```python
%cd /content/DoAn_2
!git pull
!python debug_data.py  # Should show files now
```

---

### **Bước 2B: Upload Trực Tiếp Lên Colab**

**Option 1: Upload từ Google Drive**

```python
from google.colab import drive
drive.mount('/content/drive')

# Copy files từ Drive
!cp -r /content/drive/MyDrive/your_output_folder /content/DoAn_2/output
!cp -r /content/drive/MyDrive/your_output_safe_folder /content/DoAn_2/output_safe

# Check
!python debug_data.py
```

**Option 2: Upload ZIP file**

```python
from google.colab import files
import zipfile

# Upload ZIP
uploaded = files.upload()  # Choose your data.zip

# Extract
!unzip -q data.zip -d /content/DoAn_2/
!python debug_data.py
```

**Option 3: Download từ URL**

```python
# Nếu bạn có files ở đâu đó (Dropbox, Google Drive public link)
!wget -O data.zip "YOUR_DOWNLOAD_LINK"
!unzip -q data.zip -d /content/DoAn_2/
```

---

### **Bước 3: Train Model**

Sau khi có files (debug_data.py show > 0 files):

```python
# Install dependencies
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q torch-geometric transformers scikit-learn tqdm

# Train
%cd /content/DoAn_2/src
!python train.py
```

---

## 📁 Cấu Trúc Files Cần Có

```
DoAn_2/
├── output/                    ← VULNERABLE code
│   ├── Buffer_Overflow/
│   │   ├── Buffer_Overflow_0001_vul.json
│   │   ├── Buffer_Overflow_0002_vul.json
│   │   └── ...
│   ├── Command_Injection/
│   │   ├── Command_Injection_0001_vul.json
│   │   └── ...
│   ├── Path_Traversal/
│   │   └── ...
│   └── SQL_Injection/
│       └── ...
│
└── output_safe/               ← SAFE code
    ├── Buffer_Overflow/
    │   ├── Buffer_Overflow_0001.json
    │   └── ...
    ├── Command_Injection/
    ├── Path_Traversal/
    └── SQL_Injection/
```

---

## 🎯 Quick Fix Script

Chạy trong Colab để tự động fix:

```python
import os
from pathlib import Path

# 1. Check current state
print("Checking data...")
!python /content/DoAn_2/debug_data.py

# 2. If no files, try pull from repo
print("\nTrying to pull from GitHub...")
%cd /content/DoAn_2
!git pull

# 3. Check again
!python debug_data.py

# 4. If still no files, need manual upload
data_exists = len(list(Path('/content/DoAn_2/output').rglob('*.json'))) > 0

if not data_exists:
    print("\n" + "="*70)
    print("⚠️ NO DATA FOUND!")
    print("="*70)
    print("\nPlease upload data using one of these methods:")
    print("1. Push to GitHub repo first")
    print("2. Upload via Google Drive")
    print("3. Upload ZIP file directly to Colab")
    print("\nSee COLAB_TROUBLESHOOTING.md for details")
else:
    print("\n✓ Data found! Ready to train")
    print("\nRun: %cd /content/DoAn_2/src && !python train.py")
```

---

## 🐛 Common Issues

### Issue 1: Git clone fails
```python
# Solution: Use HTTPS instead of SSH
!git clone https://github.com/CatEatSad/DoAn_2.git
```

### Issue 2: Files exist but still 0 samples
```python
# Check file permissions
!ls -la /content/DoAn_2/output/Command_Injection/

# Try reading a file manually
import json
with open('/content/DoAn_2/output/Command_Injection/Command_Injection_0001_vul.json') as f:
    data = json.load(f)
print(data.keys())
```

### Issue 3: CUDA out of memory
```python
# Use smaller batch size or CPU
# Edit train.py line 261:
BATCH_SIZE = 4  # or 2
DEVICE = 'cpu'  # if GPU fails
```

---

## 📞 Need Help?

Nếu vẫn lỗi, chạy:

```python
# Full diagnostic
!python /content/DoAn_2/debug_data.py > /content/debug_output.txt
!cat /content/debug_output.txt
```

Và gửi output để được hỗ trợ!
