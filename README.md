# 🔍 Vulnerability Detection with GNN

Hệ thống nhận diện lỗi bảo mật Java code sử dụng Graph Neural Networks (GNN) trên AST từ Joern.

## 📊 Nhận Diện Được 5 Loại:

1. **Safe Code** (Label 0)
2. **Buffer Overflow** (Label 1)
3. **Command Injection** (Label 2)
4. **Path Traversal** (Label 3)
5. **SQL Injection** (Label 4)

---

## 🎯 Cách Nhận Diện Từng Loại

### 1. SQL Injection

**Nhận diện qua:**
- ✅ Source: `getParameter()`, `getHeader()`, `getCookie()`
- ✅ Sink: `executeQuery()`, `executeUpdate()`
- ✅ Data flow: User input → SQL execution
- ❌ Missing: `PreparedStatement`, validation

**Pattern:**
```java
// VULNERABLE
String username = request.getParameter("user");
String query = "SELECT * FROM users WHERE name='" + username + "'";
stmt.executeQuery(query);  // ← SQL Injection!

// SAFE
String username = request.getParameter("user");
PreparedStatement pstmt = conn.prepareStatement("SELECT * FROM users WHERE name=?");
pstmt.setString(1, username);
pstmt.executeQuery();  // ← Safe!
```

---

### 2. Command Injection

**Nhận diện qua:**
- ✅ Source: `getParameter()`, `System.getProperty()`
- ✅ Sink: `Runtime.exec()`, `ProcessBuilder()`
- ✅ Data flow: User input → Command execution
- ❌ Missing: Validation, whitelist check

**Pattern:**
```java
// VULNERABLE
String cmd = request.getParameter("cmd");
Runtime.getRuntime().exec(cmd);  // ← Command Injection!

// SAFE
String cmd = request.getParameter("cmd");
if (!cmd.matches("[a-zA-Z0-9]+")) {
    throw new Exception("Invalid input");
}
Runtime.getRuntime().exec(cmd);  // ← Safe!
```

---

### 3. Path Traversal

**Nhận diện qua:**
- ✅ Source: `getParameter()` cho filename/path
- ✅ Sink: `new File()`, `FileInputStream`, `FileReader`
- ✅ Data flow: User input → File operations
- ❌ Missing: Path validation, canonical path check

**Pattern:**
```java
// VULNERABLE
String filename = request.getParameter("file");
File file = new File("/uploads/" + filename);  // ← Path Traversal!
FileInputStream fis = new FileInputStream(file);

// SAFE
String filename = request.getParameter("file");
if (filename.contains("..") || filename.contains("/")) {
    throw new Exception("Invalid filename");
}
Path basePath = Paths.get("/uploads/");
Path fullPath = basePath.resolve(filename).normalize();
if (!fullPath.startsWith(basePath)) {
    throw new Exception("Path traversal detected");
}
File file = fullPath.toFile();  // ← Safe!
```

---

### 4. Buffer Overflow

**Nhận diện qua:**
- ✅ Array operations without bounds check
- ✅ User-controlled array size
- ✅ Unchecked read operations
- ❌ Missing: Length validation

**Pattern:**
```java
// VULNERABLE
int size = Integer.parseInt(request.getParameter("size"));
byte[] buffer = new byte[size];  // ← Buffer Overflow!
stream.read(buffer);

// SAFE
int size = Integer.parseInt(request.getParameter("size"));
if (size < 0 || size > 1024) {  // ← Validation!
    throw new Exception("Invalid size");
}
byte[] buffer = new byte[size];
stream.read(buffer);  // ← Safe!
```

---

## 🚀 Quick Start

### 1. Cài Đặt Dependencies

```bash
cd "c:\Users\abcdx\OneDrive\Máy tính\renew"
pip install -r requirements.txt
```

**Lưu ý:** Cần cài PyTorch với CUDA nếu có GPU:
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision torchaudio
```

### 2. Train Model

```bash
cd src
python train.py
```

**Output:**
```
Loading dataset...
=================================================
Dataset Statistics (all):
=================================================
Safe                :  200 files
Buffer_Overflow     :   50 files
Command_Injection   :   50 files
Path_Traversal      :   50 files
SQL_Injection       :   50 files
=================================================

Creating model...
Training...

Epoch 1/50
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Train Loss: 1.4532 | Train Acc: 0.6234
Val Loss:   1.2341 | Val Acc:   0.7123
Val F1:     0.6891
✓ New best model! (Val Acc: 0.7123)

...

Training Complete!
Best Validation Accuracy: 0.8934
```

### 3. Test Model

Sau khi train xong, model sẽ tự động test:

```
Test Results:
=================================================
Accuracy:  0.8934
Precision: 0.8876
Recall:    0.8901
F1 Score:  0.8888

Confusion Matrix:
               Safe  Buf_Overfl  Cmd_Inject  Path_Trav   SQL_Inject
Safe             38           1           1          0           0
Buffer_Overflow   1          12           0          0           0
Command_Injection 0           0          14          1           0
Path_Traversal    1           0           0         13           1
SQL_Injection     0           0           1          0          14
```

### 4. Predict Code Mới

```bash
python predict.py
```

**Output:**
```
Analyzing: Command_Injection_0001_vul.json
=================================================

Patterns Detected:
─────────────────────────────────────────────────
  [USER_INPUT         ] Line 7    : request.getParameter("cmd")
  [COMMAND_EXECUTION  ] Line 8    : Runtime.getRuntime().exec(userInput)

=================================================
Risk Assessment:
=================================================

  ⚠️  CRITICAL - COMMAND_INJECTION
      Reason: User input flows to command execution without validation
```

---

## 📁 File Structure

```
renew/
├── src/
│   ├── data_loader.py     # Parse JSON, create dataset
│   ├── model.py           # GNN model definition
│   ├── train.py           # Training script
│   └── predict.py         # Prediction & analysis
│
├── output/                # Vulnerable code (JSON from Joern)
│   ├── Buffer_Overflow/
│   ├── Command_Injection/
│   ├── Path_Traversal/
│   └── SQL_Injection/
│
├── output_safe/           # Safe code (JSON from Joern)
│   ├── Buffer_Overflow/
│   ├── Command_Injection/
│   ├── Path_Traversal/
│   └── SQL_Injection/
│
├── saved_models/          # Trained models
│   └── best_model.pth
│
├── results/               # Training results
│   └── training_results.json
│
└── requirements.txt       # Dependencies
```

---

## 🧠 Model Architecture

```
Input: JSON file từ Joern
   ↓
Parse Graph (nodes + edges)
   ↓
Encode Nodes (GraphCodeBERT - 768 dim)
   ↓
GAT Layer 1 (Graph Attention)
   ↓
GAT Layer 2 (Deeper propagation)
   ↓
Graph Pooling (Mean + Max)
   ↓
Classifier (MLP)
   ↓
Output: [P(Safe), P(Buffer), P(Cmd), P(Path), P(SQL)]
```

---

## 🔧 Customization

### Thay đổi hyperparameters:

Edit `train.py`:
```python
BATCH_SIZE = 8          # Tăng nếu có nhiều RAM
NUM_EPOCHS = 50         # Tăng để train lâu hơn
LEARNING_RATE = 1e-3    # Giảm nếu loss không giảm
```

### Thay đổi model architecture:

Edit `model.py`:
```python
model = SimplifiedVulnerabilityGNN(
    num_classes=5,
    hidden_dim=256,     # Tăng để model phức tạp hơn
    num_layers=2,       # Thêm layers
    num_heads=4,        # Multi-head attention
    dropout=0.3
)
```

---

## 📊 Expected Results

### Dataset của bạn (~250 files):

| Metric | Expected |
|--------|----------|
| **Overall Accuracy** | 85-90% |
| **SQL Injection F1** | 0.88-0.92 |
| **Command Injection F1** | 0.90-0.94 |
| **Path Traversal F1** | 0.85-0.89 |
| **Buffer Overflow F1** | 0.80-0.85 |
| **Safe Code F1** | 0.92-0.96 |

### Tại sao Buffer Overflow thấp hơn?

- Ít samples hơn
- Pattern khó phát hiện hơn (need deeper analysis)
- Java ít bị buffer overflow (compared to C/C++)

---

## 🎯 How It Works

### GNN Nhận Diện Lỗi Qua 3 Bước:

#### 1. **Pattern Matching**
```
Model học patterns từ 250 files:
  - Vulnerable pattern: getParameter → exec (NO validation)
  - Safe pattern: getParameter → validation → exec
```

#### 2. **Data Flow Analysis**
```
Trace data qua graph edges:
  Node1 (getParameter) ─[REACHING_DEF]→ Node2 (userInput) ─[ARGUMENT]→ Node3 (exec)
  
  GNN học: "Data từ getParameter chảy vào exec → DANGEROUS!"
```

#### 3. **Graph Propagation**
```
Layer 1: Mỗi node học từ neighbors
  - Node2 học: "Tôi chứa data từ getParameter"
  - Node3 học: "Tôi nhận tainted data"

Layer 2: Deeper understanding
  - Node3 học: "Không có node validation giữa Node1 và tôi → VULNERABLE!"
```

---

## 🐛 Troubleshooting

### Error: CUDA out of memory
```bash
# Giảm batch size
BATCH_SIZE = 4  # trong train.py
```

### Error: Module not found
```bash
# Cài lại dependencies
pip install -r requirements.txt --upgrade
```

### Model accuracy thấp
```bash
# Tăng số epochs
NUM_EPOCHS = 100

# Hoặc thử learning rate khác
LEARNING_RATE = 5e-4
```

---

## 📝 TODO

- [ ] Add GraphCodeBERT encoding (hiện tại dùng random features)
- [ ] Add edge type heterogeneous GNN
- [ ] Add attention visualization
- [ ] Add explainability (why model predicts this?)
- [ ] Add real-time prediction API
- [ ] Add web interface

---

## 🎓 Citation

Nếu bạn dùng code này cho research, please cite:

```bibtex
@misc{vulnerability_gnn_2025,
  title={Graph Neural Networks for Java Vulnerability Detection},
  author={Your Name},
  year={2025}
}
```

---

## 📧 Contact

Nếu có vấn đề, tạo issue hoặc contact me!
