# Ý Tưởng Sử Dụng GraphCodeBERT Cho Phát Hiện Lỗi Bảo Mật

## 📊 Phân Tích Dataset Hiện Có

### Cấu trúc dữ liệu:
- **Vulnerable code**: `/output/` - JSON files với AST từ Joern
- **Safe code**: `/output_safe/` - JSON files với AST đã fix
- **Vulnerability types**: Buffer Overflow, Command Injection, Path Traversal, SQL Injection

### Đặc điểm của JSON files:
```json
{
  "functions": [{
    "function": "ClassName.main",
    "AST": [
      {
        "id": "...",
        "label": "METHOD_PARAMETER_IN",
        "properties": {
          "NAME": "args",
          "CODE": "String[] args",
          "LINE_NUMBER": "5",
          "TYPE_FULL_NAME": "..."
        },
        "edges": [...]
      }
    ]
  }]
}
```

## 🎯 5 Ý Tưởng Chính

### 1️⃣ Binary Classification (Đơn Giản Nhất)

**Mục tiêu**: Phân loại code là Vulnerable (1) hoặc Safe (0)

**Pipeline**:
```
JSON → Extract AST Graph → GraphCodeBERT Encoding → Binary Classifier → Safe/Vulnerable
```

**Implementation Steps**:
1. Parse JSON files để extract graph structure
2. Convert graph thành format phù hợp với GraphCodeBERT
3. Fine-tune GraphCodeBERT với binary classification head
4. Training với labeled data (vulnerable=1, safe=0)

**Code Structure**:
```python
# Preprocessing
def parse_joern_json(json_file):
    # Extract nodes, edges, properties
    # Return graph representation
    pass

# Model
class VulnerabilityDetector(nn.Module):
    def __init__(self):
        self.graphcodebert = RobertaModel.from_pretrained("microsoft/graphcodebert-base")
        self.classifier = nn.Linear(768, 2)  # Binary classification
    
    def forward(self, code_inputs, graph_inputs):
        # Encode with GraphCodeBERT
        # Classify
        pass
```

**Ưu điểm**:
- ✅ Đơn giản, dễ implement
- ✅ Phù hợp với dataset có cặp vulnerable/safe
- ✅ Baseline tốt để so sánh

**Nhược điểm**:
- ❌ Không phân biệt được loại lỗi
- ❌ Không tận dụng hết thông tin về fix

---

### 2️⃣ Multi-Class Classification (Phân Loại Chi Tiết)

**Mục tiêu**: Phân loại code theo 5 classes

**Classes**:
- 0: Safe Code
- 1: Buffer Overflow
- 2: Command Injection
- 3: Path Traversal
- 4: SQL Injection

**Pipeline**:
```
JSON → AST Graph → GraphCodeBERT → Multi-Class Classifier → Vulnerability Type
```

**Dataset Distribution** (cần kiểm tra):
```python
# Count files in each category
categories = {
    'Buffer_Overflow': len(glob('output/Buffer_Overflow/*.json')),
    'Command_Injection': len(glob('output/Command_Injection/*.json')),
    'Path_Traversal': len(glob('output/Path_Traversal/*.json')),
    'SQL_Injection': len(glob('output/SQL_Injection/*.json')),
    'Safe': total_safe_files
}
```

**Model**:
```python
class MultiClassVulnerabilityDetector(nn.Module):
    def __init__(self, num_classes=5):
        self.graphcodebert = RobertaModel.from_pretrained("microsoft/graphcodebert-base")
        self.classifier = nn.Linear(768, num_classes)
```

**Ưu điểm**:
- ✅ Phát hiện được loại lỗi cụ thể
- ✅ Hữu ích cho việc suggest fix
- ✅ Tận dụng 4 loại lỗi trong dataset

**Nhược điểm**:
- ❌ Cần balance dataset (có thể một số loại lỗi ít hơn)
- ❌ Phức tạp hơn binary classification

---

### 3️⃣ Contrastive Learning (Học Từ Cặp Code)

**Mục tiêu**: Học representation sao cho vulnerable và fixed version gần nhau trong embedding space

**Approach**:
```
Positive pairs: (vulnerable_code, safe_fix) - same functionality
Negative pairs: different functionalities
```

**Architecture**:
```
                    ┌─→ Vulnerable Code → GraphCodeBERT → Embedding_v
Input Pair ─────────┤
                    └─→ Safe Fix Code → GraphCodeBERT → Embedding_s

Loss = Contrastive Loss(Embedding_v, Embedding_s)
```

**Loss Functions**:
1. **Triplet Loss**:
   - Anchor: Vulnerable code
   - Positive: Its safe fix
   - Negative: Different vulnerability
   
2. **NT-Xent Loss** (SimCLR style):
   ```python
   def contrastive_loss(z_v, z_s, temperature=0.07):
       similarity = cosine_similarity(z_v, z_s) / temperature
       return -log(exp(similarity) / sum(exp(all_similarities)))
   ```

**Training Strategy**:
```python
for vulnerable_file, safe_file in paired_dataset:
    emb_v = model(vulnerable_code)
    emb_s = model(safe_code)
    
    # Pull together (positive pair)
    positive_loss = distance(emb_v, emb_s)
    
    # Push away (negative pairs)
    negative_loss = -distance(emb_v, random_other_code)
    
    total_loss = positive_loss + negative_loss
```

**Use Cases**:
1. **Similarity Search**: Tìm code tương tự để suggest fix
2. **Clustering**: Nhóm các lỗi tương tự nhau
3. **Transfer Learning**: Pre-train rồi fine-tune cho classification

**Ưu điểm**:
- ✅ Học được mối quan hệ vulnerable-safe
- ✅ Có thể suggest fix dựa trên similarity
- ✅ Robust với unseen vulnerability types

---

### 4️⃣ Graph Neural Network Approach (Tận Dụng AST)

**Đặc điểm data từ Joern**:
```json
{
  "id": "111669149696",
  "label": "METHOD_PARAMETER_IN",
  "properties": {
    "NAME": "args",
    "CODE": "String[] args",
    "LINE_NUMBER": "5",
    "TYPE_FULL_NAME": "<unresolvedNamespace>.String[]"
  },
  "edges": [
    {"type": "AST", "out": "107374182400"},
    {"type": "REACHING_DEF", "out": "..."}
  ]
}
```

**Graph Features**:
1. **Node Features**:
   - Label (METHOD_PARAMETER_IN, CALL, IDENTIFIER, etc.)
   - Code snippet
   - Line number
   - Type information

2. **Edge Types**:
   - AST (Abstract Syntax Tree)
   - CFG (Control Flow)
   - REACHING_DEF (Data flow)
   - EVAL_TYPE (Type information)

**Model Architecture**:
```
Graph Structure 
    ↓
GraphCodeBERT (encode nodes with code context)
    ↓
Graph Attention Networks (propagate information)
    ↓
Graph Pooling (aggregate node embeddings)
    ↓
Classification
```

**Implementation**:
```python
class GraphVulnerabilityDetector(nn.Module):
    def __init__(self):
        # Encode each node's code
        self.code_encoder = RobertaModel.from_pretrained("microsoft/graphcodebert-base")
        
        # Graph layers to propagate information
        self.gat1 = GATConv(768, 256, heads=8)
        self.gat2 = GATConv(256*8, 128, heads=4)
        
        # Pooling
        self.pool = global_mean_pool
        
        # Classifier
        self.classifier = nn.Linear(128*4, num_classes)
    
    def forward(self, node_codes, edge_index, edge_types):
        # Encode node codes
        node_embeddings = [self.code_encoder(code).last_hidden_state[:, 0] 
                          for code in node_codes]
        
        # GNN propagation
        x = self.gat1(node_embeddings, edge_index)
        x = F.relu(x)
        x = self.gat2(x, edge_index)
        
        # Pool to graph-level
        graph_embedding = self.pool(x, batch)
        
        # Classify
        return self.classifier(graph_embedding)
```

**Tận dụng edge types**:
```python
# Heterogeneous graph with different edge types
class HeteroGraphModel(nn.Module):
    def __init__(self):
        self.ast_conv = GATConv(768, 256)
        self.cfg_conv = GATConv(768, 256)
        self.data_flow_conv = GATConv(768, 256)
        
    def forward(self, x, edge_dict):
        # Different processing for different edge types
        ast_out = self.ast_conv(x, edge_dict['AST'])
        cfg_out = self.cfg_conv(x, edge_dict['CFG'])
        df_out = self.data_flow_conv(x, edge_dict['REACHING_DEF'])
        
        # Combine
        return ast_out + cfg_out + df_out
```

**Ưu điểm**:
- ✅ Tận dụng đầy đủ cấu trúc graph từ Joern
- ✅ Capture được control flow và data flow
- ✅ Phù hợp với code analysis

**Nhược điểm**:
- ❌ Phức tạp hơn, cần nhiều tài nguyên
- ❌ Khó debug

---

### 5️⃣ Hierarchical Classification (Phân Cấp)

**Motivation**: Một số lỗi dễ phát hiện hơn, một số khó hơn

**Architecture**:
```
Level 1: Safe vs Vulnerable (easier)
    ↓
Level 2: Vulnerability Type Classification (harder)
    ├─→ Buffer Overflow
    ├─→ Command Injection
    ├─→ Path Traversal
    └─→ SQL Injection
```

**Model**:
```python
class HierarchicalClassifier(nn.Module):
    def __init__(self):
        self.encoder = GraphCodeBERT()
        
        # Level 1: Binary
        self.level1_classifier = nn.Linear(768, 2)
        
        # Level 2: Multi-class (only for vulnerable)
        self.level2_classifier = nn.Linear(768, 4)
    
    def forward(self, x):
        features = self.encoder(x)
        
        # First classify safe vs vulnerable
        level1_logits = self.level1_classifier(features)
        
        # If vulnerable, classify type
        if level1_logits.argmax() == 1:  # Vulnerable
            level2_logits = self.level2_classifier(features)
            return level1_logits, level2_logits
        else:
            return level1_logits, None
```

**Training**:
```python
# Joint training with weighted loss
loss = alpha * level1_loss + beta * level2_loss
```

**Ưu điểm**:
- ✅ Chia nhỏ bài toán phức tạp
- ✅ Có thể focus vào vulnerable code
- ✅ Interpretable

---

## 🔥 Recommendation: Approach Nào Nên Bắt Đầu?

### **Giai Đoạn 1: Baseline (1-2 tuần)**
→ **Binary Classification (#1)**
- Nhanh, đơn giản
- Verify dataset quality
- Establish baseline performance

### **Giai Đoạn 2: Improve (2-3 tuần)**
→ **Multi-Class Classification (#2)**
- Phân loại chi tiết hơn
- So sánh với baseline
- Analyze per-class performance

### **Giai Đoạn 3: Advanced (3-4 tuần)**
→ **Graph-based Approach (#4)**
- Tận dụng AST structure
- Potentially best performance
- Publication-worthy

### **Giai Đoạn 4: Research (Optional)**
→ **Contrastive Learning (#3)**
- Novel approach
- Useful for code suggestion
- Good for research paper

---

## 📋 Implementation Checklist

### Data Preparation
- [ ] Count files per vulnerability type
- [ ] Parse JSON to extract graph structure
- [ ] Split train/val/test (70/15/15)
- [ ] Balance dataset (if needed)
- [ ] Create data loaders

### Model Development
- [ ] Setup GraphCodeBERT
- [ ] Implement preprocessing pipeline
- [ ] Build model architecture
- [ ] Define loss function
- [ ] Setup training loop

### Evaluation
- [ ] Accuracy, Precision, Recall, F1
- [ ] Confusion matrix
- [ ] Per-class metrics
- [ ] Error analysis

### Experimentation
- [ ] Hyperparameter tuning
- [ ] Different learning rates
- [ ] Different architectures
- [ ] Ensemble methods

---

## 🛠️ Technical Stack

```python
# Core libraries
- torch >= 2.0
- transformers >= 4.30 (for GraphCodeBERT)
- torch-geometric (for GNN approaches)
- scikit-learn (for metrics)
- pandas (for data handling)
- wandb (for experiment tracking)

# GraphCodeBERT
from transformers import RobertaTokenizer, RobertaModel
tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
model = RobertaModel.from_pretrained("microsoft/graphcodebert-base")
```

---

## 📊 Expected Results

### Binary Classification
- Expected Accuracy: **85-90%**
- Why: Clear difference between vulnerable and safe code

### Multi-Class Classification
- Expected Accuracy: **75-85%**
- Why: Some vulnerability types overlap

### Graph-based Approach
- Expected Accuracy: **88-93%**
- Why: Leverage structural information

---

## 💡 Bonus Ideas

### 6️⃣ Explainability
- Use attention weights to highlight vulnerable code snippets
- Generate explanations: "Vulnerable because of unsanitized input at line X"

### 7️⃣ Code Fix Suggestion
- Train sequence-to-sequence model
- Input: Vulnerable code
- Output: Fixed code
- Based on paired dataset

### 8️⃣ Ensemble
- Combine multiple approaches
- Voting: Binary + Multi-class + Graph
- Boost performance by 2-5%

### 9️⃣ Active Learning
- Model suggests which code to label next
- Efficient use of labeling effort

### 🔟 Transfer Learning
- Pre-train on larger code corpus
- Fine-tune on vulnerability detection
- Improve generalization
