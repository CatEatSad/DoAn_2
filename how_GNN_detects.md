# 🔍 GNN Nhận Diện Lỗi Như Thế Nào? - Chi Tiết Từng Bước

## 📋 BƯỚC 1: Training Phase - Model Học Patterns

### **Quá Trình Học:**

```
Input: 250 file JSON (125 vulnerable + 125 safe)
   ↓
Model học các PATTERNS phân biệt vulnerable vs safe
   ↓
Output: Trained model biết patterns của từng loại lỗi
```

---

## 🎓 Ví Dụ Cụ Thể: Command Injection

### **VULNERABLE CODE (từ file của bạn):**

```java
public class CommandInjectionExample {
    public static void main(String[] args) {
        String userInput = request.getParameter("cmd");
        Runtime.getRuntime().exec(userInput);  // ← NGUY HIỂM!
    }
}
```

### **Graph Representation từ Joern:**

```
Node 1:                          Node 2:                        Node 3:
┌─────────────────────┐         ┌──────────────────┐          ┌──────────────────────────┐
│ ID: 111669149696    │         │ ID: 94489280512  │          │ ID: 30064771072          │
│ Label: CALL         │         │ Label: IDENTIFIER│          │ ID: CALL                 │
│ Code: getParameter  │         │ Code: userInput  │          │ Code: exec(userInput)    │
│ Method: getParameter│    ┌───→│ Name: userInput  │─────────→│ Method: exec             │
└─────────────────────┘    │    └──────────────────┘          └──────────────────────────┘
          │                │             ↑                              ↑
          │                │             │                              │
          └────[AST]───────┘    [REACHING_DEF]              [ARGUMENT] ─┘
                                Data Flow!                   userInput là arg
```

### **GNN Học Pattern Này:**

```python
# Pattern nguy hiểm mà GNN học được:
DANGEROUS_PATTERN = {
    'source': 'getParameter',     # Nguồn từ user input
    'flow': 'REACHING_DEF',       # Data chảy qua REACHING_DEF edge
    'sink': 'Runtime.exec',       # Đổ vào hàm nguy hiểm
    'label': 'Command Injection'  # → LỖI!
}
```

---

### **SAFE CODE (đã fix):**

```java
public class CommandInjectionFixed {
    public static void main(String[] args) {
        String userInput = request.getParameter("cmd");
        
        // Validation!
        if (!userInput.matches("[a-zA-Z0-9]+")) {
            throw new Exception("Invalid input");
        }
        
        Runtime.getRuntime().exec(userInput);  // ← AN TOÀN (có validate)
    }
}
```

### **Graph Khác Biệt:**

```
Node 1:                    Node 2:                     Node 3:                    Node 4:
┌──────────────┐          ┌──────────────┐           ┌──────────────┐          ┌─────────────┐
│ getParameter │          │ userInput    │           │ matches()    │          │ exec()      │
└──────────────┘          └──────────────┘           │ VALIDATION   │          └─────────────┘
       │                         │                    └──────────────┘                 │
       │                         │                            │                        │
       └──[REACHING_DEF]────────→│──[REACHING_DEF]───────────→│──[CFG: if valid]─────→│
                                                    ↑
                                            Có node validation
```

### **GNN Học Pattern An Toàn:**

```python
SAFE_PATTERN = {
    'source': 'getParameter',
    'flow': 'REACHING_DEF',
    'validation': 'matches() or sanitize()',  # ← Có bước kiểm tra!
    'sink': 'Runtime.exec',
    'label': 'SAFE'
}
```

---

## 🧠 BƯỚC 2: Inference Phase - Nhận Diện Code Mới

### **Khi gặp code MỚI chưa từng thấy:**

```java
// Code mới người dùng submit
public class UnknownCode {
    public static void main(String[] args) {
        String cmd = System.getProperty("user.command");
        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.start();  // ← Có lỗi không?
    }
}
```

### **Step-by-Step Nhận Diện:**

#### **Step 1: Parse thành Graph**
```
Joern → JSON → Extract nodes & edges
```

#### **Step 2: Encode từng Node**
```python
# Node 1: System.getProperty("user.command")
node1_code = "System.getProperty(\"user.command\")"
node1_embedding = GraphCodeBERT(node1_code)
# Output: [0.12, -0.34, 0.56, ..., 0.89]  (768 chiều)

# Node 2: cmd
node2_code = "cmd"
node2_embedding = GraphCodeBERT(node2_code)

# Node 3: ProcessBuilder(cmd)
node3_code = "ProcessBuilder(cmd)"
node3_embedding = GraphCodeBERT(node3_code)
```

#### **Step 3: GNN Propagation**
```python
# Layer 1: Mỗi node học từ neighbors
node1_updated = node1_embedding  # Ban đầu

# Node 2 học từ Node 1 qua REACHING_DEF edge
node2_updated = attention(
    query=node2_embedding,
    key=node1_embedding,
    value=node1_embedding
) + node2_embedding

# GNN học được: "node2 (cmd) chứa data từ node1 (getProperty)"

# Node 3 học từ Node 2 qua ARGUMENT edge
node3_updated = attention(
    query=node3_embedding,
    key=node2_updated,  # ← Đã chứa info từ node1!
    value=node2_updated
) + node3_embedding

# GNN học được: "node3 (ProcessBuilder) nhận data từ node2,
#               mà node2 lại từ getProperty → TAINTED DATA!"
```

#### **Step 4: Graph Pooling**
```python
# Tổng hợp tất cả nodes
graph_embedding = mean([node1_updated, node2_updated, node3_updated])
# Hoặc dùng attention pooling
```

#### **Step 5: Classification**
```python
logits = classifier(graph_embedding)
# Output: [0.05, 0.02, 0.89, 0.01, 0.03]
#         Safe  Buf   CMD   Path  SQL
#                     ↑
#                  89% Command Injection!

prediction = argmax(logits) = 2  # Command Injection
confidence = softmax(logits)[2] = 0.89  # 89% chắc chắn
```

---

## 🎯 So Sánh Pattern Matching

### **GNN "Nhìn Thấy" Gì?**

#### **Ví dụ 1: SQL Injection - VULNERABLE**

```java
String username = request.getParameter("user");
String query = "SELECT * FROM users WHERE name='" + username + "'";
stmt.executeQuery(query);
```

**Graph Pattern:**
```
getParameter("user") ──[REACHING_DEF]──→ username ──[CONCAT]──→ query ──[ARGUMENT]──→ executeQuery()
     (Source)                              (Taint)              (Tainted)           (Sink)
```

**GNN nhận ra:**
```
✗ Source: getParameter (user input)
✗ Sink: executeQuery (SQL execution)
✗ NO sanitization in between
✗ String concatenation detected
→ VERDICT: SQL Injection! (Confidence: 94%)
```

---

#### **Ví dụ 2: SQL Injection - SAFE**

```java
String username = request.getParameter("user");
PreparedStatement pstmt = conn.prepareStatement("SELECT * FROM users WHERE name=?");
pstmt.setString(1, username);
pstmt.executeQuery();
```

**Graph Pattern:**
```
getParameter("user") ──[REACHING_DEF]──→ username ──[ARGUMENT]──→ setString() ──→ executeQuery()
     (Source)                              (Taint)    ↑                            (Sink)
                                                      │
                                           PreparedStatement (SAFE!)
```

**GNN nhận ra:**
```
✓ Source: getParameter
✓ Sink: executeQuery
✓ PreparedStatement detected (node type: CALL, method: prepareStatement)
✓ setString() used (parameterized query)
→ VERDICT: SAFE! (Confidence: 96%)
```

---

## 🔬 Attention Mechanism - Model "Chú Ý" Vào Đâu?

### **Ví dụ: Path Traversal**

```java
String filename = request.getParameter("file");
File file = new File("/uploads/" + filename);  // ← Vulnerable?
FileInputStream fis = new FileInputStream(file);
```

### **Attention Weights khi phân tích:**

```python
# Node importance scores (học được từ training)
attention_scores = {
    'getParameter("file")': 0.85,      # ← Quan trọng nhất!
    'filename': 0.72,
    'new File(... + filename)': 0.91,  # ← Rất quan trọng!
    'FileInputStream': 0.45            # ← Ít quan trọng hơn
}

# Edge importance
edge_attention = {
    'getParameter → filename': 0.88,        # Data flow quan trọng
    'filename → File constructor': 0.93,    # ← Critical!
    'File → FileInputStream': 0.35
}
```

### **Model Reasoning:**

```
Node "getParameter" có attention cao
    ↓
Edge "getParameter → filename" có attention cao
    ↓
Node "new File()" có attention cao + nhận data từ filename
    ↓
PATTERN match với "Path Traversal": 
    - User input (getParameter)
    - String concatenation ("/uploads/" + filename)
    - File system access (File constructor)
    - NO path validation
    ↓
VERDICT: Path Traversal (Confidence: 87%)
```

---

## 📊 Training Process - Model Học Như Thế Nào?

### **Dataset của bạn:**

```
Vulnerable:
  - Buffer_Overflow_0001_vul.json → Label: 1 (Buffer Overflow)
  - Command_Injection_0001_vul.json → Label: 2 (Command Injection)
  - Path_Traversal_0001_vul.json → Label: 3 (Path Traversal)
  - SQL_Injection_0001_vul.json → Label: 4 (SQL Injection)

Safe:
  - Buffer_Overflow_0001.json → Label: 0 (Safe)
  - Command_Injection_0001.json → Label: 0 (Safe)
  - ...
```

### **Training Loop:**

```python
for epoch in range(50):
    for batch in train_loader:
        # Batch có 8 graphs (4 vulnerable, 4 safe)
        
        # 1. Forward pass
        predictions = model(batch)
        # predictions = [
        #   [0.1, 0.05, 0.8, 0.03, 0.02],  # Graph 1 → Predict: Command Injection
        #   [0.92, 0.02, 0.03, 0.02, 0.01], # Graph 2 → Predict: Safe
        #   ...
        # ]
        
        # 2. Compute loss
        true_labels = [2, 0, 3, 0, 1, 0, 4, 0]  # Ground truth
        loss = CrossEntropyLoss(predictions, true_labels)
        
        # 3. Backpropagation
        loss.backward()
        
        # 4. Update weights
        optimizer.step()
        
        # Model learns:
        # - "If getParameter → exec without validation → Command Injection"
        # - "If PreparedStatement used → Safe"
        # - "If buffer size checked → Safe (no Buffer Overflow)"
```

### **Sau 50 epochs, model học được:**

```python
learned_patterns = {
    'Command Injection': {
        'sources': ['getParameter', 'readLine', 'System.getProperty'],
        'sinks': ['Runtime.exec', 'ProcessBuilder.start'],
        'safe_practices': ['whitelist validation', 'regex check'],
        'edge_patterns': 'source →[REACHING_DEF]→ sink WITHOUT validation'
    },
    
    'SQL Injection': {
        'sources': ['getParameter', 'request.getHeader'],
        'sinks': ['executeQuery', 'executeUpdate'],
        'safe_practices': ['PreparedStatement', 'setString/setInt'],
        'edge_patterns': 'source → String concat → executeQuery = VULNERABLE'
    },
    
    'Path Traversal': {
        'sources': ['getParameter', 'user input'],
        'sinks': ['File constructor', 'FileInputStream'],
        'safe_practices': ['path validation', 'canonical path check'],
        'edge_patterns': 'user_input → file_path WITHOUT sanitization'
    }
}
```

---

## 🎯 Real Example: Step-by-Step Detection

### **Input Code (chưa biết vulnerable hay không):**

```java
public class TestCode {
    public void processFile(HttpServletRequest req) {
        String path = req.getParameter("path");
        if (path.contains("..")) {  // ← Có validation nhưng yếu
            return;
        }
        File f = new File("/var/data/" + path);
        FileReader fr = new FileReader(f);
    }
}
```

### **GNN Analysis Process:**

#### **1. Graph Extraction:**
```json
{
  "nodes": [
    {"id": 1, "code": "req.getParameter(\"path\")", "label": "CALL"},
    {"id": 2, "code": "path", "label": "IDENTIFIER"},
    {"id": 3, "code": "path.contains(\"..\")", "label": "CALL"},
    {"id": 4, "code": "new File(\"/var/data/\" + path)", "label": "CALL"},
    {"id": 5, "code": "new FileReader(f)", "label": "CALL"}
  ],
  "edges": [
    {"from": 1, "to": 2, "type": "REACHING_DEF"},
    {"from": 2, "to": 3, "type": "ARGUMENT"},
    {"from": 2, "to": 4, "type": "ARGUMENT"},
    {"from": 4, "to": 5, "type": "REACHING_DEF"}
  ]
}
```

#### **2. Node Embeddings:**
```python
node_1_emb = GraphCodeBERT("req.getParameter(\"path\")")
# → [0.23, -0.45, 0.67, ..., 0.12]

node_3_emb = GraphCodeBERT("path.contains(\"..\")")
# → [0.15, 0.32, -0.28, ..., 0.56]

node_4_emb = GraphCodeBERT("new File(\"/var/data/\" + path)")
# → [0.67, -0.12, 0.34, ..., -0.23]
```

#### **3. GNN Propagation:**
```python
# Layer 1
node_2_updated = node_2_emb + attention(node_2_emb, node_1_emb)
# Node 2 học: "Tôi chứa data từ getParameter"

node_4_updated = node_4_emb + attention(node_4_emb, node_2_updated)
# Node 4 học: "Tôi nhận path từ user input"

# Layer 2 - Deeper understanding
node_4_final = node_4_updated + attention(node_4_updated, node_3_updated)
# Node 4 học thêm: "Có validation contains('..')nhưng..."
```

#### **4. Classification:**
```python
graph_emb = mean_pool([node_1_final, node_2_final, ..., node_5_final])

logits = classifier(graph_emb)
# [0.15, 0.05, 0.08, 0.68, 0.04]
#  Safe  Buf   CMD   Path  SQL
#                    ↑
#                68% Path Traversal

# Model reasoning:
# - Detected source: getParameter ✓
# - Detected sink: File constructor ✓
# - Detected validation: contains("..") ✓
# - BUT: Weak validation (can bypass with URL encoding, absolute paths)
# → Still VULNERABLE!
```

#### **5. Output:**
```json
{
  "prediction": "Path Traversal",
  "confidence": 0.68,
  "reasoning": {
    "source": "getParameter at line 3",
    "sink": "File constructor at line 7",
    "vulnerability": "Weak validation - can be bypassed",
    "suggestion": "Use Path.normalize() and canonical path check"
  }
}
```

---

## 💡 Key Insights: Tại Sao GNN Mạnh?

### **1. Context-Aware (Hiểu ngữ cảnh)**
```
Text-based: Chỉ thấy "executeQuery(query)"
GNN: Biết query từ đâu, qua những gì, có sanitize không
```

### **2. Structure-Aware (Hiểu cấu trúc)**
```
Text-based: Đọc code tuần tự dòng 1 → dòng 2 → dòng 3
GNN: Thấy toàn bộ control flow, data flow, quan hệ giữa các biến
```

### **3. Multi-hop Reasoning (Suy luận nhiều bước)**
```
getParameter → variable1 → variable2 → function → dangerous_sink
     (hop 1)      (hop 2)     (hop 3)    (hop 4)

GNN có thể trace qua 4-5 hops để tìm lỗi!
```

### **4. Edge Type Awareness (Hiểu loại quan hệ)**
```
AST edge: Quan hệ cú pháp
CFG edge: Luồng điều khiển
REACHING_DEF edge: Data flow (quan trọng nhất cho security!)
```

---

## 🚀 Summary

### **GNN Nhận Diện Lỗi Qua:**

1. **Pattern Matching**: So sánh graph pattern với patterns đã học
2. **Data Flow Analysis**: Trace data từ source → sink
3. **Attention Mechanism**: Tập trung vào nodes/edges quan trọng
4. **Multi-layer Propagation**: Hiểu sâu qua nhiều lớp GNN
5. **Classification**: Dự đoán loại lỗi dựa trên tổng hợp thông tin

### **Độ Chính Xác Cao Vì:**

- ✅ Hiểu **data flow** (quan trọng nhất)
- ✅ Hiểu **control flow** (if/else branches)
- ✅ Detect **validation** (hoặc thiếu validation)
- ✅ Phân biệt **safe practices** (PreparedStatement, sanitization)
- ✅ **Multi-hop reasoning** (trace qua nhiều biến)

### **So Với Regex/Static Analysis:**

| Method | Data Flow | Control Flow | Learning | False Positives |
|--------|-----------|--------------|----------|-----------------|
| Regex | ❌ | ❌ | ❌ | Very High |
| Static Analysis | ✓ | ✓ | ❌ | High |
| **GNN** | ✓✓ | ✓✓ | ✓✓ | **Low** |

**GNN = Static Analysis + Machine Learning!** 🎯
