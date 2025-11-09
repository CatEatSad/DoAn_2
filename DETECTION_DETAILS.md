# 🎯 Cách Model Nhận Diện Từng Loại Lỗi - Chi Tiết

## Tổng Quan

Model sử dụng **Graph Neural Network (GNN)** để phân tích **cấu trúc graph** từ Joern và nhận diện patterns của từng loại lỗi.

---

## 1️⃣ SQL Injection

### **Cách Nhận Diện:**

```python
# Model tìm pattern này trong graph:

Source Node (USER INPUT)
    ↓
    [REACHING_DEF edge] ← Data flow!
    ↓
Tainted Variable
    ↓
    [String concatenation] ← Nguy hiểm!
    ↓
Sink Node (SQL EXECUTION)
```

### **Graph Signatures:**

**VULNERABLE:**
```
Node A: getParameter("username")
  ↓ [REACHING_DEF]
Node B: username (IDENTIFIER)
  ↓ [ADDITION/CONCAT]
Node C: "SELECT * FROM users WHERE name='" + username + "'"
  ↓ [ARGUMENT]
Node D: stmt.executeQuery(query)

→ VERDICT: SQL INJECTION
```

**SAFE:**
```
Node A: getParameter("username")
  ↓ [REACHING_DEF]
Node B: username
  ↓ [ARGUMENT]
Node C: pstmt.setString(1, username)  ← PreparedStatement!
  ↓
Node D: pstmt.executeQuery()

→ VERDICT: SAFE (PreparedStatement detected)
```

### **Features Model Học:**

```python
sql_features = {
    'has_getParameter': True/False,
    'has_executeQuery': True/False,
    'has_string_concat': True/False,  # + operator
    'has_PreparedStatement': True/False,
    'has_setString': True/False,
    'data_flow_path_length': int,  # Số hops từ source → sink
}

# Decision:
if (has_getParameter and 
    has_executeQuery and 
    has_string_concat and 
    NOT has_PreparedStatement):
    → SQL_INJECTION
```

---

## 2️⃣ Command Injection

### **Cách Nhận Diện:**

```python
# Pattern:

User Input Source
    ↓
    [REACHING_DEF]
    ↓
Command Variable
    ↓
    [ARGUMENT]
    ↓
Runtime.exec() / ProcessBuilder
```

### **Graph Signatures:**

**VULNERABLE:**
```
Node A: request.getParameter("cmd")
  ↓ [REACHING_DEF]
Node B: userCmd (IDENTIFIER)
  ↓ [ARGUMENT]
Node C: Runtime.getRuntime().exec(userCmd)

→ VERDICT: COMMAND INJECTION
```

**SAFE với Validation:**
```
Node A: request.getParameter("cmd")
  ↓ [REACHING_DEF]
Node B: userCmd
  ↓ [ARGUMENT]
Node C: userCmd.matches("[a-zA-Z0-9]+")  ← VALIDATION!
  ↓ [CFG - conditional]
Node D: Runtime.exec(userCmd)

→ VERDICT: SAFE (Validation detected)
```

### **Features Model Học:**

```python
command_features = {
    'has_user_input': True/False,
    'has_Runtime_exec': True/False,
    'has_ProcessBuilder': True/False,
    'has_shell_invocation': True/False,  # /bin/sh, cmd.exe
    'has_validation_node': True/False,   # matches(), Pattern.compile()
    'has_whitelist_check': True/False,
}

# Decision:
if (has_user_input and 
    (has_Runtime_exec or has_ProcessBuilder) and 
    NOT has_validation_node):
    → COMMAND_INJECTION
```

### **Đặc Biệt: Shell Usage Detection**

```
Node: Runtime.exec("/bin/sh -c " + cmd)
                    ↑
            Shell invocation → CRITICAL!

→ Severity: CRITICAL (shell allows command chaining với ; & | )
```

---

## 3️⃣ Path Traversal

### **Cách Nhận Diện:**

```python
# Pattern:

User Input (filename/path)
    ↓
    [REACHING_DEF]
    ↓
Path Variable
    ↓
    [String concatenation]
    ↓
File Constructor / FileInputStream
```

### **Graph Signatures:**

**VULNERABLE:**
```
Node A: request.getParameter("file")
  ↓ [REACHING_DEF]
Node B: filename
  ↓ [ADDITION]
Node C: new File("/uploads/" + filename)  ← Concatenation!
  ↓ [ARGUMENT]
Node D: new FileInputStream(file)

→ VERDICT: PATH TRAVERSAL
```

**SAFE với Path Validation:**
```
Node A: request.getParameter("file")
  ↓ [REACHING_DEF]
Node B: filename
  ↓ [ARGUMENT]
Node C: filename.contains("..")  ← Check 1
  ↓ [CFG]
Node D: path.getCanonicalPath()  ← Check 2
  ↓ [CFG]
Node E: path.startsWith(basePath)  ← Check 3
  ↓
Node F: new File(path)

→ VERDICT: SAFE (Multiple validations)
```

### **Features Model Học:**

```python
path_features = {
    'has_user_input': True/False,
    'has_file_constructor': True/False,
    'has_file_stream': True/False,
    'has_path_concat': True/False,  # String + for paths
    'has_dotdot_check': True/False,  # contains("..")
    'has_canonical_check': True/False,  # getCanonicalPath()
    'has_startsWith_check': True/False,  # startsWith(basePath)
}

# Decision:
if (has_user_input and 
    (has_file_constructor or has_file_stream) and 
    has_path_concat and 
    NOT (has_canonical_check or has_startsWith_check)):
    → PATH_TRAVERSAL
```

### **Weak Validation Detection:**

```python
# Model biết validations này yếu:
weak_validations = [
    'contains("..")',  # Có thể bypass bằng URL encoding
    'replace("..", "")',  # Có thể bypass bằng "..../"
    'startsWith("/")',  # Không đủ
]

# Strong validations:
strong_validations = [
    'getCanonicalPath() + startsWith()',  # Best!
    'Path.normalize()',
    'Whitelist exact filenames',
]
```

---

## 4️⃣ Buffer Overflow

### **Cách Nhận Diện:**

```python
# Pattern (Java specific):

User-controlled Size
    ↓
    [Used in array allocation]
    ↓
byte[]/char[] Creation
    ↓
    [NO bounds check]
    ↓
Read/Write Operations
```

### **Graph Signatures:**

**VULNERABLE:**
```
Node A: request.getParameter("size")
  ↓ [REACHING_DEF]
Node B: Integer.parseInt(size)
  ↓ [ARGUMENT]
Node C: new byte[size]  ← User-controlled size!
  ↓ [REACHING_DEF]
Node D: stream.read(buffer)  ← NO limit check

→ VERDICT: BUFFER OVERFLOW RISK
```

**SAFE:**
```
Node A: request.getParameter("size")
  ↓
Node B: Integer.parseInt(size)
  ↓ [ARGUMENT]
Node C: if (size > MAX_SIZE || size < 0)  ← Validation!
  ↓ [CFG]
Node D: new byte[size]
  ↓
Node E: stream.read(buffer, 0, size)  ← Limited read

→ VERDICT: SAFE
```

### **Features Model Học:**

```python
buffer_features = {
    'has_user_input': True/False,
    'has_array_allocation': True/False,
    'size_from_user': True/False,  # Size controlled by user
    'has_bounds_check': True/False,  # if (size < MAX)
    'has_array_access': True/False,  # buffer[index]
    'has_index_check': True/False,   # if (index < length)
    'has_unchecked_read': True/False,  # read() without limit
}

# Decision:
if (has_user_input and 
    has_array_allocation and 
    size_from_user and 
    NOT has_bounds_check):
    → BUFFER_OVERFLOW
```

---

## 🧠 Graph Neural Network Learning Process

### **Training Phase:**

```python
for epoch in range(50):
    for graph in training_data:
        
        # 1. Encode nodes
        node_embeddings = GraphCodeBERT(graph.nodes)
        
        # 2. GNN propagation
        for layer in gnn_layers:
            # Node học từ neighbors
            node_embeddings = layer(node_embeddings, graph.edges)
        
        # 3. Graph pooling
        graph_embedding = pool(node_embeddings)
        
        # 4. Classify
        prediction = classifier(graph_embedding)
        
        # 5. Compare with true label
        loss = CrossEntropy(prediction, true_label)
        
        # 6. Update weights
        loss.backward()
        optimizer.step()
        
        # Model học patterns từ 250+ examples!
```

### **Sau Training, Model Học Được:**

```python
learned_knowledge = {
    'SQL_Injection': {
        'source_patterns': ['getParameter', 'getHeader', 'readLine'],
        'sink_patterns': ['executeQuery', 'executeUpdate'],
        'dangerous_operations': ['String.concat(+)', 'String.format'],
        'safe_patterns': ['PreparedStatement', 'setString'],
        'typical_path_length': 2-4 hops,
        'confidence_threshold': 0.85,
    },
    
    'Command_Injection': {
        'source_patterns': ['getParameter', 'System.getProperty'],
        'sink_patterns': ['Runtime.exec', 'ProcessBuilder'],
        'critical_indicators': ['/bin/sh', 'cmd.exe'],
        'safe_patterns': ['whitelist', 'regex validation'],
        'typical_path_length': 2-3 hops,
        'confidence_threshold': 0.90,
    },
    
    # ... similar cho Path Traversal, Buffer Overflow
}
```

---

## 🎯 Inference Process (Predict Code Mới)

### **Ví Dụ: Unknown Code**

```java
public class UnknownCode {
    public void process(HttpServletRequest req) {
        String sql = req.getParameter("query");
        Statement stmt = conn.createStatement();
        stmt.execute(sql);  // ← Có lỗi không?
    }
}
```

### **Step-by-Step Analysis:**

```python
# Step 1: Parse thành graph
graph = parse_joern_json(code)

# Graph structure:
# Node 1: req.getParameter("query")
# Node 2: sql (IDENTIFIER)
# Node 3: stmt.createStatement()
# Node 4: stmt.execute(sql)
# Edges: 1→2 (REACHING_DEF), 2→4 (ARGUMENT)

# Step 2: Encode nodes
node_1_emb = GraphCodeBERT("req.getParameter(\"query\")")  # [768 dims]
node_2_emb = GraphCodeBERT("sql")
node_3_emb = GraphCodeBERT("stmt.createStatement()")
node_4_emb = GraphCodeBERT("stmt.execute(sql)")

# Step 3: GNN propagation
# Layer 1:
node_2_updated = node_2_emb + attention(node_2_emb, node_1_emb)
# Node 2 học: "Tôi chứa data từ getParameter"

node_4_updated = node_4_emb + attention(node_4_emb, node_2_updated)
# Node 4 học: "Tôi nhận SQL string từ user input"

# Layer 2:
node_4_final = node_4_updated + attention_from_neighbors()
# Node 4 học: "Không có PreparedStatement, không có validation"

# Step 4: Graph pooling
graph_emb = mean_pool([node_1_final, node_2_final, node_3_final, node_4_final])

# Step 5: Classify
logits = classifier(graph_emb)
# Output: [0.02, 0.01, 0.03, 0.01, 0.93]
#         Safe  Buf   Cmd   Path  SQL
#                               ↑
#                          93% SQL Injection!

# Step 6: Explain
explanation = {
    'prediction': 'SQL_Injection',
    'confidence': 0.93,
    'reasoning': [
        'Detected user input: getParameter("query")',
        'Detected SQL execution: stmt.execute()',
        'Data flow: getParameter → sql → execute',
        'Missing: PreparedStatement or parameterization',
        'Missing: Input validation',
    ],
    'recommendation': 'Use PreparedStatement with parameterized queries'
}
```

---

## 📊 Accuracy by Vulnerability Type

### **Expected Performance:**

| Vulnerability Type | Precision | Recall | F1 Score | Reasoning |
|-------------------|-----------|--------|----------|-----------|
| **SQL Injection** | 0.91 | 0.89 | 0.90 | Clear patterns, easy to detect |
| **Command Injection** | 0.93 | 0.91 | 0.92 | Very distinctive sinks |
| **Path Traversal** | 0.87 | 0.85 | 0.86 | More variations in patterns |
| **Buffer Overflow** | 0.82 | 0.80 | 0.81 | Complex, need deeper analysis |
| **Safe Code** | 0.95 | 0.96 | 0.95 | Majority class, well-represented |

### **Tại Sao Command Injection Cao Nhất?**

1. Sinks rất distinctive: `Runtime.exec()`, `ProcessBuilder`
2. Ít variations
3. Clear data flow patterns
4. Easy to spot missing validation

### **Tại Sao Buffer Overflow Thấp Nhất?**

1. Java tự động handle nhiều cases
2. Patterns phức tạp hơn
3. Ít samples trong dataset
4. Cần deeper multi-hop reasoning

---

## 🔍 Key Takeaways

### **Model Nhận Diện Qua:**

1. ✅ **Source-Sink Pairs**: User input → Dangerous function
2. ✅ **Data Flow Analysis**: Trace data qua REACHING_DEF edges
3. ✅ **Validation Detection**: Có node validation hay không?
4. ✅ **Pattern Matching**: So sánh với learned patterns
5. ✅ **Multi-hop Reasoning**: Understand complex flows

### **Không Phải Regex/Keywords:**

❌ Không chỉ tìm keyword "executeQuery"
✅ Phải hiểu WHERE data comes from và WHERE it goes

### **Graph > Text:**

**Text-based (BERT):** Chỉ thấy code dưới dạng sequence
**Graph-based (GNN):** Hiểu structure, flow, relationships

→ **GNN chính xác hơn 10-15% so với text-based methods!**
