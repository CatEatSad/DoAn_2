# 🚀 Implementation Plan - Nhận Diện Vulnerability Từ JSON

## 📋 Pipeline Tổng Quan

```
File JSON → Parse Graph → Encode Nodes → GNN → Classify → Output (Safe/SQL/Buffer/...)
```

## 🎯 Cách Nhận Diện Từng Loại Lỗi

### **1. SQL Injection - Nhận Diện Qua:**

#### Signature Patterns:
```python
SQL_INJECTION_SIGNATURES = {
    'sources': [
        'getParameter',
        'getHeader', 
        'getCookie',
        'readLine',
        'request.get*'
    ],
    'sinks': [
        'executeQuery',
        'executeUpdate',
        'execute',
        'prepareStatement',
        'createStatement'
    ],
    'dangerous_operations': [
        'String concatenation (+)',
        'String.format',
        'StringBuilder.append'
    ],
    'safe_patterns': [
        'PreparedStatement',
        'setString',
        'setInt',
        'parameterized query'
    ]
}
```

#### Graph Features:
```python
def extract_sql_features(graph):
    features = {
        'has_user_input': False,      # Có node getParameter?
        'has_sql_execution': False,   # Có node executeQuery?
        'has_string_concat': False,   # Có + hoặc concat?
        'has_prepared_stmt': False,   # Có PreparedStatement?
        'data_flow_exists': False     # User input → SQL sink?
    }
    
    for node in graph['nodes']:
        # Check source
        if 'getParameter' in node['properties']['CODE']:
            features['has_user_input'] = True
        
        # Check sink
        if 'executeQuery' in node['properties']['CODE']:
            features['has_sql_execution'] = True
        
        # Check concatenation
        if node['label'] == 'CALL' and 'ADDITION' in node['properties'].get('NAME', ''):
            features['has_string_concat'] = True
        
        # Check safe practice
        if 'PreparedStatement' in node['properties']['CODE']:
            features['has_prepared_stmt'] = True
    
    # Check data flow
    for edge in graph['edges']:
        if edge['edgeType'] == 'REACHING_DEF':
            # Trace from source to sink
            features['data_flow_exists'] = True
    
    return features
```

#### Decision Logic:
```python
if (has_user_input and 
    has_sql_execution and 
    data_flow_exists and 
    NOT has_prepared_stmt):
    → SQL INJECTION!
else:
    → SAFE
```

---

### **2. Command Injection - Nhận Diện Qua:**

#### Signature Patterns:
```python
COMMAND_INJECTION_SIGNATURES = {
    'sources': [
        'getParameter',
        'readLine',
        'System.getProperty',
        'System.getenv'
    ],
    'sinks': [
        'Runtime.exec',
        'Runtime.getRuntime().exec',
        'ProcessBuilder',
        'ProcessBuilder.start'
    ],
    'safe_patterns': [
        'whitelist validation',
        'regex check: ^[a-zA-Z0-9]+$',
        'ProcessBuilder with array'  # Not string concatenation
    ]
}
```

#### Graph Features:
```python
def extract_command_features(graph):
    features = {
        'has_user_input': False,
        'has_runtime_exec': False,
        'has_process_builder': False,
        'has_validation': False,
        'uses_shell': False  # Nguy hiểm hơn
    }
    
    for node in graph['nodes']:
        code = node['properties'].get('CODE', '')
        
        # Source
        if any(src in code for src in ['getParameter', 'readLine']):
            features['has_user_input'] = True
        
        # Sinks
        if 'Runtime.exec' in code or 'exec(' in code:
            features['has_runtime_exec'] = True
        
        if 'ProcessBuilder' in code:
            features['has_process_builder'] = True
        
        # Check for shell usage
        if '/bin/sh' in code or 'cmd.exe' in code:
            features['uses_shell'] = True
        
        # Validation
        if any(val in code for val in ['matches(', 'Pattern.', 'validate']):
            features['has_validation'] = True
    
    return features
```

---

### **3. Path Traversal - Nhận Diện Qua:**

#### Signature Patterns:
```python
PATH_TRAVERSAL_SIGNATURES = {
    'sources': [
        'getParameter',
        'getHeader',
        'request.*'
    ],
    'sinks': [
        'new File(',
        'FileInputStream',
        'FileOutputStream',
        'FileReader',
        'FileWriter',
        'Files.read',
        'Files.write'
    ],
    'dangerous_patterns': [
        'String concatenation for path',
        'No path normalization'
    ],
    'safe_patterns': [
        'Path.normalize',
        'getCanonicalPath',
        'startsWith(basePath)',
        'whitelist check'
    ]
}
```

#### Graph Features:
```python
def extract_path_traversal_features(graph):
    features = {
        'has_user_input': False,
        'has_file_operation': False,
        'has_path_concat': False,
        'has_path_validation': False,
        'has_canonical_check': False
    }
    
    for node in graph['nodes']:
        code = node['properties'].get('CODE', '')
        
        if 'getParameter' in code:
            features['has_user_input'] = True
        
        if any(file_op in code for file_op in ['File(', 'FileInputStream', 'FileReader']):
            features['has_file_operation'] = True
        
        # Path concatenation
        if 'new File(' in code and '+' in code:
            features['has_path_concat'] = True
        
        # Validation
        if 'contains("..")' in code or 'validate' in code:
            features['has_path_validation'] = True
        
        if 'getCanonicalPath' in code or 'normalize' in code:
            features['has_canonical_check'] = True
    
    return features
```

---

### **4. Buffer Overflow - Nhận Diện Qua:**

#### Signature Patterns (Java specific):
```python
BUFFER_OVERFLOW_SIGNATURES = {
    'dangerous_operations': [
        'byte[] without size check',
        'array access without bounds check',
        'BufferedReader.read without limit',
        'InputStream.read in loop',
        'System.arraycopy without validation'
    ],
    'safe_patterns': [
        'array.length check',
        'if (index < array.length)',
        'ByteBuffer with limit',
        'read with max size parameter'
    ]
}
```

#### Graph Features:
```python
def extract_buffer_overflow_features(graph):
    features = {
        'has_array_operation': False,
        'has_bounds_check': False,
        'has_unchecked_read': False,
        'has_user_controlled_size': False
    }
    
    for node in graph['nodes']:
        code = node['properties'].get('CODE', '')
        label = node['label']
        
        # Array operations
        if '[' in code and ']' in code:
            features['has_array_operation'] = True
        
        # Bounds check
        if '.length' in code or 'size()' in code:
            features['has_bounds_check'] = True
        
        # Unchecked read
        if 'read(' in code and label == 'CALL':
            features['has_unchecked_read'] = True
        
        # User-controlled size
        if 'getParameter' in code:
            # Check if this is used for array size
            features['has_user_controlled_size'] = True
    
    return features
```

---

## 💻 Complete Implementation

### **File Structure:**
```
renew/
├── src/
│   ├── data_loader.py       # Parse JSON files
│   ├── model.py             # GNN model
│   ├── features.py          # Feature extraction
│   ├── train.py             # Training script
│   └── predict.py           # Inference script
├── output/                  # Vulnerable JSON files
├── output_safe/             # Safe JSON files
└── requirements.txt
```

Bạn muốn tôi tạo code implementation chi tiết không?
