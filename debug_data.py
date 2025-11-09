"""
Debug script - Kiểm tra tại sao không load được data
Chạy script này trong Colab để debug
"""

import os
from pathlib import Path

def check_data_structure(root_dir='/content/DoAn_2'):
    """Kiểm tra cấu trúc thư mục"""
    
    root = Path(root_dir)
    
    print("=" * 70)
    print("DATA STRUCTURE CHECK")
    print("=" * 70)
    
    print(f"\n1. Root directory: {root}")
    print(f"   Exists: {root.exists()}")
    
    if not root.exists():
        print("   ❌ Root directory not found!")
        return
    
    # Check output directories
    print("\n2. Output directories:")
    
    output_dir = root / 'output'
    output_safe_dir = root / 'output_safe'
    
    print(f"   output/ : {output_dir.exists()}")
    print(f"   output_safe/ : {output_safe_dir.exists()}")
    
    # Check subdirectories
    vuln_types = ['Buffer_Overflow', 'Command_Injection', 'Path_Traversal', 'SQL_Injection']
    
    print("\n3. Vulnerable files:")
    total_vuln = 0
    for vuln_type in vuln_types:
        type_dir = output_dir / vuln_type
        if type_dir.exists():
            json_files = list(type_dir.glob('*.json'))
            # Exclude prediction results
            json_files = [f for f in json_files if 'prediction' not in f.name]
            count = len(json_files)
            total_vuln += count
            print(f"   {vuln_type:20s}: {count:4d} files")
            
            if count > 0:
                print(f"      Example: {json_files[0].name}")
        else:
            print(f"   {vuln_type:20s}: Directory not found ❌")
    
    print(f"\n   Total vulnerable: {total_vuln} files")
    
    print("\n4. Safe files:")
    total_safe = 0
    for vuln_type in vuln_types:
        type_dir = output_safe_dir / vuln_type
        if type_dir.exists():
            json_files = list(type_dir.glob('*.json'))
            count = len(json_files)
            total_safe += count
            print(f"   {vuln_type:20s}: {count:4d} files")
            
            if count > 0:
                print(f"      Example: {json_files[0].name}")
        else:
            print(f"   {vuln_type:20s}: Directory not found ❌")
    
    print(f"\n   Total safe: {total_safe} files")
    
    print("\n" + "=" * 70)
    print(f"TOTAL: {total_vuln + total_safe} files")
    print("=" * 70)
    
    # Check sample file
    if total_vuln > 0 or total_safe > 0:
        print("\n5. Sample file content:")
        
        # Find first JSON file
        for vuln_type in vuln_types:
            type_dir = output_dir / vuln_type
            if type_dir.exists():
                json_files = [f for f in type_dir.glob('*.json') if 'prediction' not in f.name]
                if json_files:
                    sample_file = json_files[0]
                    print(f"   File: {sample_file.name}")
                    
                    import json
                    with open(sample_file, 'r') as f:
                        data = json.load(f)
                    
                    print(f"   Keys: {list(data.keys())}")
                    
                    if 'functions' in data:
                        print(f"   Number of functions: {len(data['functions'])}")
                        if len(data['functions']) > 0:
                            func = data['functions'][0]
                            print(f"   Function: {func.get('function', 'N/A')}")
                            print(f"   AST nodes: {len(func.get('AST', []))}")
                    
                    break
    else:
        print("\n⚠️ NO JSON FILES FOUND!")
        print("\nPossible issues:")
        print("1. Files not in repository")
        print("2. Wrong directory structure")
        print("3. Need to generate JSON files from Joern first")
        
        print("\nExpected structure:")
        print("DoAn_2/")
        print("├── output/")
        print("│   ├── Buffer_Overflow/*.json")
        print("│   ├── Command_Injection/*.json")
        print("│   ├── Path_Traversal/*.json")
        print("│   └── SQL_Injection/*.json")
        print("└── output_safe/")
        print("    ├── Buffer_Overflow/*.json")
        print("    ├── Command_Injection/*.json")
        print("    ├── Path_Traversal/*.json")
        print("    └── SQL_Injection/*.json")


if __name__ == '__main__':
    import sys
    
    # Auto detect environment
    if os.path.exists('/content'):
        root_dir = '/content/DoAn_2'
    else:
        root_dir = input("Enter root directory path: ")
    
    check_data_structure(root_dir)
