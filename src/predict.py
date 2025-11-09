"""
Prediction Script - Dùng trained model để predict code mới
"""

import torch
import json
from pathlib import Path
from model import create_model


class VulnerabilityPredictor:
    """Class để predict vulnerability từ JSON file"""
    
    LABEL_NAMES = {
        0: 'Safe',
        1: 'Buffer_Overflow',
        2: 'Command_Injection',
        3: 'Path_Traversal',
        4: 'SQL_Injection'
    }
    
    def __init__(self, model_path, device='cuda'):
        """
        Args:
            model_path: Path to saved model (.pth file)
            device: 'cuda' or 'cpu'
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Load model
        self.model = create_model('simplified', num_classes=5, device=self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Best validation accuracy: {checkpoint.get('best_val_acc', 'N/A')}")
    
    def predict_file(self, json_path):
        """
        Predict vulnerability cho 1 file JSON
        
        Args:
            json_path: Path to JSON file
        
        Returns:
            dict with prediction results
        """
        from data_loader import VulnerabilityDataset
        
        # Parse JSON (simplified - you'd use proper data loader)
        # For now, return dummy prediction
        
        # TODO: Implement proper prediction
        prediction = {
            'file': str(json_path),
            'prediction': 'Command_Injection',
            'confidence': 0.89,
            'all_probabilities': {
                'Safe': 0.05,
                'Buffer_Overflow': 0.02,
                'Command_Injection': 0.89,
                'Path_Traversal': 0.03,
                'SQL_Injection': 0.01
            }
        }
        
        return prediction
    
    def predict_batch(self, json_files):
        """Predict cho nhiều files"""
        results = []
        for json_file in json_files:
            result = self.predict_file(json_file)
            results.append(result)
        return results
    
    def analyze_code_patterns(self, json_path):
        """
        Phân tích chi tiết patterns trong code
        Giải thích TẠI SAO code bị lỗi
        """
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        analysis = {
            'file': str(json_path),
            'patterns_detected': [],
            'risk_factors': []
        }
        
        # Analyze AST nodes
        if 'functions' in data and len(data['functions']) > 0:
            ast_nodes = data['functions'][0].get('AST', [])
            
            # Detect patterns
            for node in ast_nodes:
                code = node.get('properties', {}).get('CODE', '')
                label = node.get('label', '')
                
                # SQL Injection patterns
                if 'executeQuery' in code or 'executeUpdate' in code:
                    analysis['patterns_detected'].append({
                        'type': 'SQL_EXECUTION',
                        'code': code,
                        'line': node.get('properties', {}).get('LINE_NUMBER', 'unknown')
                    })
                
                if 'getParameter' in code:
                    analysis['patterns_detected'].append({
                        'type': 'USER_INPUT',
                        'code': code,
                        'line': node.get('properties', {}).get('LINE_NUMBER', 'unknown')
                    })
                
                # Command Injection patterns
                if 'Runtime.exec' in code or 'ProcessBuilder' in code:
                    analysis['patterns_detected'].append({
                        'type': 'COMMAND_EXECUTION',
                        'code': code,
                        'line': node.get('properties', {}).get('LINE_NUMBER', 'unknown')
                    })
                
                # Path Traversal patterns
                if 'File(' in code or 'FileInputStream' in code:
                    analysis['patterns_detected'].append({
                        'type': 'FILE_OPERATION',
                        'code': code,
                        'line': node.get('properties', {}).get('LINE_NUMBER', 'unknown')
                    })
                
                # Check for validation
                if any(val in code for val in ['validate', 'sanitize', 'matches(', 'PreparedStatement']):
                    analysis['patterns_detected'].append({
                        'type': 'VALIDATION',
                        'code': code,
                        'line': node.get('properties', {}).get('LINE_NUMBER', 'unknown')
                    })
            
            # Analyze data flow
            user_inputs = [p for p in analysis['patterns_detected'] if p['type'] == 'USER_INPUT']
            sql_execs = [p for p in analysis['patterns_detected'] if p['type'] == 'SQL_EXECUTION']
            cmd_execs = [p for p in analysis['patterns_detected'] if p['type'] == 'COMMAND_EXECUTION']
            file_ops = [p for p in analysis['patterns_detected'] if p['type'] == 'FILE_OPERATION']
            validations = [p for p in analysis['patterns_detected'] if p['type'] == 'VALIDATION']
            
            # Risk assessment
            if user_inputs and sql_execs and not validations:
                analysis['risk_factors'].append({
                    'severity': 'HIGH',
                    'type': 'SQL_INJECTION',
                    'reason': 'User input flows to SQL execution without validation'
                })
            
            if user_inputs and cmd_execs and not validations:
                analysis['risk_factors'].append({
                    'severity': 'CRITICAL',
                    'type': 'COMMAND_INJECTION',
                    'reason': 'User input flows to command execution without validation'
                })
            
            if user_inputs and file_ops and not validations:
                analysis['risk_factors'].append({
                    'severity': 'HIGH',
                    'type': 'PATH_TRAVERSAL',
                    'reason': 'User input used in file operations without validation'
                })
        
        return analysis


def main():
    """Example usage"""
    
    ROOT_DIR = Path(r"c:\Users\abcdx\OneDrive\Máy tính\renew")
    MODEL_PATH = ROOT_DIR / 'saved_models' / 'best_model.pth'
    
    # Check if model exists
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please train the model first using train.py")
        return
    
    # Create predictor
    predictor = VulnerabilityPredictor(MODEL_PATH)
    
    # Example: Predict một file
    example_file = ROOT_DIR / 'output' / 'Command_Injection' / 'Command_Injection_0001_vul.json'
    
    if example_file.exists():
        print(f"\n{'='*70}")
        print(f"Analyzing: {example_file.name}")
        print(f"{'='*70}\n")
        
        # Analyze patterns
        analysis = predictor.analyze_code_patterns(example_file)
        
        print("Patterns Detected:")
        print(f"{'─'*70}")
        for pattern in analysis['patterns_detected']:
            print(f"  [{pattern['type']:20s}] Line {pattern['line']:5s}: {pattern['code'][:50]}")
        
        print(f"\n{'='*70}")
        print("Risk Assessment:")
        print(f"{'='*70}")
        for risk in analysis['risk_factors']:
            print(f"\n  ⚠️  {risk['severity']} - {risk['type']}")
            print(f"      Reason: {risk['reason']}")
        
        print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
