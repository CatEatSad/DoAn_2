"""
Data Loader - Parse JSON files từ Joern output
Tạo dataset cho training GNN model
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from torch_geometric.data import Data, Dataset
from transformers import RobertaTokenizer


class VulnerabilityDataset(Dataset):
    """Dataset class cho vulnerability detection"""
    
    # Mapping vulnerability types to labels
    LABEL_MAP = {
        'Safe': 0,
        'Buffer_Overflow': 1,
        'Command_Injection': 2,
        'Path_Traversal': 3,
        'SQL_Injection': 4
    }
    
    def __init__(self, root_dir: str, split='train'):
        """
        Args:
            root_dir: Path to renew folder
            split: 'train', 'val', or 'test'
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.tokenizer = RobertaTokenizer.from_pretrained('microsoft/graphcodebert-base')
        
        # Load all JSON files
        self.data_list = self._load_all_files()
        
        print(f"Loaded {len(self.data_list)} samples for {split}")
        self._print_statistics()
    
    def _load_all_files(self) -> List[Dict]:
        """Load tất cả JSON files"""
        data_list = []
        
        # Load vulnerable files
        vuln_dir = self.root_dir / 'output'
        for vuln_type in ['Buffer_Overflow', 'Command_Injection', 'Path_Traversal', 'SQL_Injection']:
            type_dir = vuln_dir / vuln_type
            if type_dir.exists():
                for json_file in type_dir.glob('*.json'):
                    # Skip prediction results files
                    if 'prediction_results' in json_file.name:
                        continue
                    
                    data_list.append({
                        'path': json_file,
                        'label': self.LABEL_MAP[vuln_type],
                        'type': vuln_type
                    })
        
        # Load safe files
        safe_dir = self.root_dir / 'output_safe'
        for vuln_type in ['Buffer_Overflow', 'Command_Injection', 'Path_Traversal', 'SQL_Injection']:
            type_dir = safe_dir / vuln_type
            if type_dir.exists():
                for json_file in type_dir.glob('*.json'):
                    data_list.append({
                        'path': json_file,
                        'label': self.LABEL_MAP['Safe'],
                        'type': 'Safe'
                    })
        
        return data_list
    
    def _print_statistics(self):
        """Print dataset statistics"""
        label_counts = {}
        for item in self.data_list:
            label = item['type']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"\n{'='*50}")
        print(f"Dataset Statistics ({self.split}):")
        print(f"{'='*50}")
        for label, count in sorted(label_counts.items()):
            print(f"{label:20s}: {count:4d} files")
        print(f"{'='*50}\n")
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        """Get một sample"""
        item = self.data_list[idx]
        
        # Parse JSON file
        graph_data = self._parse_json(item['path'])
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=graph_data['node_features'],
            edge_index=graph_data['edge_index'],
            edge_attr=graph_data['edge_types'],
            y=torch.tensor([item['label']], dtype=torch.long),
            num_nodes=graph_data['num_nodes']
        )
        
        return data
    
    def _parse_json(self, json_path: Path) -> Dict:
        """Parse Joern JSON file và extract graph structure"""
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract nodes và edges từ AST
        nodes = []
        edges = []
        node_id_map = {}  # Map Joern ID to sequential index
        
        if 'functions' not in data or len(data['functions']) == 0:
            # Empty graph
            return self._create_empty_graph()
        
        ast_nodes = data['functions'][0].get('AST', [])
        
        # Bước 1: Collect all nodes
        for idx, node in enumerate(ast_nodes):
            node_id = node.get('id', str(idx))
            node_id_map[node_id] = idx
            
            # Extract node information
            node_info = {
                'id': node_id,
                'label': node.get('label', 'UNKNOWN'),
                'code': node.get('properties', {}).get('CODE', ''),
                'line_number': node.get('properties', {}).get('LINE_NUMBER', '0'),
                'name': node.get('properties', {}).get('NAME', ''),
                'method_name': node.get('properties', {}).get('METHOD_FULL_NAME', ''),
            }
            
            nodes.append(node_info)
        
        # Bước 2: Collect all edges
        edge_types_map = {
            'AST': 0,
            'CFG': 1,
            'REACHING_DEF': 2,
            'ARGUMENT': 3,
            'EVAL_TYPE': 4,
            'RECEIVER': 5,
            'CALL': 6
        }
        
        for node in ast_nodes:
            node_id = node.get('id')
            if node_id not in node_id_map:
                continue
            
            source_idx = node_id_map[node_id]
            
            for edge in node.get('edges', []):
                edge_type = edge.get('edgeType', 'UNKNOWN')
                target_id = edge.get('in', edge.get('out'))
                
                if target_id in node_id_map:
                    target_idx = node_id_map[target_id]
                    edge_type_idx = edge_types_map.get(edge_type, 7)  # 7 for UNKNOWN
                    
                    edges.append({
                        'source': source_idx,
                        'target': target_idx,
                        'type': edge_type_idx
                    })
        
        # Bước 3: Encode nodes using GraphCodeBERT
        node_features = self._encode_nodes(nodes)
        
        # Bước 4: Create edge_index tensor
        if len(edges) > 0:
            edge_index = torch.tensor(
                [[e['source'] for e in edges],
                 [e['target'] for e in edges]],
                dtype=torch.long
            )
            edge_types = torch.tensor([e['type'] for e in edges], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_types = torch.zeros(0, dtype=torch.long)
        
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_types': edge_types,
            'num_nodes': len(nodes)
        }
    
    def _encode_nodes(self, nodes: List[Dict]) -> torch.Tensor:
        """Encode nodes using GraphCodeBERT"""
        
        if len(nodes) == 0:
            return torch.zeros((1, 768))  # Empty embedding
        
        # Combine node information into text
        node_texts = []
        for node in nodes:
            # Tạo text representation của node
            text_parts = []
            
            # Add label
            if node['label']:
                text_parts.append(f"[{node['label']}]")
            
            # Add code (most important)
            if node['code']:
                text_parts.append(node['code'])
            
            # Add name
            if node['name']:
                text_parts.append(f"name:{node['name']}")
            
            # Add method name
            if node['method_name']:
                text_parts.append(f"method:{node['method_name']}")
            
            text = ' '.join(text_parts) if text_parts else '[EMPTY]'
            node_texts.append(text)
        
        # Tokenize
        encoded = self.tokenizer(
            node_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Simple encoding: average token embeddings
        # (In practice, you'd use GraphCodeBERT model here)
        # For now, create random features (will be replaced in model)
        node_features = torch.randn(len(nodes), 768)
        
        return node_features
    
    def _create_empty_graph(self) -> Dict:
        """Create empty graph for edge cases"""
        return {
            'node_features': torch.zeros((1, 768)),
            'edge_index': torch.zeros((2, 0), dtype=torch.long),
            'edge_types': torch.zeros(0, dtype=torch.long),
            'num_nodes': 1
        }


def create_dataloaders(root_dir: str, batch_size=8, train_ratio=0.7, val_ratio=0.15):
    """Create train/val/test dataloaders"""
    
    from torch_geometric.loader import DataLoader
    from sklearn.model_selection import train_test_split
    
    # Load all data
    dataset = VulnerabilityDataset(root_dir, split='all')
    
    # Split into train/val/test
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(
        indices, test_size=(1 - train_ratio), random_state=42
    )
    val_idx, test_idx = train_test_split(
        test_idx, test_size=(val_ratio / (1 - train_ratio)), random_state=42
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        dataset[train_idx],
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        dataset[val_idx],
        batch_size=batch_size,
        shuffle=False
    )
    
    test_loader = DataLoader(
        dataset[test_idx],
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test dataset loading
    root_dir = r"c:\Users\abcdx\OneDrive\Máy tính\renew"
    
    dataset = VulnerabilityDataset(root_dir)
    
    # Get first sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample data:")
        print(f"Node features shape: {sample.x.shape}")
        print(f"Edge index shape: {sample.edge_index.shape}")
        print(f"Label: {sample.y.item()}")
