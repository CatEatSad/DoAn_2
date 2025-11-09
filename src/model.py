"""
GNN Model - Graph Neural Network cho Vulnerability Detection
Sử dụng GraphCodeBERT + GAT layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from transformers import RobertaModel, RobertaTokenizer


class VulnerabilityGNN(nn.Module):
    """
    GNN Model cho phát hiện lỗi bảo mật
    
    Architecture:
        1. GraphCodeBERT encoder (encode node code)
        2. GAT layers (propagate information)
        3. Graph pooling (aggregate to graph-level)
        4. Classifier (predict vulnerability type)
    """
    
    def __init__(self, 
                 num_classes=5,
                 hidden_dim=256,
                 num_gat_layers=2,
                 num_heads=4,
                 dropout=0.3):
        super().__init__()
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # 1. GraphCodeBERT encoder
        print("Loading GraphCodeBERT...")
        self.code_encoder = RobertaModel.from_pretrained('microsoft/graphcodebert-base')
        self.tokenizer = RobertaTokenizer.from_pretrained('microsoft/graphcodebert-base')
        
        # Freeze GraphCodeBERT (option: unfreeze later for fine-tuning)
        for param in self.code_encoder.parameters():
            param.requires_grad = False
        
        # 2. GAT layers
        self.gat_layers = nn.ModuleList()
        
        # First GAT layer
        self.gat_layers.append(
            GATConv(
                in_channels=768,  # GraphCodeBERT output dim
                out_channels=hidden_dim,
                heads=num_heads,
                dropout=dropout,
                concat=True
            )
        )
        
        # Additional GAT layers
        for _ in range(num_gat_layers - 1):
            self.gat_layers.append(
                GATConv(
                    in_channels=hidden_dim * num_heads,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True
                )
            )
        
        # 3. Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 4. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_heads * 2, 512),  # *2 for mean + max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def encode_nodes(self, node_texts):
        """Encode node texts using GraphCodeBERT"""
        
        if len(node_texts) == 0:
            return torch.zeros((1, 768), device=self.code_encoder.device)
        
        # Tokenize
        encoded = self.tokenizer(
            node_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(self.code_encoder.device)
        
        # Encode with GraphCodeBERT
        with torch.no_grad():  # No grad for frozen encoder
            outputs = self.code_encoder(**encoded)
            # Take [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        
        return embeddings
    
    def forward(self, data):
        """
        Args:
            data: PyTorch Geometric Data object with:
                - x: Node features (num_nodes, 768)
                - edge_index: Edge connections (2, num_edges)
                - batch: Batch assignment for each node
        """
        
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Move to device
        x = x.to(next(self.parameters()).device)
        edge_index = edge_index.to(next(self.parameters()).device)
        batch = batch.to(next(self.parameters()).device)
        
        # GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            x = gat_layer(x, edge_index)
            x = F.elu(x)
            x = self.dropout(x)
        
        # Graph pooling (combine mean and max)
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        graph_embedding = torch.cat([mean_pool, max_pool], dim=1)
        
        # Classification
        logits = self.classifier(graph_embedding)
        
        return logits
    
    def predict(self, data):
        """Predict vulnerability type"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(data)
            probs = F.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
        
        return predictions, probs


class SimplifiedVulnerabilityGNN(nn.Module):
    """
    Simplified version - không dùng GraphCodeBERT (faster training)
    Chỉ dùng node features từ data_loader
    """
    
    def __init__(self, 
                 num_classes=5,
                 input_dim=768,
                 hidden_dim=256,
                 num_layers=2,
                 num_heads=4,
                 dropout=0.3):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(
                GATConv(
                    hidden_dim * num_heads if _ > 0 else hidden_dim,
                    hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True
                )
            )
        
        self.dropout = nn.Dropout(dropout)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_heads * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Project input
        x = self.input_proj(x)
        x = F.relu(x)
        
        # GAT layers
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index)
            x = F.elu(x)
            x = self.dropout(x)
        
        # Pooling
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        graph_embedding = torch.cat([mean_pool, max_pool], dim=1)
        
        # Classification
        logits = self.classifier(graph_embedding)
        
        return logits


def create_model(model_type='simplified', num_classes=5, device='cuda'):
    """Factory function to create model"""
    
    if model_type == 'full':
        model = VulnerabilityGNN(num_classes=num_classes)
    else:
        model = SimplifiedVulnerabilityGNN(num_classes=num_classes)
    
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*50}")
    print(f"Model: {model_type}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"{'='*50}\n")
    
    return model


if __name__ == '__main__':
    # Test model
    from torch_geometric.data import Data, Batch
    
    # Create dummy data
    num_nodes = 10
    num_edges = 15
    
    data1 = Data(
        x=torch.randn(num_nodes, 768),
        edge_index=torch.randint(0, num_nodes, (2, num_edges)),
        y=torch.tensor([2])  # Command Injection
    )
    
    data2 = Data(
        x=torch.randn(8, 768),
        edge_index=torch.randint(0, 8, (2, 12)),
        y=torch.tensor([0])  # Safe
    )
    
    # Create batch
    batch = Batch.from_data_list([data1, data2])
    
    # Test model
    model = create_model('simplified', device='cpu')
    logits = model(batch)
    
    print(f"Input batch: {batch}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Output logits: {logits}")
    
    predictions = torch.argmax(logits, dim=1)
    print(f"Predictions: {predictions}")
