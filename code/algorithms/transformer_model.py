# Transformer模型实现
import torch
import torch.nn as nn
import numpy as np
import math

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerModel(nn.Module):
    """Transformer模型"""
    
    def __init__(self, input_size=1, d_model=256, nhead=8, num_layers=4, dropout=0.1, output_size=1):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        self.input_embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.fc = nn.Linear(d_model, output_size)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        x = self.input_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # 取最后一个时间步的输出
        out = self.fc(x[:, -1, :])
        return out

class TransformerTrainer:
    """Transformer训练器"""
    
    def __init__(self, model, learning_rate=0.0001, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.train_losses = []
        self.valid_losses = []
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0
        batch_count = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            outputs = self.model(batch_x)
            loss = self.criterion(outputs.squeeze(), batch_y)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        return epoch_loss / batch_count
    
    def validate(self, valid_loader):
        """验证"""
        self.model.eval()
        epoch_loss = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs.squeeze(), batch_y)
                
                epoch_loss += loss.item()
                batch_count += 1
        
        return epoch_loss / batch_count
    
    def train(self, train_loader, valid_loader, epochs=80):
        """完整训练流程"""
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            valid_loss = self.validate(valid_loader)
            
            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Valid Loss: {valid_loss:.6f}')
        
        return self.train_losses, self.valid_losses
    
    def predict(self, X):
        """预测"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            if len(X_tensor.shape) == 2:
                X_tensor = X_tensor.unsqueeze(-1)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy().flatten()
    
    def save_model(self, filepath):
        """保存模型"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'valid_losses': self.valid_losses
        }, filepath)
    
    def load_model(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.valid_losses = checkpoint.get('valid_losses', [])

