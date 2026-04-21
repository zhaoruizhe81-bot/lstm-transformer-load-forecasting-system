# LSTM-Transformer混合模型实现
import torch
import torch.nn as nn
import numpy as np
import math

class HybridModel(nn.Module):
    """LSTM-Transformer混合模型"""
    
    def __init__(self, input_size=1, lstm_hidden=64, transformer_layers=2, nhead=4, dropout=0.15, output_size=1):
        super(HybridModel, self).__init__()
        
        # LSTM部分
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True
        )
        
        # Transformer部分
        self.d_model = lstm_hidden
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=self.d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, transformer_layers)
        
        # 输出层
        self.fc = nn.Linear(lstm_hidden, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        # LSTM提取时序特征
        lstm_out, _ = self.lstm(x)
        
        # Transformer增强特征
        transformer_out = self.transformer(lstm_out)
        
        # 取最后一个时间步
        out = self.dropout(transformer_out[:, -1, :])
        out = self.fc(out)
        return out

class HybridTrainer:
    """混合模型训练器"""
    
    def __init__(self, model, learning_rate=0.0005, device='cpu'):
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
    
    def train(self, train_loader, valid_loader, epochs=120):
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

