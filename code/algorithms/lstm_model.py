# LSTM模型实现
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import os

class LSTMModel(nn.Module):
    """LSTM模型"""
    
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        out = self.fc(lstm_out[:, -1, :])
        return out

class LSTMTrainer:
    """LSTM训练器"""
    
    def __init__(self, model, learning_rate=0.001, device='cpu'):
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
            
            # 前向传播
            outputs = self.model(batch_x)
            loss = self.criterion(outputs.squeeze(), batch_y)
            
            # 反向传播
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
    
    def train(self, train_loader, valid_loader, epochs=100):
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

def create_dataloader(X, y, batch_size=32, shuffle=True):
    """创建数据加载器"""
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    if len(X_tensor.shape) == 2:
        X_tensor = X_tensor.unsqueeze(-1)
    
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

