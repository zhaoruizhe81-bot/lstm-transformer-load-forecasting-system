# LSTM-Transformer混合模型实现
import copy
import torch
import torch.nn as nn

class HybridModel(nn.Module):
    """LSTM-Transformer混合模型"""

    def __init__(
        self,
        input_size=1,
        lstm_hidden=64,
        transformer_layers=2,
        nhead=4,
        dropout=0.15,
        output_size=1,
        lstm_layers=1,
        feedforward_multiplier=4
    ):
        super(HybridModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True,
        )

        self.d_model = lstm_hidden
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=self.d_model * feedforward_multiplier,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, transformer_layers)

        self.lstm_norm = nn.LayerNorm(lstm_hidden)
        self.transformer_norm = nn.LayerNorm(lstm_hidden)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Sequential(
            nn.Linear(lstm_hidden * 2 + input_size, lstm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, max(16, lstm_hidden // 2)),
            nn.GELU(),
            nn.Linear(max(16, lstm_hidden // 2), output_size)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        lstm_features = self.lstm_norm(lstm_out)

        transformer_out = self.transformer(lstm_features)
        transformer_features = self.transformer_norm(transformer_out)

        # 同时保留LSTM短期记忆、Transformer上下文增强特征和最近观测值。
        features = torch.cat(
            [
                lstm_features[:, -1, :],
                transformer_features[:, -1, :],
                x[:, -1, :]
            ],
            dim=1
        )
        return self.regressor(self.dropout(features))

class HybridTrainer:
    """混合模型训练器"""
    
    def __init__(self, model, learning_rate=0.0005, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.SmoothL1Loss(beta=0.02)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=8,
            min_lr=1e-6
        )
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
        best_loss = float('inf')
        best_state = None
        best_epoch = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            valid_loss = self.validate(valid_loader)
            self.scheduler.step(valid_loss)
            
            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch + 1
                best_state = copy.deepcopy(self.model.state_dict())
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Valid Loss: {valid_loss:.6f}')

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f'Best Hybrid checkpoint: epoch {best_epoch}, Valid Loss: {best_loss:.6f}')

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
