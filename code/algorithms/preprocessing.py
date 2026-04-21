# 数据预处理算法
import numpy as np
import pandas as pd
from scipy import interpolate
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import json

class DataPreprocessor:
    """数据预处理类"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
    
    def detect_outliers_zscore(self, data, threshold=3):
        """Z-score异常检测"""
        mean = np.mean(data)
        std = np.std(data)
        z_scores = np.abs((data - mean) / std)
        outliers = np.where(z_scores > threshold)[0]
        return outliers.tolist(), z_scores.tolist()
    
    def detect_outliers_iqr(self, data):
        """IQR异常检测"""
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = np.where((data < lower_bound) | (data > upper_bound))[0]
        return outliers.tolist(), {'q1': float(q1), 'q3': float(q3), 'iqr': float(iqr)}
    
    def detect_outliers_boxplot(self, data):
        """箱型图异常检测（与IQR相同）"""
        return self.detect_outliers_iqr(data)
    
    def fill_missing_linear(self, data, missing_indices):
        """线性插值填充缺失值"""
        df = pd.Series(data)
        df.iloc[missing_indices] = np.nan
        df_filled = df.interpolate(method='linear')
        return df_filled.tolist()
    
    def fill_missing_spline(self, data, missing_indices, order=3):
        """样条插值填充缺失值"""
        df = pd.Series(data)
        valid_indices = [i for i in range(len(data)) if i not in missing_indices]
        valid_data = [data[i] for i in valid_indices]
        
        if len(valid_data) < order + 1:
            # 数据点不足，使用线性插值
            return self.fill_missing_linear(data, missing_indices)
        
        # 样条插值
        spline = interpolate.UnivariateSpline(valid_indices, valid_data, k=order, s=0)
        filled_data = data.copy()
        for idx in missing_indices:
            filled_data[idx] = float(spline(idx))
        return filled_data
    
    def normalize_minmax(self, data, feature_range=(0, 1)):
        """Min-Max归一化"""
        data_array = np.array(data).reshape(-1, 1)
        self.scaler = MinMaxScaler(feature_range=feature_range)
        normalized = self.scaler.fit_transform(data_array)
        return normalized.flatten().tolist()
    
    def denormalize_minmax(self, normalized_data):
        """反归一化"""
        data_array = np.array(normalized_data).reshape(-1, 1)
        denormalized = self.scaler.inverse_transform(data_array)
        return denormalized.flatten().tolist()
    
    def correlation_analysis(self, df, target_col='loadvalue'):
        """相关性分析"""
        corr_matrix = df.corr()
        target_corr = corr_matrix[target_col].to_dict()
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'target_correlation': target_corr
        }
    
    def pca_analysis(self, data, n_components=2):
        """PCA主成分分析"""
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(data)
        return {
            'transformed_data': transformed.tolist(),
            'explained_variance': pca.explained_variance_ratio_.tolist(),
            'components': pca.components_.tolist()
        }
    
    def create_sequences(self, data, seq_length=24):
        """创建时间序列数据集"""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)
    
    def split_train_test(self, X, y, train_ratio=0.8):
        """划分训练集和测试集"""
        split_idx = int(len(X) * train_ratio)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        return X_train, X_test, y_train, y_test

def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # MAE: 平均绝对误差
    mae = np.mean(np.abs(y_true - y_pred))
    
    # RMSE: 均方根误差
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # MAPE: 平均绝对百分比误差
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # R2: 决定系数
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape),
        'r2score': float(r2)
    }

