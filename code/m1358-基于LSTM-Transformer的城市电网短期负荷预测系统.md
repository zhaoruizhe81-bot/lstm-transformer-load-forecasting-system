# 基于 LSTM-Transformer 的城市电网短期负荷预测系统

## 一、项目概述

本系统是一套面向城市电网调度场景的**短期电力负荷预测平台**，以深度学习模型为核心预测引擎，实现了从原始数据导入、数据预处理、模型训练到负荷预测的完整流程。系统采用 LSTM、Transformer 以及 **LSTM-Transformer 混合模型**三种架构，其中混合模型融合了 LSTM 对局部时序规律的提取能力与 Transformer 对长距离依赖的建模能力，能够有效提升短期负荷预测精度。

**技术栈**：

| 层次 | 技术 |
|------|------|
| 前端界面 | Streamlit |
| 后端服务 | Flask + Flask-SQLAlchemy |
| 深度学习框架 | PyTorch（LSTM / Transformer / 混合模型） |
| 数据库 | MySQL 8.0（PyMySQL 驱动） |
| 数据处理 | Pandas、NumPy、Scikit-learn |
| 可视化 | Matplotlib、Seaborn |

---

## 二、数据来源与数据分析

### 2.1 数据来源

本项目使用 **ETT-small（Electricity Transformer Temperature）** 数据集中的 `ETTh1.csv` 文件。ETT 是电力变压器油温数据集，由清华大学公开发布，广泛用于时间序列预测研究。

- **文件路径**：`ETT-small/ETTh1.csv`
- **数据规模**：17,420 条记录
- **采样频率**：逐小时（hourly）
- **时间跨度**：2016-07-01 00:00:00 ~ 2018-06-26 19:00:00

**原始数据字段**：

| 列名 | 数据类型 | 含义说明 |
|------|---------|---------|
| `date` | datetime | 记录时间戳（逐小时） |
| `HUFL` | float | 高压有用功率负荷（High UseFul Load） |
| `HULL` | float | 高压无用功率负荷（High UseLess Load） |
| `MUFL` | float | 中压有用功率负荷（Middle UseFul Load） |
| `MULL` | float | 中压无用功率负荷（Middle UseLess Load） |
| `LUFL` | float | 低压有用功率负荷（Low UseFul Load） |
| `LULL` | float | 低压无用功率负荷（Low UseLess Load） |
| `OT` | float | 油温（Oil Temperature）—— **预测目标** |

### 2.2 数据分析

原始数据的基本统计特征：

- **OT（油温/负荷目标）**：范围约 -4.08 ~ 46.01，呈现明显的季节性波动和日内周期性
- **HUFL/HULL**：高压负荷特征，与油温存在较强正相关
- **MUFL/MULL**：中压负荷特征，波动较小
- **LUFL/LULL**：低压负荷特征

数据导入系统后，OT 值经线性缩放映射到 **3000~10000 MW** 的合理电网负荷范围，以模拟真实城市电网的负荷量级：

$$loadvalue = \frac{OT - OT_{min}}{OT_{max} - OT_{min}} \times 7000 + 3000$$

同时，利用其他特征列模拟气象数据：
- **温度**：取 `HUFL` 与 `HULL` 的均值，线性映射到 -15℃~45℃
- **湿度**：取 `MUFL` 与 `MULL` 的均值，线性映射到 20%~95%
- **星期**：根据日期自动计算（1=周一 ~ 7=周日）
- **节假日**：根据日期自动判定（元旦、劳动节、国庆等）

**对应代码**：`import_data.py` — `generate_weather_from_features()` 函数

---

## 三、数据预处理

数据预处理由 `algorithms/preprocessing.py` 中的 `DataPreprocessor` 类统一管理，前端交互入口在 `frontend/views/data_management.py`，后端服务层在 `services/data_service.py`。

### 3.1 数据清洗与格式转换

**目的**：将 ETTh1 原始 CSV 格式转换为系统内部的 `loaddata` 数据库表格式。

**处理步骤**（代码位于 `import_data.py` 第59-80行）：

```python
# 1. 读取CSV，解析日期
df['datetime'] = pd.to_datetime(df['date'])
df['loadvalue'] = df['OT']           # OT列作为负荷预测目标

# 2. 按时间排序并去除空值
df = df.sort_values('datetime').reset_index(drop=True)
df = df.dropna()

# 3. 负荷值缩放到合理电网范围
load_min, load_max = df['loadvalue'].min(), df['loadvalue'].max()
df['loadvalue'] = (df['loadvalue'] - load_min) / (load_max - load_min) * 7000 + 3000
```

**格式映射关系**：

| 原始列 | 转换后字段 | 转换方式 |
|--------|-----------|---------|
| `date` | `recordtime` | `pd.to_datetime()` 解析 |
| `OT` | `loadvalue` | 线性缩放至 3000-10000 MW |
| `HUFL + HULL` | `temperature` | 均值 × 3.0 + 10.0 |
| `MUFL + MULL` | `humidity` | 均值 × 15.0 + 50.0 |
| 日期计算 | `weekday` | `dt.isoweekday()` (1-7) |
| 日期判定 | `holiday` | 匹配预定义节假日列表 (0/1) |

### 3.2 异常值检测与处理

系统提供三种异常值检测算法（代码位于 `algorithms/preprocessing.py` 第15-35行）：

**① Z-score 方法**

计算每个数据点的标准分数，超过阈值则判定为异常：

$$z_i = \frac{|x_i - \mu|}{\sigma}$$

当 $z_i > threshold$（默认阈值=3）时，$x_i$ 被标记为异常值。

```python
def detect_outliers_zscore(self, data, threshold=3):
    mean = np.mean(data)
    std = np.std(data)
    z_scores = np.abs((data - mean) / std)
    outliers = np.where(z_scores > threshold)[0]
    return outliers.tolist(), z_scores.tolist()
```

**② IQR（四分位距）方法**

基于四分位数定义正常范围，超出范围的为异常值：

$$IQR = Q_3 - Q_1$$
$$正常范围：[Q_1 - 1.5 \times IQR,\ Q_3 + 1.5 \times IQR]$$

```python
def detect_outliers_iqr(self, data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = np.where((data < lower_bound) | (data > upper_bound))[0]
    return outliers.tolist(), {'q1': q1, 'q3': q3, 'iqr': iqr}
```

**③ 箱型图方法**

与 IQR 方法算法相同，区别在于同时生成箱型图可视化，直观展示数据分布和异常点位置。

检测后系统生成箱型图，并将检测结果记录到 `dataquality` 数据库表中。

### 3.3 缺失值填充

系统提供两种缺失值插值填充方法（代码位于 `algorithms/preprocessing.py` 第37-59行）：

**① 线性插值**

利用缺失位置前后的已知数据点进行线性内插：

```python
def fill_missing_linear(self, data, missing_indices):
    df = pd.Series(data)
    df.iloc[missing_indices] = np.nan
    df_filled = df.interpolate(method='linear')
    return df_filled.tolist()
```

**② 样条插值**

使用三阶样条函数 `scipy.interpolate.UnivariateSpline` 进行平滑插值，相比线性插值更贴合曲线形态：

```python
def fill_missing_spline(self, data, missing_indices, order=3):
    valid_indices = [i for i in range(len(data)) if i not in missing_indices]
    valid_data = [data[i] for i in valid_indices]
    spline = interpolate.UnivariateSpline(valid_indices, valid_data, k=order, s=0)
    for idx in missing_indices:
        filled_data[idx] = float(spline(idx))
    return filled_data
```

填充完成后自动更新数据库中的对应记录值，并在 `dataprocess` 表中记录处理历史。

### 3.4 数据归一化与标准化

系统采用 **Min-Max 归一化**（代码位于 `algorithms/preprocessing.py` 第61-72行），将负荷值映射到 [0, 1] 区间：

$$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

```python
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def normalize_minmax(self, data, feature_range=(0, 1)):
        data_array = np.array(data).reshape(-1, 1)
        self.scaler = MinMaxScaler(feature_range=feature_range)
        normalized = self.scaler.fit_transform(data_array)
        return normalized.flatten().tolist()

    def denormalize_minmax(self, normalized_data):
        data_array = np.array(normalized_data).reshape(-1, 1)
        denormalized = self.scaler.inverse_transform(data_array)
        return denormalized.flatten().tolist()
```

**关键要点**：
- 训练集使用 `fit_transform` 拟合并记录 scaler 参数（min/max）
- 验证集使用同一个 scaler 的 `transform` 保证归一化一致性
- 预测结果输出前使用 `inverse_transform` 反归一化，恢复真实 MW 值

### 3.5 特征工程与相关性分析

系统在数据分析阶段使用以下特征（代码位于 `services/data_service.py` 第229-283行）：

| 特征 | 说明 | 来源 |
|------|------|------|
| `loadvalue` | 负荷值（预测目标） | OT 列缩放 |
| `temperature` | 温度 | HUFL/HULL 模拟 |
| `humidity` | 湿度 | MUFL/MULL 模拟 |
| `weekday` | 星期几 (1-7) | 日期计算 |
| `holiday` | 是否节假日 (0/1) | 日期判定 |

**相关性分析**使用 Pearson 相关系数矩阵：

```python
def correlation_analysis(self, df, target_col='loadvalue'):
    corr_matrix = df.corr()
    target_corr = corr_matrix[target_col].to_dict()
    return {'correlation_matrix': corr_matrix.to_dict(), 'target_correlation': target_corr}
```

**PCA 主成分分析**用于降维和特征重要性评估：

```python
def pca_analysis(self, data, n_components=2):
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(data)
    return {'explained_variance': pca.explained_variance_ratio_.tolist()}
```

系统会生成特征相关性热力图，辅助用户理解各特征与负荷值之间的关联程度。

> **注意**：实际模型训练阶段采用**单变量时序预测**方式，仅使用负荷值（`loadvalue`）作为模型输入（`input_size=1`）。气象特征（温度、湿度等）主要服务于数据分析和可视化环节。

### 3.6 数据集划分与模型输入构建

#### 数据集划分

数据按照**时间顺序**划分为训练集和验证集（代码位于 `train_all.py` 第37-42行）：

| 数据集 | 时间范围 | 约占比 | 数据量 |
|--------|---------|-------|--------|
| **训练集** | 2016-07-01 ~ 2017-10-31 | ~68% | 约 11,880 条 |
| **验证集** | 2017-11-01 ~ 2018-06-26 | ~32% | 约 5,540 条 |

采用时间顺序切分（而非随机切分），避免未来数据泄漏，符合时间序列预测的评估规范。

#### 模型输入构建 — 滑动窗口法

使用滑动窗口将一维时序数据转换为监督学习样本（代码位于 `algorithms/preprocessing.py` 第93-99行）：

```python
def create_sequences(self, data, seq_length=24):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])     # 前24小时作为输入特征
        y.append(data[i+seq_length])        # 第25小时作为预测标签
    return np.array(X), np.array(y)
```

**参数说明**：
- **序列长度（seq_length）**：默认 24，即使用前 24 小时的归一化负荷值预测下 1 小时
- **输入 X 形状**：`(样本数, 24, 1)` — `(batch_size, seq_len, input_size)`
- **标签 y 形状**：`(样本数,)` — 归一化后的单个负荷值

#### DataLoader 构建

使用 PyTorch 的 `TensorDataset + DataLoader` 封装数据（代码位于 `algorithms/lstm_model.py` 第129-139行）：

```python
def create_dataloader(X, y, batch_size=32, shuffle=True):
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    if len(X_tensor.shape) == 2:
        X_tensor = X_tensor.unsqueeze(-1)   # (N, 24) → (N, 24, 1)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
```

**完整数据流水线**：

```
原始CSV → 时间排序+去空值 → 负荷缩放(3000-10000MW)
       → Min-Max归一化(0-1) → 滑动窗口(24→1)
       → TensorDataset → DataLoader(batch=32)
```

---

## 四、模型详细说明

### 4.1 模型代码文件

| 模型 | 代码文件 | 核心类 |
|------|---------|--------|
| LSTM | `algorithms/lstm_model.py` | `LSTMModel` + `LSTMTrainer` |
| Transformer | `algorithms/transformer_model.py` | `TransformerModel` + `TransformerTrainer` |
| **LSTM-Transformer 混合** | `algorithms/hybrid_model.py` | `HybridModel` + `HybridTrainer` |

模型训练服务：`services/model_service.py` — `ModelService.train_model()`
一键训练脚本：`train_all.py`

### 4.2 LSTM 模型

LSTM（Long Short-Term Memory，长短期记忆网络）是一种特殊的循环神经网络，通过门控机制解决传统 RNN 的梯度消失问题，擅长捕捉时间序列中的短期依赖关系。

**模型结构**（代码位于 `algorithms/lstm_model.py` 第8-31行）：

```python
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,      # 输入维度：1（单变量负荷值）
            hidden_size=hidden_size,    # 隐藏层维度：128
            num_layers=num_layers,      # LSTM堆叠层数：2
            dropout=dropout,            # 层间Dropout：0.2
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)  # 全连接输出层：128→1

    def forward(self, x):                    # x: (batch, 24, 1)
        lstm_out, _ = self.lstm(x)           # (batch, 24, 128)
        out = self.fc(lstm_out[:, -1, :])    # 取最后时间步 → (batch, 1)
        return out
```

**数据流**：`(batch, 24, 1) → LSTM(2层×128) → (batch, 24, 128) → 取最后时间步 → FC(128→1) → (batch, 1)`

**默认超参数**：

| 参数 | 值 | 说明 |
|------|------|------|
| hidden_size | 128 | LSTM 隐藏状态维度 |
| num_layers | 2 | LSTM 堆叠层数 |
| dropout | 0.2 | 层间 Dropout 比率 |
| learning_rate | 0.001 | Adam 优化器学习率 |
| epochs | 30 | 训练轮数 |

### 4.3 Transformer 模型

Transformer 模型基于**多头自注意力机制（Multi-Head Self-Attention）**，能够直接建模序列中任意两个位置之间的依赖关系，不受距离限制。

**模型结构**（代码位于 `algorithms/transformer_model.py` 第7-54行）：

```python
class PositionalEncoding(nn.Module):
    """正弦余弦位置编码"""
    def __init__(self, d_model, max_len=5000):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)   # 偶数维：sin
        pe[:, 1::2] = torch.cos(position * div_term)   # 奇数维：cos
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerModel(nn.Module):
    def __init__(self, input_size=1, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        self.input_embedding = nn.Linear(input_size, d_model)   # 输入嵌入：1→256
        self.pos_encoder = PositionalEncoding(d_model)          # 位置编码
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):                                        # x: (batch, 24, 1)
        x = self.input_embedding(x) * math.sqrt(self.d_model)   # (batch, 24, 256)
        x = self.pos_encoder(x)                                  # 加位置编码
        x = self.transformer_encoder(x)                          # 多头注意力编码
        out = self.fc(x[:, -1, :])                               # 最后时间步 → (batch, 1)
        return out
```

**位置编码公式**：

$$PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

**默认超参数**：

| 参数 | 值 | 说明 |
|------|------|------|
| d_model | 64 | 模型嵌入维度 |
| nhead | 4 | 多头注意力头数 |
| num_layers | 2 | Transformer Encoder 层数 |
| dim_feedforward | 256 | 前馈网络维度 (d_model×4) |
| dropout | 0.1 | Dropout 比率 |
| learning_rate | 0.0001 | Adam 优化器学习率 |
| epochs | 20 | 训练轮数 |

### 4.4 LSTM-Transformer 混合模型（核心模型）

混合模型是本系统的**核心创新点**，将 LSTM 与 Transformer 串联组合，充分发挥两者的优势互补。

**设计思想**：
- **LSTM 阶段**：将原始 1 维输入升维到 64 维特征空间，同时编码局部时序信息（相邻小时间的短期变化规律）
- **Transformer 阶段**：在 LSTM 输出的 64 维特征空间上，通过多头自注意力机制建模全局依赖关系（24 小时内任意两个时间步之间的关联）
- **输出阶段**：Dropout 正则化后通过全连接层映射到 1 维预测值

**完整模型代码**（`algorithms/hybrid_model.py` 第7-47行）：

```python
class HybridModel(nn.Module):
    """LSTM-Transformer混合模型"""

    def __init__(self, input_size=1, lstm_hidden=64, transformer_layers=2,
                 nhead=4, dropout=0.15, output_size=1):
        super().__init__()

        # 第一阶段：LSTM 提取局部时序特征
        self.lstm = nn.LSTM(
            input_size=input_size,       # 输入维度：1
            hidden_size=lstm_hidden,     # 隐藏维度：64
            num_layers=1,                # 单层LSTM（轻量化）
            batch_first=True
        )

        # 第二阶段：Transformer 建模全局依赖
        self.d_model = lstm_hidden       # Transformer输入维度 = LSTM输出维度 = 64
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,        # 模型维度：64
            nhead=nhead,                 # 注意力头数：4（每头16维）
            dim_feedforward=self.d_model * 4,  # FFN维度：256
            dropout=dropout,             # Dropout：0.15
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, transformer_layers)

        # 第三阶段：输出层
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, output_size)  # 64→1

    def forward(self, x):
        # x shape: (batch, 24, 1)

        # Step 1: LSTM 提取时序特征
        lstm_out, _ = self.lstm(x)
        # lstm_out: (batch, 24, 64) — 每个时间步输出64维特征向量

        # Step 2: Transformer 增强全局依赖
        transformer_out = self.transformer(lstm_out)
        # transformer_out: (batch, 24, 64) — 注意力增强后的特征

        # Step 3: 取最后时间步 + Dropout + 全连接输出
        out = self.dropout(transformer_out[:, -1, :])  # (batch, 64)
        out = self.fc(out)                              # (batch, 1)
        return out
```

**模型结构示意图**：

```
输入序列 x: (batch, 24, 1)
    │
    │  ┌─────────────────────────────────────┐
    └──│          LSTM 层 (1层×64)           │
       │  input_size=1, hidden_size=64       │
       │  捕捉局部时序依赖：                 │
       │  · 相邻小时间的上升/下降趋势        │
       │  · 日内负荷周期性波动               │
       └────────────┬────────────────────────┘
                    │ (batch, 24, 64)
    ┌───────────────▼────────────────────────┐
    │      Transformer Encoder (2层)         │
    │  d_model=64, nhead=4, ffn=256          │
    │  多头自注意力建模全局依赖：             │
    │  · 24小时内任意两个时间步的关联         │
    │  · 例：凌晨3点与上午10点的负荷关系     │
    │  · 不受距离限制，直接建模远距离依赖     │
    └────────────────┬───────────────────────┘
                     │ (batch, 24, 64)
                     │ 取最后时间步 [:, -1, :]
    ┌────────────────▼───────────────────────┐
    │  Dropout(0.15) + FC(64→1)              │
    │  防止过拟合 + 全连接输出               │
    └────────────────┬───────────────────────┘
                     │
              预测值: (batch, 1)
```

**混合模型相比单一模型的优势**：

| 对比维度 | 纯 LSTM | 纯 Transformer | LSTM-Transformer 混合 |
|---------|---------|----------------|---------------------|
| 局部时序建模 | ✅ 强（门控递归） | ❌ 弱（需位置编码辅助） | ✅ LSTM 负责 |
| 全局依赖建模 | ❌ 弱（长距离衰减） | ✅ 强（自注意力） | ✅ Transformer 负责 |
| 参数效率 | 中等 | 较高（d_model大） | ✅ 高（LSTM降维后Transformer轻量） |
| 训练稳定性 | ✅ 好 | 需要精细调参 | ✅ LSTM预编码使Transformer更稳定 |

**默认超参数**：

| 参数 | 值 | 说明 |
|------|------|------|
| lstm_hidden | 64 | LSTM 隐藏维度（也是 Transformer 的 d_model） |
| transformer_layers | 2 | Transformer Encoder 层数 |
| nhead | 4 | 多头注意力头数（每头 64/4=16 维） |
| dropout | 0.15 | Dropout 比率 |
| learning_rate | 0.0005 | Adam 优化器学习率 |
| epochs | 25 | 训练轮数 |

### 4.5 模型训练流程

三种模型共享统一的训练框架（代码位于各模型文件的 Trainer 类中）：

**训练步骤**：

```
1. 从数据库查询指定时间范围的负荷数据
2. Min-Max 归一化到 [0, 1]
3. 滑动窗口（seq_length=24）构建 (X, y) 样本对
4. 创建 PyTorch DataLoader（batch_size=32, 训练集shuffle=True）
5. 循环 epochs 轮：
   ├── train_epoch()：
   │   model.train() → 遍历batch → 前向传播 → MSELoss → 反向传播 → Adam更新
   └── validate()：
       model.eval() → torch.no_grad() → 遍历batch → 计算验证损失
6. 保存模型权重为 .pth 文件
7. 计算验证集评估指标（MAE / RMSE / MAPE / R²）
```

**训练器核心代码**（以混合模型为例，`algorithms/hybrid_model.py` 第49-113行）：

```python
class HybridTrainer:
    def __init__(self, model, learning_rate=0.0005, device='cpu'):
        self.model = model.to(device)
        self.criterion = nn.MSELoss()                          # 损失函数：均方误差
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化器

    def train_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            outputs = self.model(batch_x)                      # 前向传播
            loss = self.criterion(outputs.squeeze(), batch_y)  # 计算损失
            self.optimizer.zero_grad()                         # 梯度清零
            loss.backward()                                    # 反向传播
            self.optimizer.step()                              # 参数更新
            epoch_loss += loss.item()
        return epoch_loss / len(train_loader)

    def validate(self, valid_loader):
        self.model.eval()
        with torch.no_grad():                                  # 不计算梯度
            epoch_loss = sum(
                self.criterion(self.model(bx).squeeze(), by).item()
                for bx, by in valid_loader
            )
        return epoch_loss / len(valid_loader)

    def train(self, train_loader, valid_loader, epochs=25):
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            valid_loss = self.validate(valid_loader)
            # 每10轮打印一次
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train: {train_loss:.6f}, Valid: {valid_loss:.6f}')
        return self.train_losses, self.valid_losses
```

**损失函数**：MSELoss（均方误差）— $L = \frac{1}{n}\sum(y_i - \hat{y}_i)^2$

**优化器**：Adam — 自适应学习率优化算法

---

## 五、预测方法详细说明

### 5.1 预测流程

预测服务由 `services/predict_service.py` 的 `PredictService.execute_predict_task()` 方法实现（第78-169行）。

**完整预测流程**：

```
1. 获取预测任务信息（预测起止时间、模型版本）
2. 加载已训练好的模型权重
3. 查询预测起点前 7 天的历史负荷数据（至少需要 24 条）
4. 对历史数据做 Min-Max 归一化
5. 取最近 24 个归一化值作为初始输入序列
6. 自回归循环预测（每 6 小时一个预测点）：
   ├── 将当前 24 个值输入模型 → 输出 1 个归一化预测值
   ├── 反归一化 → 得到真实 MW 预测值
   ├── 将归一化预测值追加到输入序列末尾，移除最早的值
   └── 保存预测结果到 predictresult 数据库表
7. 更新任务状态为"已完成"
```

**自回归预测核心代码**（`services/predict_service.py` 第120-142行）：

```python
# 取最近24个时间步作为初始输入
input_sequence = normalized_history[-24:]

for predict_time in predict_times:          # 每6小时一个预测点
    X_input = np.array([input_sequence])     # 构造模型输入 (1, 24)
    pred_normalized = trainer.predict(X_input)[0]  # 模型推理

    # 反归一化，恢复真实MW值
    pred_value = self.preprocessor.denormalize_minmax([pred_normalized])[0]
    predictions.append(pred_value)

    # 自回归：用预测值更新输入序列（滑动窗口前移）
    input_sequence = input_sequence[1:] + [pred_normalized]

    # 保存到数据库
    result = PredictResult(taskid=taskid, predicttime=predict_time, predictvalue=pred_value)
    db.session.add(result)
```

**自回归预测示意**：

```
初始输入:  [h1, h2, h3, ..., h24] → 模型 → 预测 p1
第二步:    [h2, h3, h4, ..., h24, p1] → 模型 → 预测 p2
第三步:    [h3, h4, h5, ..., p1, p2] → 模型 → 预测 p3
...
```

### 5.2 预测时间粒度

- **预测步长**：6 小时
- **预测范围**：用户可自定义起止日期
- **示例**：预测未来 7 天 → 生成 28 个预测点（7×24/6）

---

## 六、预测结果与结果分析

### 6.1 评估指标

预测结果通过 4 个指标进行量化评估（代码位于 `algorithms/preprocessing.py` 第108-132行）：

| 指标 | 公式 | 含义 | 理想值 |
|------|------|------|--------|
| **MAE** | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | 平均绝对误差（MW） | 越小越好 |
| **RMSE** | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ | 均方根误差（MW），对大偏差更敏感 | 越小越好 |
| **MAPE** | $\frac{100\%}{n}\sum\left\|\frac{y_i - \hat{y}_i}{y_i}\right\|$ | 平均绝对百分比误差 | <5% 为优秀 |
| **R²** | $1 - \frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}$ | 决定系数 | 越接近 1 越好 |

**指标计算代码**：

```python
def calculate_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2score': r2}
```

### 6.2 预测结果可视化

系统提供两种可视化方式（代码位于 `services/predict_service.py` 第243-281行）：

**① 预测值与实际值对比折线图**

在同一坐标系下绘制预测值和实际值两条曲线，直观对比偏差程度。当更新了实际值后自动生成。

**② 预测散点图**

以实际值为横轴、预测值为纵轴绘制散点图。理想情况下所有点应分布在 y=x 对角线附近，偏离越远说明预测误差越大。

### 6.3 三种模型预测效果对比分析

#### 6.3.1 训练配置

基于 ETTh1 数据集，三种模型的训练配置如下：

| 模型 | 训练轮数 | 学习率 | Batch Size | 训练样本 | 验证样本 |
|------|---------|--------|-----------|---------|---------|
| LSTM | 30 | 0.001 | 64 | 11,688 | 5,684 |
| Transformer | 20 | 0.0001 | 64 | 11,688 | 5,684 |
| **LSTM-Transformer** | 25 | 0.0005 | 64 | 11,688 | 5,684 |

训练集时间范围为 2016-07-01 ~ 2017-10-31，验证集时间范围为 2017-11-01 ~ 2018-06-26，采用严格的时间顺序划分，避免未来数据泄漏。

#### 6.3.2 验证集评估结果对比

三种模型在验证集上的评估结果如下表所示（评估脚本：`evaluate_and_plot.py`）：

| 模型 | MAE | RMSE | MAPE(%) | R² |
|:----:|:---:|:----:|:-------:|:--:|
| LSTM | 0.0133 | 0.0189 | 7.58 | 0.9316 |
| Transformer | **0.0126** | **0.0163** | **7.17** | **0.9487** |
| LSTM-Transformer | 0.0133 | 0.0180 | 7.52 | 0.9378 |

> 注：上表中的 MAE、RMSE 为归一化空间下的值（数据经 Min-Max 归一化至 [0,1] 区间），MAPE 和 R² 不受归一化影响。**加粗**表示该指标最优。

#### 6.3.3 结果分析

**（1）整体表现**

三种模型的 R² 均超过 0.93，说明模型整体拟合效果良好，能够解释超过 93% 的负荷变化。MAPE 控制在 7%~8% 之间，达到了短期负荷预测的实用水平。

**（2）各模型表现分析**

- **LSTM 模型**（MAE=0.0133, R²=0.9316）：作为基准模型，LSTM 利用门控机制有效捕捉了负荷数据的短期时序依赖关系，在 30 个 epoch 内即可收敛。但由于 LSTM 的递归结构使其对长距离依赖的建模能力有限，在验证集上的 R² 为三者中最低（0.9316），对负荷突变和跨日周期规律的捕捉稍显不足。

- **Transformer 模型**（MAE=0.0126, R²=0.9487）：在所有四项指标上均取得最优结果。Transformer 的自注意力机制能够直接建模任意时间步之间的依赖关系，无需逐步传递信息，因此在捕捉负荷数据的日周期性（24h）和周周期性（168h）方面具有天然优势。其 MAPE 仅为 7.17%，比 LSTM 降低了约 5.4%（相对降幅）。该模型采用较小的 d_model=64 和仅 2 层编码器，有效控制了参数规模，避免了在当前数据量下的过拟合。

- **LSTM-Transformer 混合模型**（MAE=0.0133, R²=0.9378）：混合模型的表现介于 LSTM 和 Transformer 之间，RMSE（0.0180）优于 LSTM（0.0189）但略逊于 Transformer（0.0163），R² 较 LSTM 提升了 0.67%。混合架构中 LSTM 层将 1 维原始输入编码为 64 维隐状态向量，为后续 Transformer 层提供了更丰富的特征表示。然而在当前超参数配置下，其性能未能超越经过良好调参的纯 Transformer 模型，这可能与以下因素有关：①  LSTM 层固定为单层，特征提取能力受限；② 混合模型的超参数（学习率、隐藏维度等）仍有进一步优化空间。

**（3）模型对比总结**

| 对比维度 | LSTM | Transformer | LSTM-Transformer |
|---------|------|-------------|------------------|
| 短期趋势捕捉 | ★★★★☆ | ★★★☆☆ | ★★★★☆ |
| 长距离依赖建模 | ★★☆☆☆ | ★★★★★ | ★★★★☆ |
| 训练稳定性 | ★★★★★ | ★★★☆☆ | ★★★★☆ |
| 参数效率 | ★★★★☆ | ★★★★☆ | ★★★☆☆ |
| 综合预测精度 | 第三 | **第一** | 第二 |

**（4）预测局限性**

1. **误差累积问题**：采用自回归方式进行多步预测时（即将上一步预测值作为下一步输入），误差会逐步累积——预测时间越长，精度下降越明显
2. **单变量输入限制**：当前模型仅使用负荷历史值作为输入特征，未引入温度、湿度、节假日等外部变量，限制了模型的预测上界
3. **泛化能力**：模型在 ETTh1 数据集上训练和验证，对不同地区、不同用电模式的电网数据需要重新训练或微调

---

## 七、代码文件对照表

| 功能模块 | 代码文件 | 关键类/函数 |
|---------|---------|------------|
| 数据导入 | `import_data.py` | `import_data()`, `generate_weather_from_features()` |
| 数据预处理 | `algorithms/preprocessing.py` | `DataPreprocessor` 类, `calculate_metrics()` |
| LSTM 模型 | `algorithms/lstm_model.py` | `LSTMModel`, `LSTMTrainer`, `create_dataloader()` |
| Transformer 模型 | `algorithms/transformer_model.py` | `TransformerModel`, `TransformerTrainer`, `PositionalEncoding` |
| 混合模型 | `algorithms/hybrid_model.py` | `HybridModel`, `HybridTrainer` |
| 数据管理服务 | `services/data_service.py` | `DataService` 类 |
| 模型管理服务 | `services/model_service.py` | `ModelService` 类 |
| 预测服务 | `services/predict_service.py` | `PredictService` 类 |
| 一键训练脚本 | `train_all.py` | `main()`, `train_one_model()` |
| 前端-数据管理 | `frontend/views/data_management.py` | 上传/查询/异常检测/处理 |
| 前端-模型管理 | `frontend/views/model_management.py` | 配置/训练/版本管理 |
| 前端-预测管理 | `frontend/views/predict_management.py` | 创建任务/执行/结果分析 |
| 数据库模型 | `models/database.py` | SQLAlchemy ORM 定义 |
| 可视化工具 | `utils/visualization.py` | 图表生成函数 |

---

## 八、项目结构

```
code/
├── app.py                    # Flask REST API 主入口
├── database.sql              # 数据库建表及初始化脚本
├── import_data.py            # ETTh1 数据导入脚本
├── train_all.py              # 一键训练三种模型脚本
├── requirements.txt          # Python 依赖清单
├── ETT-small/                # ETT-small 数据集
│   ├── ETTh1.csv             # 当前使用的数据集文件
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
├── algorithms/               # 深度学习模型实现
│   ├── preprocessing.py      # 数据预处理（归一化/异常检测/序列构建）
│   ├── lstm_model.py         # LSTM 模型 + 训练器
│   ├── transformer_model.py  # Transformer 模型 + 训练器
│   └── hybrid_model.py       # LSTM-Transformer 混合模型 + 训练器
├── models/
│   └── database.py           # SQLAlchemy ORM 数据库模型定义
├── services/                 # 业务逻辑服务层
│   ├── auth_service.py       # 认证服务
│   ├── data_service.py       # 数据管理服务
│   ├── model_service.py      # 模型管理服务
│   ├── predict_service.py    # 预测服务
│   └── admin_service.py      # 系统管理服务
├── utils/
│   ├── response.py           # 统一响应格式工具
│   └── visualization.py      # Matplotlib 图表生成工具
├── saved_models/             # 已训练模型权重 (.pth)
└── frontend/                 # Streamlit 前端
    ├── main.py               # 主入口（路由与导航）
    ├── config.py             # 前端配置（角色、模型类型等）
    ├── ui_utils/
    │   ├── api_client.py     # 服务层调用封装
    │   └── helpers.py        # UI 工具函数
    └── views/
        ├── login.py          # 登录页面
        ├── admin.py          # 系统管理页面
        ├── data_management.py    # 数据管理页面
        ├── model_management.py   # 模型管理页面
        └── predict_management.py # 预测管理页面
```
