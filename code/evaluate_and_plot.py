# 评估脚本：生成模型对比表格 + LSTM-Transformer残差分析图
import sys, os, glob, json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Songti SC', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algorithms.preprocessing import DataPreprocessor, calculate_metrics
from algorithms.lstm_model import LSTMModel, LSTMTrainer, create_dataloader
from algorithms.transformer_model import TransformerModel, TransformerTrainer
from algorithms.hybrid_model import HybridModel, HybridTrainer
from flask import Flask
from models.database import db, ModelConfig, TrainRecord, ModelVersion

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEQ_LENGTH = 24
TRAIN_START = '2016-07-01'
TRAIN_END = '2017-10-31'
VALID_START = '2017-11-01'
VALID_END = '2018-06-30'

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:123456@localhost:3306/electric_load_forecasting_system_with_deep_learning_2026'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# ===== 1. 从CSV加载数据 =====
print("📊 加载ETTh1数据...")
df = pd.read_csv('./ETT-small/ETTh1.csv', parse_dates=['date'])
df = df.set_index('date')

train_df = df[TRAIN_START:TRAIN_END]
valid_df = df[VALID_START:VALID_END]

# 使用OT列作为负荷值
train_values = train_df['OT'].values.tolist()
valid_values = valid_df['OT'].values.tolist()
print(f"   训练集: {len(train_values)} 条, 验证集: {len(valid_values)} 条")

# ===== 2. 数据预处理 =====
preprocessor = DataPreprocessor()
train_norm = preprocessor.normalize_minmax(train_values)
valid_array = np.array(valid_values).reshape(-1, 1)
valid_norm = preprocessor.scaler.transform(valid_array).flatten().tolist()

X_train, y_train = preprocessor.create_sequences(train_norm, SEQ_LENGTH)
X_valid, y_valid = preprocessor.create_sequences(valid_norm, SEQ_LENGTH)
print(f"   验证样本: {len(X_valid)}")

# 保存验证集时间戳（从第25个点开始，因为前24个用于输入序列）
valid_dates = valid_df.index[SEQ_LENGTH:]

def latest_model_path(prefix):
    """优先使用最新训练出的模型文件，避免评估脚本继续读取旧权重。"""
    paths = glob.glob(f'./saved_models/{prefix}_*.pth')
    if not paths:
        raise FileNotFoundError(f'未找到模型文件: ./saved_models/{prefix}_*.pth')
    return max(paths, key=os.path.getmtime)


def db_model_info(modelname, active_only=False):
    """读取数据库中训练脚本选定的模型路径和超参数，失败时返回None。"""
    try:
        with app.app_context():
            query = (
                db.session.query(ModelConfig, TrainRecord, ModelVersion)
                .join(TrainRecord, ModelConfig.configid == TrainRecord.configid)
                .join(ModelVersion, TrainRecord.trainid == ModelVersion.trainid)
                .filter(ModelConfig.modelname == modelname)
            )
            if active_only:
                query = query.filter(ModelVersion.isactive == 1)
            row = query.order_by(TrainRecord.endtime.desc()).first()
            if not row:
                return None
            config, record, _ = row
            return {
                'path': record.modelpath,
                'hyperparams': json.loads(config.hyperparams) if config.hyperparams else {}
            }
    except Exception as exc:
        print(f"⚠️ 无法从数据库读取{modelname}模型信息，使用本地最新文件: {exc}")
        return None


# ===== 3. 加载模型并预测 =====
db_lstm = db_model_info('LSTM_v1')
db_transformer = db_model_info('Transformer_v1')
db_hybrid = db_model_info('Hybrid_v1', active_only=True)

lstm_params = (db_lstm or {}).get('hyperparams', {})
transformer_params = (db_transformer or {}).get('hyperparams', {})
hybrid_params = (db_hybrid or {}).get('hyperparams', {})

models_info = {
    'LSTM': {
        'model': LSTMModel(
            input_size=1,
            hidden_size=lstm_params.get('hidden_size', 128),
            num_layers=lstm_params.get('num_layers', 2),
            dropout=lstm_params.get('dropout', 0.2)
        ),
        'trainer_cls': LSTMTrainer,
        'path': (db_lstm or {}).get('path') or latest_model_path('lstm'),
        'lr': lstm_params.get('learning_rate', 0.001)
    },
    'Transformer': {
        'model': TransformerModel(
            input_size=1,
            d_model=transformer_params.get('d_model', 64),
            nhead=transformer_params.get('nhead', 4),
            num_layers=transformer_params.get('num_layers', 2),
            dropout=transformer_params.get('dropout', 0.1)
        ),
        'trainer_cls': TransformerTrainer,
        'path': (db_transformer or {}).get('path') or latest_model_path('transformer'),
        'lr': transformer_params.get('learning_rate', 0.0001)
    },
    'LSTM-Transformer': {
        'model': HybridModel(
            input_size=1,
            lstm_hidden=hybrid_params.get('lstm_hidden', 64),
            transformer_layers=hybrid_params.get('transformer_layers', 2),
            nhead=hybrid_params.get('nhead', 4),
            dropout=hybrid_params.get('dropout', 0.15)
        ),
        'trainer_cls': HybridTrainer,
        'path': (db_hybrid or {}).get('path') or latest_model_path('hybrid'),
        'lr': hybrid_params.get('learning_rate', 0.0005)
    }
}

all_metrics = {}
all_predictions = {}

for name, info in models_info.items():
    print(f"\n🔄 加载模型: {name} ...")
    trainer = info['trainer_cls'](info['model'], info['lr'], DEVICE)
    trainer.load_model(info['path'])
    preds = trainer.predict(X_valid)
    metrics = calculate_metrics(y_valid.tolist(), preds.tolist())
    all_metrics[name] = metrics
    all_predictions[name] = preds
    print(f"   MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  MAPE={metrics['mape']:.2f}%  R²={metrics['r2score']:.4f}")

# ===== 4. 生成对比表格 =====
print("\n" + "=" * 70)
print("📋 各模型在验证集上的评估结果对比")
print("=" * 70)
print(f"{'模型':<20s} {'MAE':>10s} {'RMSE':>10s} {'MAPE(%)':>10s} {'R²':>10s}")
print("-" * 60)
for name, m in all_metrics.items():
    print(f"{name:<20s} {m['mae']:>10.4f} {m['rmse']:>10.4f} {m['mape']:>10.2f} {m['r2score']:>10.4f}")
print("-" * 60)

# 保存为CSV
metrics_df = pd.DataFrame(all_metrics).T
metrics_df.columns = ['MAE', 'RMSE', 'MAPE(%)', 'R2']
metrics_df.index.name = '模型'
os.makedirs('images', exist_ok=True)
metrics_df.to_csv('images/model_comparison.csv', encoding='utf-8-sig')
print("✅ 表格已保存到 model_comparison.csv")

# ===== 5. LSTM-Transformer 残差分析图 =====
hybrid_preds = all_predictions['LSTM-Transformer']
residuals = y_valid - hybrid_preds  # 残差 = 真实值 - 预测值

# 获取时段信息
hours = np.array([d.hour for d in valid_dates[:len(residuals)]])

# 定义时段
def get_period(h):
    if 6 <= h < 12:
        return '早高峰(6-12h)'
    elif 12 <= h < 18:
        return '午后(12-18h)'
    elif 18 <= h < 24:
        return '晚高峰(18-24h)'
    return '夜间(0-6h)'

periods = np.array([get_period(h) for h in hours])
period_names = ['早高峰(6-12h)', '午后(12-18h)', '晚高峰(18-24h)', '夜间(0-6h)']
period_colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']

# ----- 图1: 残差直方图（按时段分组） -----
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('LSTM-Transformer 模型预测残差直方图（按时段分组）', fontsize=16, fontweight='bold')

for idx, (pname, color) in enumerate(zip(period_names, period_colors)):
    ax = axes[idx // 2][idx % 2]
    mask = periods == pname
    res_period = residuals[mask]
    ax.hist(res_period, bins=40, color=color, alpha=0.75, edgecolor='black', linewidth=0.5, density=True)
    # 拟合正态分布曲线
    mu, sigma = np.mean(res_period), np.std(res_period)
    x_range = np.linspace(res_period.min(), res_period.max(), 100)
    ax.plot(x_range, stats.norm.pdf(x_range, mu, sigma), 'k--', linewidth=2, label=f'N({mu:.4f},{sigma:.4f}^2)')
    ax.set_title(f'{pname}\n(n={len(res_period)}, μ={mu:.4f}, σ={sigma:.4f})', fontsize=12)
    ax.set_xlabel('残差值', fontsize=10)
    ax.set_ylabel('概率密度', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/residual_histogram.png', dpi=200, bbox_inches='tight')
print("✅ 残差直方图已保存到 residual_histogram.png")

# ----- 图2: Q-Q图（按时段分组） -----
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('LSTM-Transformer 模型预测残差Q-Q图（按时段分组）', fontsize=16, fontweight='bold')

for idx, (pname, color) in enumerate(zip(period_names, period_colors)):
    ax = axes2[idx // 2][idx % 2]
    mask = periods == pname
    res_period = residuals[mask]
    (osm, osr), (slope, intercept, r) = stats.probplot(res_period, dist="norm")
    ax.scatter(osm, osr, color=color, alpha=0.5, s=8, label=f'残差点 (n={len(res_period)})')
    ax.plot(osm, slope * osm + intercept, 'k--', linewidth=2, label=f'拟合线 (R2={r**2:.4f})')
    ax.set_title(f'{pname}', fontsize=12)
    ax.set_xlabel('理论分位数', fontsize=10)
    ax.set_ylabel('样本分位数', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/residual_qq_plot.png', dpi=200, bbox_inches='tight')
print("✅ Q-Q图已保存到 residual_qq_plot.png")

print("\n🎉 全部完成！")
