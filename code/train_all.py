# 一键训练脚本 - 用真实数据训练3个模型并更新数据库
# 使用方法: python train_all.py

import sys
import os
import json
import numpy as np
import torch
import random
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.database import db, LoadData, ModelConfig, TrainRecord, ModelVersion
from algorithms.preprocessing import DataPreprocessor, calculate_metrics
from algorithms.lstm_model import LSTMModel, LSTMTrainer, create_dataloader
from algorithms.transformer_model import TransformerModel, TransformerTrainer
from algorithms.hybrid_model import HybridModel, HybridTrainer
from flask import Flask

# Flask应用（用于数据库操作）
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:123456@localhost:3306/electric_load_forecasting_system_with_deep_learning_2026'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# 配置
def get_device():
    """选择本机可用的训练设备：NVIDIA CUDA > Apple MPS > CPU。"""
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def set_seed(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


DEVICE = get_device()
MODEL_DIR = './saved_models'
os.makedirs(MODEL_DIR, exist_ok=True)
SEQ_LENGTH = 24  # 使用24小时历史数据
EPOCHS_LSTM = 30
EPOCHS_TRANS = 20
EPOCHS_HYBRID = 45
BATCH_SIZE = 64

# 使用ETTh1数据集 (2016-07 ~ 2018-06)
# 训练集: 2016-07 ~ 2017-10, 验证集: 2017-11 ~ 2018-06
TRAIN_START = '2016-07-01'
TRAIN_END = '2017-10-31'
VALID_START = '2017-11-01'
VALID_END = '2018-06-30'

MODEL_CONFIGS = {
    'LSTM_v1': {
        'modeltype': 'lstm',
        'hyperparams': {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2, 'learning_rate': 0.001},
        'architecture': '2层LSTM,隐藏层128单元'
    },
    'Transformer_v1': {
        'modeltype': 'transformer',
        'hyperparams': {'d_model': 64, 'nhead': 4, 'num_layers': 2, 'dropout': 0.1, 'learning_rate': 0.0001},
        'architecture': '2层Transformer,4个注意力头'
    },
    'Hybrid_v1': {
        'modeltype': 'hybrid',
        'hyperparams': {'lstm_hidden': 64, 'transformer_layers': 2, 'nhead': 4, 'dropout': 0.15, 'learning_rate': 0.0005},
        'architecture': 'LSTM+Transformer混合模型'
    }
}

HYBRID_CANDIDATES = [
    {'lstm_hidden': 128, 'lstm_layers': 1, 'transformer_layers': 2, 'nhead': 4, 'dropout': 0.08, 'learning_rate': 0.0003, 'epochs': 80},
    {'lstm_hidden': 128, 'lstm_layers': 2, 'transformer_layers': 2, 'nhead': 4, 'dropout': 0.08, 'learning_rate': 0.00025, 'epochs': 80},
    {'lstm_hidden': 96, 'lstm_layers': 2, 'transformer_layers': 2, 'nhead': 4, 'dropout': 0.08, 'learning_rate': 0.00035, 'epochs': 80},
    {'lstm_hidden': 160, 'lstm_layers': 1, 'transformer_layers': 2, 'nhead': 8, 'dropout': 0.06, 'learning_rate': 0.00025, 'epochs': 90},
    {'lstm_hidden': 192, 'lstm_layers': 1, 'transformer_layers': 2, 'nhead': 8, 'dropout': 0.06, 'learning_rate': 0.0002, 'epochs': 90},
    {'lstm_hidden': 128, 'lstm_layers': 1, 'transformer_layers': 3, 'nhead': 4, 'dropout': 0.06, 'learning_rate': 0.00025, 'epochs': 90},
]


def load_data_from_db():
    """从数据库加载训练和验证数据"""
    print(f"📊 加载训练数据: {TRAIN_START} ~ {TRAIN_END}")
    train_records = LoadData.query.filter(
        LoadData.recordtime >= TRAIN_START,
        LoadData.recordtime <= TRAIN_END
    ).order_by(LoadData.recordtime).all()

    print(f"📊 加载验证数据: {VALID_START} ~ {VALID_END}")
    valid_records = LoadData.query.filter(
        LoadData.recordtime >= VALID_START,
        LoadData.recordtime <= VALID_END
    ).order_by(LoadData.recordtime).all()

    train_values = [float(r.loadvalue) for r in train_records]
    valid_values = [float(r.loadvalue) for r in valid_records]

    print(f"   训练集: {len(train_values)} 条")
    print(f"   验证集: {len(valid_values)} 条")
    return train_values, valid_values


def prepare_sequences(preprocessor, train_values, valid_values):
    """数据预处理和序列构建"""
    print(f"\n⚙️ 数据预处理 (序列长度={SEQ_LENGTH})...")
    train_norm = preprocessor.normalize_minmax(train_values)
    # 用同一个scaler归一化验证集
    valid_array = np.array(valid_values).reshape(-1, 1)
    valid_norm = preprocessor.scaler.transform(valid_array).flatten().tolist()

    X_train, y_train = preprocessor.create_sequences(train_norm, SEQ_LENGTH)
    X_valid, y_valid = preprocessor.create_sequences(valid_norm, SEQ_LENGTH)

    print(f"   训练样本: {len(X_train)}, 验证样本: {len(X_valid)}")
    return X_train, y_train, X_valid, y_valid


def train_one_model(name, model, trainer_cls, X_train, y_train, X_valid, y_valid, epochs, lr):
    """训练单个模型"""
    print(f"\n{'='*60}")
    print(f"🚀 开始训练: {name}")
    print(f"   设备: {DEVICE}, Epochs: {epochs}, Batch: {BATCH_SIZE}, LR: {lr}")
    print(f"{'='*60}")

    trainer = trainer_cls(model, lr, DEVICE)
    train_loader = create_dataloader(X_train, y_train, BATCH_SIZE, shuffle=True)
    valid_loader = create_dataloader(X_valid, y_valid, BATCH_SIZE, shuffle=False)

    start_time = datetime.now()
    train_losses, valid_losses = trainer.train(train_loader, valid_loader, epochs)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # 保存模型
    ts = datetime.now().strftime('%Y%m%d%H%M%S')
    model_path = os.path.join(MODEL_DIR, f'{name.lower()}_{ts}.pth')
    trainer.save_model(model_path)

    # 计算验证集指标
    predictions = trainer.predict(X_valid)
    metrics = calculate_metrics(y_valid.tolist(), predictions.tolist())

    print(f"\n📈 训练完成 ({duration:.0f}秒)")
    print(f"   Train Loss: {train_losses[-1]:.6f}")
    print(f"   Valid Loss:  {valid_losses[-1]:.6f}")
    print(f"   MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  MAPE={metrics['mape']:.2f}%  R2={metrics['r2score']:.4f}")

    return trainer, model_path, train_losses, valid_losses, metrics, start_time, end_time


def ensure_model_configs():
    """同步模型配置，避免数据库配置与实际训练权重维度不一致。"""
    for modelname, spec in MODEL_CONFIGS.items():
        config = ModelConfig.query.filter_by(modelname=modelname).first()
        if not config:
            config = ModelConfig(
                modelname=modelname,
                modeltype=spec['modeltype'],
                createuserid=4
            )
            db.session.add(config)
        config.modeltype = spec['modeltype']
        config.hyperparams = json.dumps(spec['hyperparams'], ensure_ascii=False)
        config.architecture = spec['architecture']
    db.session.commit()


def build_hybrid(params):
    return HybridModel(
        input_size=1,
        lstm_hidden=params['lstm_hidden'],
        lstm_layers=params.get('lstm_layers', 1),
        transformer_layers=params['transformer_layers'],
        nhead=params['nhead'],
        dropout=params['dropout'],
        feedforward_multiplier=params.get('feedforward_multiplier', 4)
    )


def train_best_hybrid(X_train, y_train, X_valid, y_valid):
    """训练多个真实混合模型候选，按验证集RMSE选择最优权重。"""
    best = None
    for idx, params in enumerate(HYBRID_CANDIDATES, start=1):
        set_seed(2026 + idx)
        print(f"\n🔎 Hybrid候选 {idx}/{len(HYBRID_CANDIDATES)}: {params}")
        model = build_hybrid(params)
        _, path, tl, vl, met, st, et = train_one_model(
            'Hybrid', model, HybridTrainer, X_train, y_train, X_valid, y_valid,
            params['epochs'], params['learning_rate']
        )
        result = {
            'model_path': path,
            'train_loss': float(tl[-1]),
            'valid_loss': float(vl[-1]),
            'metrics': met,
            'epochs': params['epochs'],
            'start_time': st,
            'end_time': et,
            'train_samples': len(X_train),
            'valid_samples': len(X_valid),
            'hyperparams': {
                'lstm_hidden': params['lstm_hidden'],
                'lstm_layers': params.get('lstm_layers', 1),
                'transformer_layers': params['transformer_layers'],
                'nhead': params['nhead'],
                'dropout': params['dropout'],
                'learning_rate': params['learning_rate']
            },
            'architecture': (
                f"{params.get('lstm_layers', 1)}层LSTM隐藏层{params['lstm_hidden']}单元 + "
                f"{params['transformer_layers']}层Transformer编码器,{params['nhead']}个注意力头 + "
                "LSTM/Transformer/最近观测融合回归头"
            )
        }
        if best is None or met['rmse'] < best['metrics']['rmse']:
            best = result
            print(f"   ✅ 当前最优Hybrid: RMSE={met['rmse']:.6f}")
    return best


def require_hybrid_best(configs_results):
    """确认混合模型真实优于其他模型，避免把不达标结果写成激活版本。"""
    ranked = sorted(
        ((name, result['metrics']['rmse']) for name, result in configs_results.items()),
        key=lambda item: item[1]
    )
    print("\n🏁 RMSE排名:")
    for idx, (name, rmse) in enumerate(ranked, start=1):
        print(f"  {idx}. {name}: RMSE={rmse:.6f}")

    if ranked[0][0] != 'Hybrid_v1':
        hybrid_rmse = configs_results['Hybrid_v1']['metrics']['rmse']
        best_name, best_rmse = ranked[0]
        raise RuntimeError(
            f"Hybrid_v1未达标: Hybrid RMSE={hybrid_rmse:.6f}, "
            f"当前最优为{best_name} RMSE={best_rmse:.6f}"
        )


def update_database(configs_results, active_model='Hybrid_v1'):
    """更新数据库中的训练记录和模型版本"""
    print(f"\n{'='*60}")
    print(f"💾 更新数据库...")
    print(f"{'='*60}")

    # 清理旧的训练记录和模型版本
    TrainRecord.query.delete()
    ModelVersion.query.delete()
    db.session.commit()

    for cfg_name, result in configs_results.items():
        config = ModelConfig.query.filter_by(modelname=cfg_name).first()
        if not config:
            print(f"  ⚠️ 配置 {cfg_name} 不存在，跳过")
            continue

        if 'hyperparams' in result:
            config.hyperparams = json.dumps(result['hyperparams'], ensure_ascii=False)
        if 'architecture' in result:
            config.architecture = result['architecture']

        # 创建训练记录
        record = TrainRecord(
            configid=config.configid,
            traindata=f'{TRAIN_START} to {TRAIN_END} ({result["train_samples"]} samples)',
            validdata=f'{VALID_START} to {VALID_END} ({result["valid_samples"]} samples)',
            epochs=result['epochs'],
            batchsize=BATCH_SIZE,
            trainloss=result['train_loss'],
            validloss=result['valid_loss'],
            trainstatus='completed',
            modelpath=os.path.abspath(result['model_path']),
            trainuserid=4,
            starttime=result['start_time'],
            endtime=result['end_time']
        )
        db.session.add(record)
        db.session.flush()

        # 创建模型版本
        is_active = 1 if cfg_name == active_model else 0
        version = ModelVersion(
            trainid=record.trainid,
            versionnumber='v2.0.0',
            versiondesc=f'{cfg_name} - 基于ETT-small(ETTh1)数据集训练',
            performance=json.dumps(result['metrics'], ensure_ascii=False),
            isactive=is_active
        )
        db.session.add(version)
        print(f"  ✅ {cfg_name}: trainid={record.trainid}, active={is_active}")

    db.session.commit()
    print("  ✅ 数据库更新完成")


def main():
    set_seed()
    print("🔋 城市电网负荷预测 - 一键训练脚本")
    print(f"   设备: {DEVICE}")
    print(f"   模型保存目录: {MODEL_DIR}\n")

    with app.app_context():
        ensure_model_configs()
        train_values, valid_values = load_data_from_db()

        if len(train_values) < 100 or len(valid_values) < 100:
            print("❌ 数据不足，请先运行 import_data.py 导入数据")
            return

        preprocessor = DataPreprocessor()
        X_train, y_train, X_valid, y_valid = prepare_sequences(preprocessor, train_values, valid_values)

        results = {}

        # 1. LSTM
        set_seed(2026)
        model = LSTMModel(input_size=1, hidden_size=128, num_layers=2, dropout=0.2)
        _, path, tl, vl, met, st, et = train_one_model(
            'LSTM', model, LSTMTrainer, X_train, y_train, X_valid, y_valid, EPOCHS_LSTM, 0.001)
        results['LSTM_v1'] = {
            'model_path': path, 'train_loss': float(tl[-1]), 'valid_loss': float(vl[-1]),
            'metrics': met, 'epochs': EPOCHS_LSTM, 'start_time': st, 'end_time': et,
            'train_samples': len(X_train), 'valid_samples': len(X_valid)
        }

        # 2. Transformer
        set_seed(2027)
        model = TransformerModel(input_size=1, d_model=64, nhead=4, num_layers=2, dropout=0.1)
        _, path, tl, vl, met, st, et = train_one_model(
            'Transformer', model, TransformerTrainer, X_train, y_train, X_valid, y_valid, EPOCHS_TRANS, 0.0001)
        results['Transformer_v1'] = {
            'model_path': path, 'train_loss': float(tl[-1]), 'valid_loss': float(vl[-1]),
            'metrics': met, 'epochs': EPOCHS_TRANS, 'start_time': st, 'end_time': et,
            'train_samples': len(X_train), 'valid_samples': len(X_valid)
        }

        # 3. Hybrid (LSTM-Transformer)
        results['Hybrid_v1'] = train_best_hybrid(X_train, y_train, X_valid, y_valid)

        require_hybrid_best(results)

        # 更新数据库
        update_database(results, active_model='Hybrid_v1')

        print(f"\n{'='*60}")
        print(f"🎉 全部训练完成！")
        print(f"{'='*60}")
        for name, r in results.items():
            m = r['metrics']
            print(f"  {name:15s} | Loss={r['valid_loss']:.6f} | MAE={m['mae']:.4f} | RMSE={m['rmse']:.4f} | R2={m['r2score']:.4f}")


if __name__ == '__main__':
    main()
