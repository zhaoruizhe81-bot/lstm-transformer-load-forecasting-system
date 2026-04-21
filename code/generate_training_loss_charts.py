# 生成三种模型训练损失曲线，横纵坐标使用中文
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from flask import Flask

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.database import db, ModelConfig, TrainRecord, ModelVersion

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Songti SC', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:123456@localhost:3306/electric_load_forecasting_system_with_deep_learning_2026'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

MODEL_TITLES = {
    'LSTM_v1': 'LSTM模型训练损失曲线',
    'Transformer_v1': 'Transformer模型训练损失曲线',
    'Hybrid_v1': 'LSTM-Transformer混合模型训练损失曲线',
}

OUTPUT_NAMES = {
    'LSTM_v1': 'lstm_training_loss.png',
    'Transformer_v1': 'transformer_training_loss.png',
    'Hybrid_v1': 'hybrid_training_loss.png',
}


def latest_record(modelname):
    row = (
        db.session.query(ModelConfig, TrainRecord, ModelVersion)
        .join(TrainRecord, ModelConfig.configid == TrainRecord.configid)
        .join(ModelVersion, TrainRecord.trainid == ModelVersion.trainid)
        .filter(ModelConfig.modelname == modelname)
        .order_by(TrainRecord.endtime.desc())
        .first()
    )
    if not row:
        raise RuntimeError(f'未找到训练记录: {modelname}')
    return row[1]


def plot_loss(modelname, record):
    checkpoint = torch.load(record.modelpath, map_location='cpu')
    train_losses = checkpoint.get('train_losses', [])
    valid_losses = checkpoint.get('valid_losses', [])
    if not train_losses or not valid_losses:
        raise RuntimeError(f'{modelname} 的模型文件缺少训练损失记录: {record.modelpath}')

    epochs = list(range(1, len(train_losses) + 1))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_losses, label='训练损失', linewidth=2, color='#d62728')
    ax.plot(epochs, valid_losses, label='验证损失', linewidth=2, color='#9467bd')
    ax.set_xlabel('训练轮次', fontsize=12)
    ax.set_ylabel('损失值', fontsize=12)
    ax.set_title(MODEL_TITLES[modelname], fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs('images', exist_ok=True)
    output_path = os.path.join('images', OUTPUT_NAMES[modelname])
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ {MODEL_TITLES[modelname]} -> {output_path}')


def main():
    with app.app_context():
        for modelname in MODEL_TITLES:
            record = latest_record(modelname)
            plot_loss(modelname, record)


if __name__ == '__main__':
    main()
