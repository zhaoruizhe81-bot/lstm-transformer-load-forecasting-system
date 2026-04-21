# 生成预测结果可视化图：① 预测值与实际值对比折线图  ② 预测散点图
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Songti SC', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from models.database import db, PredictTask, PredictResult, ErrorMetric

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:123456@localhost:3306/electric_load_forecasting_system_with_deep_learning_2026'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

os.makedirs('images', exist_ok=True)

with app.app_context():
    tasks = PredictTask.query.filter_by(taskstatus='completed').all()
    print(f"找到 {len(tasks)} 个已完成的预测任务\n")

    # ===== 图1: 预测值与实际值对比折线图（三个模型合一张图） =====
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    fig.suptitle('各模型预测值与实际值对比折线图', fontsize=18, fontweight='bold', y=0.98)

    colors = {'LSTM': '#e74c3c', 'Transformer': '#3498db', 'LSTM-Transformer': '#2ecc71'}
    model_names_display = {'LSTM_v1': 'LSTM', 'Transformer_v1': 'Transformer', 'Hybrid_v1': 'LSTM-Transformer'}
    model_order = {'LSTM': 0, 'Transformer': 1, 'LSTM-Transformer': 2}

    all_task_data = []
    for t in tasks:
        results = PredictResult.query.filter_by(taskid=t.taskid).order_by(PredictResult.predicttime).all()
        valid = [r for r in results if r.actualvalue is not None]
        if not valid:
            continue
        times = [r.predicttime for r in valid]
        actual = [float(r.actualvalue) for r in valid]
        predicted = [float(r.predictvalue) for r in valid]
        # 从任务名提取模型名
        model_key = t.taskname.split(' ')[0]
        display_name = model_names_display.get(model_key, model_key)
        em = ErrorMetric.query.filter_by(taskid=t.taskid).first()
        all_task_data.append({
            'taskid': t.taskid, 'model': display_name, 'model_key': model_key,
            'times': times, 'actual': actual, 'predicted': predicted, 'em': em
        })

    all_task_data.sort(key=lambda item: model_order.get(item['model'], 99))
    if not all_task_data:
        raise RuntimeError('没有可绘制的已完成预测任务，请先运行 update_predictions.py')

    best_model = None
    metrics_with_rmse = [td for td in all_task_data if td['em']]
    if metrics_with_rmse:
        best_model = min(metrics_with_rmse, key=lambda td: float(td['em'].rmse))['model']

    for idx, td in enumerate(all_task_data):
        ax = axes[idx]
        x = range(len(td['actual']))
        ax.plot(x, td['actual'], label='实际值', linewidth=1.5, color='#333333', alpha=0.8)
        color = colors.get(td['model'], '#666666')
        ax.plot(x, td['predicted'], label=f'{td["model"]} 预测值', linewidth=1.5, color=color, linestyle='--', alpha=0.85)
        ax.fill_between(x, td['actual'], td['predicted'], alpha=0.15, color=color)

        # 误差信息
        if td['em']:
            mae, rmse = float(td['em'].mae), float(td['em'].rmse)
            best_label = '，最优' if td['model'] == best_model else ''
            ax.set_title(f'{td["model"]}{best_label}  (MAE={mae:.2f}, RMSE={rmse:.2f})', fontsize=13, fontweight='bold')
        else:
            ax.set_title(td['model'], fontsize=13, fontweight='bold')

        ax.set_ylabel('负荷值', fontsize=11)
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)

        # X轴刻度
        n = len(td['times'])
        step = max(1, n // 12)
        tick_idx = list(range(0, n, step))
        if tick_idx[-1] != n - 1:
            tick_idx.append(n - 1)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([td['times'][i].strftime('%m-%d %H:%M') for i in tick_idx],
                           rotation=45, ha='right', fontsize=8)

    axes[-1].set_xlabel('时间', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('images/prediction_comparison.png', dpi=200, bbox_inches='tight')
    print("✅ 预测值与实际值对比折线图 -> images/prediction_comparison.png")

    # ===== 图2: 预测散点图（三个模型合一张图） =====
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle('各模型预测散点图（实际值 vs 预测值）', fontsize=16, fontweight='bold')

    for idx, td in enumerate(all_task_data):
        ax = axes2[idx]
        color = colors.get(td['model'], '#666666')
        ax.scatter(td['actual'], td['predicted'], alpha=0.4, s=12, color=color, label='预测点')

        # 对角线（理想预测线）
        all_vals = td['actual'] + td['predicted']
        vmin, vmax = min(all_vals), max(all_vals)
        margin = (vmax - vmin) * 0.05
        ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin],
                'k--', linewidth=1.5, alpha=0.6, label='理想预测线')

        # R2
        if td['em']:
            r2 = float(td['em'].r2score)
            ax.text(0.05, 0.92, f'R2 = {r2:.4f}', transform=ax.transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('实际值', fontsize=11)
        ax.set_ylabel('预测值', fontsize=11)
        best_label = '（最优）' if td['model'] == best_model else ''
        ax.set_title(f'{td["model"]}{best_label}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(vmin - margin, vmax + margin)
        ax.set_ylim(vmin - margin, vmax + margin)

    plt.tight_layout()
    plt.savefig('images/prediction_scatter.png', dpi=200, bbox_inches='tight')
    print("✅ 预测散点图 -> images/prediction_scatter.png")

    print("\n🎉 全部完成！图片保存在 images/ 文件夹下")
