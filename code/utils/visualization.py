# 可视化工具类
import matplotlib
matplotlib.use('Agg')  # 非GUI后端
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import base64
from io import BytesIO

# 设置中文字体，兼容 Windows 和 macOS
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Songti SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def _set_xticks(ax, times, n_ticks=10):
    """从 times 列表里均匀挑 n_ticks 个点作为刻度，彻底避免密集重叠"""
    n = len(times)
    if n == 0:
        return
    ts = pd.to_datetime(times)
    span_days = (ts[-1] - ts[0]).total_seconds() / 86400

    # 根据跨度选显示格式
    if span_days <= 3:
        fmt = '%m-%d %H:%M'
    elif span_days <= 365:
        fmt = '%m-%d'
    else:
        fmt = '%Y-%m'

    # 均匀取最多 n_ticks 个索引
    step = max(1, n // n_ticks)
    indices = list(range(0, n, step))
    if indices[-1] != n - 1:
        indices.append(n - 1)

    ax.set_xticks(indices)
    ax.set_xticklabels([ts[i].strftime(fmt) for i in indices],
                       rotation=45, ha='right', fontsize=9)

def plot_to_base64(fig):
    """将matplotlib图表转换为base64编码"""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{image_base64}"

def plot_load_curve(times, values, title="负荷曲线"):
    """绘制负荷曲线"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(len(values)), values, linewidth=2, color='#1f77b4')
    ax.set_xlabel('时间', fontsize=12)
    ax.set_ylabel('负荷值 (MW)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    _set_xticks(ax, times)
    plt.tight_layout()
    return plot_to_base64(fig)

def plot_prediction_comparison(times, actual, predicted, title="预测对比"):
    """绘制预测值与实际值对比图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(actual))
    ax.plot(x, actual, label='实际值', linewidth=2, color='#2ca02c')
    ax.plot(x, predicted, label='预测值', linewidth=2, color='#ff7f0e', linestyle='--')
    ax.set_xlabel('时间', fontsize=12)
    ax.set_ylabel('负荷值 (MW)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    _set_xticks(ax, times)
    plt.tight_layout()
    return plot_to_base64(fig)

def plot_boxplot(data, title="箱型图异常检测"):
    """绘制箱型图"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(data, vert=True, patch_artist=True)
    ax.set_ylabel('负荷值 (MW)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    return plot_to_base64(fig)

def plot_correlation_matrix(corr_matrix, labels, title="相关性矩阵"):
    """绘制相关性热力图"""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                xticklabels=labels, yticklabels=labels, ax=ax, 
                center=0, vmin=-1, vmax=1)
    ax.set_title(title, fontsize=14, fontweight='bold')
    return plot_to_base64(fig)

def plot_training_history(epochs, train_loss, valid_loss, title="训练历史"):
    """绘制训练损失曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_loss, label='训练损失', linewidth=2, color='#d62728')
    ax.plot(epochs, valid_loss, label='验证损失', linewidth=2, color='#9467bd')
    ax.set_xlabel('训练轮次', fontsize=12)
    ax.set_ylabel('损失值', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    return plot_to_base64(fig)

def plot_scatter(x, y, xlabel="实际值", ylabel="预测值", title="散点图"):
    """绘制散点图"""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x, y, alpha=0.6, color='#1f77b4')
    
    # 添加对角线
    min_val = min(min(x), min(y))
    max_val = max(max(x), max(y))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    return plot_to_base64(fig)
