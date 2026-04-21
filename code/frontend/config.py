# Streamlit前端配置文件

# 页面配置
PAGE_TITLE = "电力负荷预测系统"
PAGE_ICON = "⚡"

# 角色权限配置
ROLE_PERMISSIONS = {
    'admin': ['用户管理', '数据管理', '模型管理', '预测管理', '系统管理', '日志查询'],
    'analyst': ['数据管理', '数据分析', '数据可视化'],
    'engineer': ['模型管理', '模型训练', '模型评估'],
    'business': ['预测管理', '历史查询', '结果导出']
}

# 角色中文名称
ROLE_NAMES = {
    'admin': '系统管理员',
    'analyst': '数据分析师',
    'engineer': '模型工程师',
    'business': '业务用户'
}

# 模型类型
MODEL_TYPES = {
    'lstm': 'LSTM模型',
    'transformer': 'Transformer模型',
    'hybrid': 'LSTM-Transformer混合模型'
}

# 异常检测方法
OUTLIER_METHODS = {
    'zscore': 'Z-score方法',
    'iqr': 'IQR方法',
    'boxplot': '箱型图方法'
}

# 缺失值填充方法
FILL_METHODS = {
    'linear': '线性插值',
    'spline': '样条插值'
}

