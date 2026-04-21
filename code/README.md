# 电力负荷预测系统 - 后端实现说明

## 项目结构

```
code/
├── database.sql                          # 数据库SQL文件
├── app.py                                # Flask主入口文件
├── requirements.txt                      # Python依赖库
├── models/
│   └── database.py                       # 数据库模型定义
├── services/
│   ├── auth_service.py                   # 认证服务
│   ├── data_service.py                   # 数据管理服务
│   ├── model_service.py                  # 模型管理服务
│   ├── predict_service.py                # 预测服务
│   └── admin_service.py                  # 系统管理服务
├── algorithms/
│   ├── preprocessing.py                  # 数据预处理算法
│   ├── lstm_model.py                     # LSTM模型
│   ├── transformer_model.py              # Transformer模型
│   └── hybrid_model.py                   # LSTM-Transformer混合模型
└── utils/
    ├── response.py                       # 统一响应工具
    └── visualization.py                  # 可视化工具
```

## 安装步骤

### 1. 安装Python依赖
```bash
pip install -r requirements.txt
```

### 2. 导入数据库
在Navicat或MySQL命令行中执行：
```bash
mysql -u root -p < database.sql
```

### 3. 启动Flask服务
```bash
python app.py
```

服务将在 http://localhost:5000 启动

## API接口说明

### 认证接口
- POST /api/auth/login - 用户登录
- POST /api/auth/reset-password - 重置密码
- POST /api/auth/change-password - 修改密码
- GET /api/auth/user/<userid> - 获取用户信息
- PUT /api/auth/user/<userid> - 更新用户信息

### 数据管理接口
- POST /api/data/upload - 上传负荷数据
- GET /api/data/query - 查询负荷数据
- POST /api/data/detect-outliers - 异常检测
- POST /api/data/fill-missing - 填充缺失值
- POST /api/data/normalize - 数据归一化
- POST /api/data/correlation - 相关性分析
- POST /api/data/visualize - 可视化负荷曲线
- GET /api/data/statistics - 获取数据统计

### 模型管理接口
- POST /api/model/config - 创建模型配置
- GET /api/model/configs - 获取模型配置列表
- POST /api/model/train - 训练模型
- GET /api/model/train-records - 获取训练记录
- POST /api/model/version - 创建模型版本
- GET /api/model/versions - 获取模型版本列表
- PUT /api/model/version/<versionid>/activate - 激活模型版本

### 预测管理接口
- POST /api/predict/task - 创建预测任务
- GET /api/predict/tasks - 获取预测任务列表
- POST /api/predict/task/<taskid>/execute - 执行预测任务
- GET /api/predict/task/<taskid>/results - 获取预测结果
- POST /api/predict/task/<taskid>/update-actual - 更新实际值
- POST /api/predict/task/<taskid>/metrics - 计算误差指标
- GET /api/predict/task/<taskid>/visualize - 可视化预测结果
- GET /api/predict/task/<taskid>/export - 导出预测结果

### 系统管理接口
- GET /api/admin/users - 获取所有用户
- POST /api/admin/user - 创建用户
- PUT /api/admin/user/<userid> - 更新用户状态
- DELETE /api/admin/user/<userid> - 删除用户
- GET /api/admin/logs - 获取操作日志
- GET /api/admin/alerts - 获取告警记录
- PUT /api/admin/alert/<alertid>/handle - 处理告警
- GET /api/system/config - 获取系统配置
- PUT /api/system/config - 更新系统配置

## 测试账号

| 用户名 | 密码 | 角色 | 说明 |
|--------|------|------|------|
| admin | 123456 | admin | 系统管理员 |
| analyst01 | 123456 | analyst | 数据分析师 |
| engineer01 | 123456 | engineer | 模型工程师 |
| business01 | 123456 | business | 业务用户 |

## 核心功能实现

### 1. 数据预处理
- 异常检测：Z-score、IQR、箱型图
- 缺失值填充：线性插值、样条插值
- 数据归一化：Min-Max归一化
- 特征工程：相关性分析、PCA主成分分析

### 2. 深度学习模型
- LSTM模型：2层LSTM，隐藏层128单元
- Transformer模型：4层Transformer，8个注意力头
- 混合模型：LSTM+Transformer结合

### 3. 模型训练
- 支持自定义超参数
- 训练过程可视化
- 模型版本管理
- 性能指标评估（MAE、RMSE、MAPE、R2）

### 4. 负荷预测
- 短期负荷预测（小时级）
- 预测结果可视化
- 预测误差分析
- 结果导出功能

## 注意事项

1. 确保MySQL服务已启动
2. 数据库连接配置：localhost:3306，用户名root，密码123456
3. 模型文件保存在./models目录
4. 首次训练模型需要较长时间，建议使用GPU加速
5. 图表生成使用base64编码返回，可直接在前端显示

## 依赖说明

- Flask：Web框架
- Flask-SQLAlchemy：ORM数据库操作
- PyMySQL：MySQL数据库驱动
- PyTorch：深度学习框架
- NumPy/Pandas：数据处理
- Scikit-learn：机器学习工具
- Matplotlib/Seaborn：数据可视化

## 开发完成情况

✅ 第一步：数据库设计完成
✅ 第二步：Flask后端实现完成

所有核心功能已实现，包括：
- 用户认证与权限管理
- 负荷数据管理与预处理
- 深度学习模型训练
- 负荷预测与结果分析
- 系统管理与日志记录

第三步前端实现等待后续指令。

