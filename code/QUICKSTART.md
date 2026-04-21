# 快速启动指南

## 环境要求
- Python 3.8+
- MySQL 5.7+
- 建议使用虚拟环境

## 启动步骤

### 1. 创建虚拟环境（可选但推荐）
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 导入数据库
打开Navicat或MySQL命令行，执行database.sql文件

或使用命令行：
```bash
mysql -u root -p < database.sql
```

### 4. 启动Flask服务
```bash
python app.py
```

服务启动后访问：http://localhost:5000

## 测试API

使用Postman或curl测试登录接口：
```bash
curl -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d "{\"username\":\"admin\",\"password\":\"123456\"}"
```

## 常见问题

### 1. 数据库连接失败
检查MySQL服务是否启动，用户名密码是否正确（root/123456）

### 2. 模块导入错误
确保在项目根目录下运行app.py

### 3. PyTorch安装问题
如果需要GPU支持，访问 https://pytorch.org 选择对应版本

### 4. 中文显示问题
确保安装了SimHei字体，或修改visualization.py中的字体设置

## 下一步

前端实现等待后续指令。

