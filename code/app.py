# Flask主入口文件
from flask import Flask, request, jsonify
from models.database import db
from services.auth_service import AuthService
from services.data_service import DataService
from services.model_service import ModelService
from services.predict_service import PredictService
from utils.response import success, error

app = Flask(__name__)

# 数据库配置
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:123456@localhost:3306/electric_load_forecasting_system_with_deep_learning_2026'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ECHO'] = False

# 初始化数据库
db.init_app(app)

# 实例化服务
auth_service = AuthService()
data_service = DataService()
model_service = ModelService()
predict_service = PredictService()

# ==================== 认证相关接口 ====================

@app.route('/api/auth/login', methods=['POST'])
def login():
    """用户登录"""
    data = request.json
    username = data.get('username')
    password = data.get('password')
    ipaddress = request.remote_addr
    
    result = auth_service.login(username, password, ipaddress)
    return jsonify(result)

@app.route('/api/auth/reset-password', methods=['POST'])
def reset_password():
    """重置密码"""
    data = request.json
    username = data.get('username')
    security_answer = data.get('security_answer')
    new_password = data.get('new_password')
    
    result = auth_service.reset_password(username, security_answer, new_password)
    return jsonify(result)

@app.route('/api/auth/change-password', methods=['POST'])
def change_password():
    """修改密码"""
    data = request.json
    userid = data.get('userid')
    old_password = data.get('old_password')
    new_password = data.get('new_password')
    
    result = auth_service.change_password(userid, old_password, new_password)
    return jsonify(result)

@app.route('/api/auth/user/<int:userid>', methods=['GET'])
def get_user_info(userid):
    """获取用户信息"""
    result = auth_service.get_user_info(userid)
    return jsonify(result)

@app.route('/api/auth/user/<int:userid>', methods=['PUT'])
def update_user_info(userid):
    """更新用户信息"""
    data = request.json
    result = auth_service.update_user_info(
        userid,
        data.get('realname'),
        data.get('email'),
        data.get('phone')
    )
    return jsonify(result)

# ==================== 数据管理接口 ====================

@app.route('/api/data/upload', methods=['POST'])
def upload_data():
    """上传负荷数据"""
    data = request.json
    data_list = data.get('data_list', [])
    userid = data.get('userid')
    
    result = data_service.upload_data(data_list, userid)
    return jsonify(result)

@app.route('/api/data/query', methods=['GET'])
def query_data():
    """查询负荷数据"""
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('page_size', 100))
    
    result = data_service.query_data(start_time, end_time, page, page_size)
    return jsonify(result)

@app.route('/api/data/detect-outliers', methods=['POST'])
def detect_outliers():
    """异常检测"""
    data = request.json
    start_time = data.get('start_time')
    end_time = data.get('end_time')
    method = data.get('method', 'zscore')
    threshold = data.get('threshold', 3)
    userid = data.get('userid')
    
    result = data_service.detect_outliers(start_time, end_time, method, threshold, userid)
    return jsonify(result)

@app.route('/api/data/fill-missing', methods=['POST'])
def fill_missing():
    """填充缺失值"""
    data = request.json
    start_time = data.get('start_time')
    end_time = data.get('end_time')
    method = data.get('method', 'linear')
    userid = data.get('userid')
    
    result = data_service.fill_missing(start_time, end_time, method, userid)
    return jsonify(result)

@app.route('/api/data/normalize', methods=['POST'])
def normalize_data():
    """数据归一化"""
    data = request.json
    start_time = data.get('start_time')
    end_time = data.get('end_time')
    method = data.get('method', 'minmax')
    userid = data.get('userid')
    
    result = data_service.normalize_data(start_time, end_time, method, userid)
    return jsonify(result)

@app.route('/api/data/correlation', methods=['POST'])
def correlation_analysis():
    """相关性分析"""
    data = request.json
    start_time = data.get('start_time')
    end_time = data.get('end_time')
    userid = data.get('userid')
    
    result = data_service.correlation_analysis(start_time, end_time, userid)
    return jsonify(result)

@app.route('/api/data/visualize', methods=['POST'])
def visualize_load_curve():
    """可视化负荷曲线"""
    data = request.json
    start_time = data.get('start_time')
    end_time = data.get('end_time')
    
    result = data_service.visualize_load_curve(start_time, end_time)
    return jsonify(result)

@app.route('/api/data/statistics', methods=['GET'])
def get_data_statistics():
    """获取数据统计信息"""
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')
    
    result = data_service.get_data_statistics(start_time, end_time)
    return jsonify(result)

# ==================== 模型管理接口 ====================

@app.route('/api/model/config', methods=['POST'])
def create_model_config():
    """创建模型配置"""
    data = request.json
    result = model_service.create_model_config(
        data.get('modelname'),
        data.get('modeltype'),
        data.get('hyperparams'),
        data.get('architecture'),
        data.get('userid')
    )
    return jsonify(result)

@app.route('/api/model/configs', methods=['GET'])
def get_model_configs():
    """获取模型配置列表"""
    modeltype = request.args.get('modeltype')
    result = model_service.get_model_configs(modeltype)
    return jsonify(result)

@app.route('/api/model/train', methods=['POST'])
def train_model():
    """训练模型"""
    data = request.json
    
    # 这里需要从数据库获取训练数据
    from models.database import LoadData
    from algorithms.preprocessing import DataPreprocessor
    
    start_time = data.get('train_start')
    end_time = data.get('train_end')
    valid_start = data.get('valid_start')
    valid_end = data.get('valid_end')
    
    # 获取训练数据
    train_data_list = LoadData.query.filter(
        LoadData.recordtime >= start_time,
        LoadData.recordtime <= end_time
    ).order_by(LoadData.recordtime).all()
    
    valid_data_list = LoadData.query.filter(
        LoadData.recordtime >= valid_start,
        LoadData.recordtime <= valid_end
    ).order_by(LoadData.recordtime).all()
    
    if not train_data_list or not valid_data_list:
        return jsonify(error('训练或验证数据不足'))
    
    # 数据预处理
    preprocessor = DataPreprocessor()
    train_values = [float(item.loadvalue) for item in train_data_list]
    valid_values = [float(item.loadvalue) for item in valid_data_list]
    
    # 归一化
    train_normalized = preprocessor.normalize_minmax(train_values)
    valid_normalized = preprocessor.normalize_minmax(valid_values)
    
    # 创建序列
    seq_length = data.get('seq_length', 24)
    X_train, y_train = preprocessor.create_sequences(train_normalized, seq_length)
    X_valid, y_valid = preprocessor.create_sequences(valid_normalized, seq_length)
    
    result = model_service.train_model(
        data.get('configid'),
        (X_train, y_train),
        (X_valid, y_valid),
        data.get('epochs', 100),
        data.get('batchsize', 32),
        data.get('userid')
    )
    return jsonify(result)

@app.route('/api/model/train-records', methods=['GET'])
def get_train_records():
    """获取训练记录"""
    configid = request.args.get('configid')
    result = model_service.get_train_records(configid)
    return jsonify(result)

@app.route('/api/model/version', methods=['POST'])
def create_model_version():
    """创建模型版本"""
    data = request.json
    result = model_service.create_model_version(
        data.get('trainid'),
        data.get('versionnumber'),
        data.get('versiondesc'),
        data.get('performance'),
        data.get('isactive', 0)
    )
    return jsonify(result)

@app.route('/api/model/versions', methods=['GET'])
def get_model_versions():
    """获取模型版本列表"""
    trainid = request.args.get('trainid')
    result = model_service.get_model_versions(trainid)
    return jsonify(result)

@app.route('/api/model/version/<int:versionid>/activate', methods=['PUT'])
def activate_model_version(versionid):
    """激活模型版本"""
    result = model_service.activate_model_version(versionid)
    return jsonify(result)

# ==================== 预测管理接口 ====================

@app.route('/api/predict/task', methods=['POST'])
def create_predict_task():
    """创建预测任务"""
    data = request.json
    result = predict_service.create_predict_task(
        data.get('taskname'),
        data.get('versionid'),
        data.get('predictstart'),
        data.get('predictend'),
        data.get('userid')
    )
    return jsonify(result)

@app.route('/api/predict/tasks', methods=['GET'])
def get_predict_tasks():
    """获取预测任务列表"""
    userid = request.args.get('userid')
    status = request.args.get('status')
    result = predict_service.get_predict_tasks(userid, status)
    return jsonify(result)

@app.route('/api/predict/task/<int:taskid>/execute', methods=['POST'])
def execute_predict_task(taskid):
    """执行预测任务"""
    data = request.json
    userid = data.get('userid')
    result = predict_service.execute_predict_task(taskid, userid)
    return jsonify(result)

@app.route('/api/predict/task/<int:taskid>/results', methods=['GET'])
def get_predict_results(taskid):
    """获取预测结果"""
    result = predict_service.get_predict_results(taskid)
    return jsonify(result)

@app.route('/api/predict/task/<int:taskid>/update-actual', methods=['POST'])
def update_actual_values(taskid):
    """更新实际值"""
    result = predict_service.update_actual_values(taskid)
    return jsonify(result)

@app.route('/api/predict/task/<int:taskid>/metrics', methods=['POST'])
def calculate_error_metrics(taskid):
    """计算误差指标"""
    result = predict_service.calculate_error_metrics(taskid)
    return jsonify(result)

@app.route('/api/predict/task/<int:taskid>/visualize', methods=['GET'])
def visualize_prediction(taskid):
    """可视化预测结果"""
    result = predict_service.visualize_prediction(taskid)
    return jsonify(result)

@app.route('/api/predict/task/<int:taskid>/export', methods=['GET'])
def export_predict_results(taskid):
    """导出预测结果"""
    result = predict_service.export_predict_results(taskid)
    return jsonify(result)

# ==================== 系统管理接口 ====================

@app.route('/api/admin/users', methods=['GET'])
def get_all_users():
    """获取所有用户（管理员）"""
    from models.database import User
    try:
        users = User.query.all()
        result = []
        for user in users:
            result.append({
                'userid': user.userid,
                'username': user.username,
                'realname': user.realname,
                'email': user.email,
                'phone': user.phone,
                'role': user.role,
                'status': user.status,
                'createtime': user.createtime.strftime('%Y-%m-%d %H:%M:%S') if user.createtime else None
            })
        return jsonify(success({'users': result, 'count': len(result)}))
    except Exception as e:
        return jsonify(error(f'获取用户列表失败: {str(e)}'))

@app.route('/api/admin/user', methods=['POST'])
def create_user():
    """创建用户（管理员）"""
    from models.database import User, OperationLog
    data = request.json
    try:
        user = User(
            username=data.get('username'),
            password=data.get('password', '123456'),
            realname=data.get('realname'),
            email=data.get('email'),
            phone=data.get('phone'),
            role=data.get('role'),
            securityquestion=data.get('securityquestion'),
            securityanswer=data.get('securityanswer'),
            status=data.get('status', 1)
        )
        db.session.add(user)
        db.session.commit()

        # 记录日志
        log = OperationLog(
            userid=data.get('admin_userid'),
            operation='create_user',
            module='admin',
            detail=f'创建用户: {user.username}'
        )
        db.session.add(log)
        db.session.commit()

        return jsonify(success({'userid': user.userid}, '用户创建成功'))
    except Exception as e:
        db.session.rollback()
        return jsonify(error(f'用户创建失败: {str(e)}'))

@app.route('/api/admin/user/<int:userid>', methods=['PUT'])
def update_user_status(userid):
    """更新用户状态（管理员）"""
    from models.database import User
    data = request.json
    try:
        user = User.query.get(userid)
        if not user:
            return jsonify(error('用户不存在'))

        if 'status' in data:
            user.status = data['status']
        if 'role' in data:
            user.role = data['role']

        db.session.commit()
        return jsonify(success(None, '用户信息更新成功'))
    except Exception as e:
        db.session.rollback()
        return jsonify(error(f'用户信息更新失败: {str(e)}'))

@app.route('/api/admin/user/<int:userid>', methods=['DELETE'])
def delete_user(userid):
    """删除用户（管理员）"""
    from models.database import User
    try:
        user = User.query.get(userid)
        if not user:
            return jsonify(error('用户不存在'))

        db.session.delete(user)
        db.session.commit()
        return jsonify(success(None, '用户删除成功'))
    except Exception as e:
        db.session.rollback()
        return jsonify(error(f'用户删除失败: {str(e)}'))

@app.route('/api/admin/logs', methods=['GET'])
def get_operation_logs():
    """获取操作日志（管理员）"""
    from models.database import OperationLog
    try:
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 50))

        logs = OperationLog.query.order_by(OperationLog.operationtime.desc()).paginate(
            page=page, per_page=page_size, error_out=False
        )

        result = []
        for log in logs.items:
            result.append({
                'logid': log.logid,
                'userid': log.userid,
                'operation': log.operation,
                'module': log.module,
                'detail': log.detail,
                'ipaddress': log.ipaddress,
                'operationtime': log.operationtime.strftime('%Y-%m-%d %H:%M:%S') if log.operationtime else None
            })

        return jsonify(success({
            'logs': result,
            'total': logs.total,
            'page': page,
            'page_size': page_size
        }))
    except Exception as e:
        return jsonify(error(f'获取操作日志失败: {str(e)}'))

@app.route('/api/admin/alerts', methods=['GET'])
def get_alert_records():
    """获取告警记录（管理员）"""
    from models.database import AlertRecord
    try:
        status = request.args.get('status')
        query = AlertRecord.query

        if status is not None:
            query = query.filter_by(handlestatus=int(status))

        alerts = query.order_by(AlertRecord.alerttime.desc()).all()

        result = []
        for alert in alerts:
            result.append({
                'alertid': alert.alertid,
                'alerttype': alert.alerttype,
                'alertlevel': alert.alertlevel,
                'alertmessage': alert.alertmessage,
                'alerttime': alert.alerttime.strftime('%Y-%m-%d %H:%M:%S') if alert.alerttime else None,
                'handlestatus': alert.handlestatus,
                'handleuserid': alert.handleuserid,
                'handletime': alert.handletime.strftime('%Y-%m-%d %H:%M:%S') if alert.handletime else None
            })

        return jsonify(success({'alerts': result, 'count': len(result)}))
    except Exception as e:
        return jsonify(error(f'获取告警记录失败: {str(e)}'))

@app.route('/api/admin/alert/<int:alertid>/handle', methods=['PUT'])
def handle_alert(alertid):
    """处理告警（管理员）"""
    from models.database import AlertRecord
    from datetime import datetime
    data = request.json
    try:
        alert = AlertRecord.query.get(alertid)
        if not alert:
            return jsonify(error('告警记录不存在'))

        alert.handlestatus = 1
        alert.handleuserid = data.get('userid')
        alert.handletime = datetime.now()
        db.session.commit()

        return jsonify(success(None, '告警处理成功'))
    except Exception as e:
        db.session.rollback()
        return jsonify(error(f'告警处理失败: {str(e)}'))

@app.route('/api/system/config', methods=['GET'])
def get_system_config():
    """获取系统配置"""
    from models.database import SystemConfig
    try:
        configs = SystemConfig.query.all()
        result = {}
        for config in configs:
            result[config.configkey] = {
                'value': config.configvalue,
                'desc': config.configdesc
            }
        return jsonify(success(result))
    except Exception as e:
        return jsonify(error(f'获取系统配置失败: {str(e)}'))

@app.route('/api/system/config', methods=['PUT'])
def update_system_config():
    """更新系统配置"""
    from models.database import SystemConfig
    data = request.json
    try:
        for key, value in data.items():
            config = SystemConfig.query.get(key)
            if config:
                config.configvalue = value
        db.session.commit()
        return jsonify(success(None, '系统配置更新成功'))
    except Exception as e:
        db.session.rollback()
        return jsonify(error(f'系统配置更新失败: {str(e)}'))

# ==================== 启动应用 ====================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4500, debug=True)

