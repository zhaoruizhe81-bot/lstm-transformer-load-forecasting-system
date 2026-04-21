# 直接调用service层（不走Flask HTTP）
import sys, os
_code_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _code_dir not in sys.path:
    sys.path.insert(0, _code_dir)

from flask import Flask
from models.database import db, User, LoadData, OperationLog, AlertRecord, SystemConfig

def success(data=None, message="操作成功"):
    return {'code': 200, 'message': message, 'data': data}

def error(message="操作失败", code=400):
    return {'code': code, 'message': message, 'data': None}

_DB_URI = "mysql+pymysql://root:123456@localhost:3306/electric_load_forecasting_system_with_deep_learning_2026"

_app = Flask(__name__)
_app.config['SQLALCHEMY_DATABASE_URI'] = _DB_URI
_app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
_app.config['SQLALCHEMY_ECHO'] = False
db.init_app(_app)

# 服务单例，懒加载（避免模块导入时触发 PyTorch/matplotlib 初始化造成递归）
_auth_svc = None
_data_svc = None
_model_svc = None
_predict_svc = None


class APIClient:
    # 每次调用都在 app context 内执行，解决 Streamlit 多线程问题
    def _call(self, func, *args, **kwargs):
        with _app.app_context():
            return func(*args, **kwargs)

    def _get_auth(self):
        global _auth_svc
        if _auth_svc is None:
            from services.auth_service import AuthService
            _auth_svc = AuthService()
        return _auth_svc

    def _get_data(self):
        global _data_svc
        if _data_svc is None:
            from services.data_service import DataService
            _data_svc = DataService()
        return _data_svc

    def _get_model(self):
        global _model_svc
        if _model_svc is None:
            from services.model_service import ModelService
            _model_svc = ModelService()
        return _model_svc

    def _get_predict(self):
        global _predict_svc
        if _predict_svc is None:
            from services.predict_service import PredictService
            _predict_svc = PredictService()
        return _predict_svc

    # ========== 认证 ==========
    def login(self, username, password):
        return self._call(self._get_auth().login, username, password)

    def reset_password(self, username, security_answer, new_password):
        return self._call(self._get_auth().reset_password, username, security_answer, new_password)

    def get_user_info(self, userid):
        return self._call(self._get_auth().get_user_info, userid)

    # ========== 数据管理 ==========
    def upload_data(self, data_list, userid):
        return self._call(self._get_data().upload_data, data_list, userid)

    def query_data(self, start_time=None, end_time=None, page=1, page_size=100):
        return self._call(self._get_data().query_data, start_time, end_time, page, page_size)

    def detect_outliers(self, start_time, end_time, method='zscore', threshold=3, userid=None):
        return self._call(self._get_data().detect_outliers, start_time, end_time, method, threshold, userid)

    def fill_missing(self, start_time, end_time, method='linear', userid=None):
        return self._call(self._get_data().fill_missing, start_time, end_time, method, userid)

    def normalize_data(self, start_time, end_time, method='minmax', userid=None):
        return self._call(self._get_data().normalize_data, start_time, end_time, method, userid)

    def correlation_analysis(self, start_time, end_time, userid=None):
        return self._call(self._get_data().correlation_analysis, start_time, end_time, userid)

    def visualize_load_curve(self, start_time, end_time):
        return self._call(self._get_data().visualize_load_curve, start_time, end_time)

    def get_data_statistics(self, start_time, end_time):
        return self._call(self._get_data().get_data_statistics, start_time, end_time)

    def get_data_date_range(self):
        """获取数据库中负荷数据的实际时间范围"""
        def _do():
            from sqlalchemy import func
            result = db.session.query(
                func.min(LoadData.recordtime),
                func.max(LoadData.recordtime)
            ).first()
            if result and result[0]:
                return success({
                    'min_date': result[0].strftime('%Y-%m-%d %H:%M:%S'),
                    'max_date': result[1].strftime('%Y-%m-%d %H:%M:%S'),
                })
            return success(None)
        try:
            return self._call(_do)
        except Exception as e:
            return error(f'获取时间范围失败: {str(e)}')

    # ========== 模型管理 ==========
    def create_model_config(self, modelname, modeltype, hyperparams, architecture, userid):
        return self._call(self._get_model().create_model_config, modelname, modeltype, hyperparams, architecture, userid)

    def get_model_configs(self, modeltype=None):
        return self._call(self._get_model().get_model_configs, modeltype)

    def train_model(self, configid, train_start, train_end, valid_start, valid_end,
                    epochs, batchsize, seq_length, userid):
        def _do():
            from algorithms.preprocessing import DataPreprocessor
            preprocessor = DataPreprocessor()
            train_list = LoadData.query.filter(
                LoadData.recordtime >= train_start, LoadData.recordtime <= train_end
            ).order_by(LoadData.recordtime).all()
            valid_list = LoadData.query.filter(
                LoadData.recordtime >= valid_start, LoadData.recordtime <= valid_end
            ).order_by(LoadData.recordtime).all()
            if not train_list or not valid_list:
                return error('训练或验证数据不足')
            train_norm = preprocessor.normalize_minmax([float(i.loadvalue) for i in train_list])
            valid_norm = preprocessor.normalize_minmax([float(i.loadvalue) for i in valid_list])
            X_train, y_train = preprocessor.create_sequences(train_norm, seq_length)
            X_valid, y_valid = preprocessor.create_sequences(valid_norm, seq_length)
            return self._get_model().train_model(configid, (X_train, y_train), (X_valid, y_valid), epochs, batchsize, userid)
        return self._call(_do)

    def get_train_records(self, configid=None):
        return self._call(self._get_model().get_train_records, configid)

    def create_model_version(self, trainid, versionnumber, versiondesc, performance, isactive=0):
        return self._call(self._get_model().create_model_version, trainid, versionnumber, versiondesc, performance, isactive)

    def get_model_versions(self, trainid=None):
        return self._call(self._get_model().get_model_versions, trainid)

    def activate_model_version(self, versionid):
        return self._call(self._get_model().activate_model_version, versionid)

    # ========== 预测管理 ==========
    def create_predict_task(self, taskname, versionid, predictstart, predictend, userid):
        return self._call(self._get_predict().create_predict_task, taskname, versionid, predictstart, predictend, userid)

    def get_predict_tasks(self, userid=None, status=None):
        return self._call(self._get_predict().get_predict_tasks, userid, status)

    def execute_predict_task(self, taskid, userid):
        return self._call(self._get_predict().execute_predict_task, taskid, userid)

    def get_predict_results(self, taskid):
        return self._call(self._get_predict().get_predict_results, taskid)

    def update_actual_values(self, taskid):
        return self._call(self._get_predict().update_actual_values, taskid)

    def calculate_error_metrics(self, taskid):
        return self._call(self._get_predict().calculate_error_metrics, taskid)

    def visualize_prediction(self, taskid):
        return self._call(self._get_predict().visualize_prediction, taskid)

    def export_predict_results(self, taskid):
        return self._call(self._get_predict().export_predict_results, taskid)

    # ========== 系统管理 ==========
    def get_all_users(self):
        def _do():
            users = User.query.all()
            result = [{'userid': u.userid, 'username': u.username, 'realname': u.realname,
                       'email': u.email, 'phone': u.phone, 'role': u.role, 'status': u.status,
                       'createtime': u.createtime.strftime('%Y-%m-%d %H:%M:%S') if u.createtime else None}
                      for u in users]
            return success({'users': result, 'count': len(result)})
        return self._call(_do)

    def create_user(self, username, password, realname, email, phone, role,
                    securityquestion, securityanswer, admin_userid):
        def _do():
            user = User(username=username, password=password, realname=realname,
                        email=email, phone=phone, role=role,
                        securityquestion=securityquestion, securityanswer=securityanswer, status=1)
            db.session.add(user)
            db.session.commit()
            db.session.add(OperationLog(userid=admin_userid, operation='create_user',
                                        module='admin', detail=f'创建用户: {username}'))
            db.session.commit()
            return success({'userid': user.userid}, '用户创建成功')
        try:
            return self._call(_do)
        except Exception as e:
            return error(f'用户创建失败: {str(e)}')

    def update_user_status(self, userid, status=None, role=None):
        def _do():
            user = User.query.get(userid)
            if not user:
                return error('用户不存在')
            if status is not None:
                user.status = status
            if role:
                user.role = role
            db.session.commit()
            return success(None, '用户信息更新成功')
        try:
            return self._call(_do)
        except Exception as e:
            return error(f'用户信息更新失败: {str(e)}')

    def delete_user(self, userid):
        def _do():
            user = User.query.get(userid)
            if not user:
                return error('用户不存在')
            db.session.delete(user)
            db.session.commit()
            return success(None, '用户删除成功')
        try:
            return self._call(_do)
        except Exception as e:
            return error(f'用户删除失败: {str(e)}')

    def get_operation_logs(self, page=1, page_size=50):
        def _do():
            logs = OperationLog.query.order_by(OperationLog.operationtime.desc()).paginate(
                page=page, per_page=page_size, error_out=False)
            result = [{'logid': l.logid, 'userid': l.userid, 'operation': l.operation,
                       'module': l.module, 'detail': l.detail, 'ipaddress': l.ipaddress,
                       'operationtime': l.operationtime.strftime('%Y-%m-%d %H:%M:%S') if l.operationtime else None}
                      for l in logs.items]
            return success({'logs': result, 'total': logs.total, 'page': page, 'page_size': page_size})
        try:
            return self._call(_do)
        except Exception as e:
            return error(f'获取操作日志失败: {str(e)}')

    def get_alert_records(self, status=None):
        def _do():
            q = AlertRecord.query
            if status is not None:
                q = q.filter_by(handlestatus=int(status))
            alerts = q.order_by(AlertRecord.alerttime.desc()).all()
            result = [{'alertid': a.alertid, 'alerttype': a.alerttype, 'alertlevel': a.alertlevel,
                       'alertmessage': a.alertmessage,
                       'alerttime': a.alerttime.strftime('%Y-%m-%d %H:%M:%S') if a.alerttime else None,
                       'handlestatus': a.handlestatus, 'handleuserid': a.handleuserid,
                       'handletime': a.handletime.strftime('%Y-%m-%d %H:%M:%S') if a.handletime else None}
                      for a in alerts]
            return success({'alerts': result, 'count': len(result)})
        try:
            return self._call(_do)
        except Exception as e:
            return error(f'获取告警记录失败: {str(e)}')

    def handle_alert(self, alertid, userid):
        from datetime import datetime
        def _do():
            a = AlertRecord.query.get(alertid)
            if not a:
                return error('告警记录不存在')
            a.handlestatus = 1
            a.handleuserid = userid
            a.handletime = datetime.now()
            db.session.commit()
            return success(None, '告警处理成功')
        try:
            return self._call(_do)
        except Exception as e:
            return error(f'告警处理失败: {str(e)}')

    def get_system_config(self):
        def _do():
            configs = SystemConfig.query.all()
            result = {c.configkey: {'value': c.configvalue, 'desc': c.configdesc} for c in configs}
            return success(result)
        try:
            return self._call(_do)
        except Exception as e:
            return error(f'获取系统配置失败: {str(e)}')

    def update_system_config(self, data):
        def _do():
            for key, value in data.items():
                c = SystemConfig.query.get(key)
                if c:
                    c.configvalue = value
            db.session.commit()
            return success(None, '系统配置更新成功')
        try:
            return self._call(_do)
        except Exception as e:
            return error(f'系统配置更新失败: {str(e)}')

