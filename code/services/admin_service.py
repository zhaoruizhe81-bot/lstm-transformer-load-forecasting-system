# 系统管理服务
from models.database import db, User, OperationLog, AlertRecord, SystemConfig
from utils.response import success, error
from datetime import datetime

class AdminService:
    """系统管理服务类"""
    
    @staticmethod
    def get_system_statistics():
        """获取系统统计信息"""
        try:
            from models.database import LoadData, ModelConfig, PredictTask
            
            stats = {
                'user_count': User.query.count(),
                'data_count': LoadData.query.count(),
                'model_count': ModelConfig.query.count(),
                'task_count': PredictTask.query.count(),
                'active_users': User.query.filter_by(status=1).count(),
                'pending_tasks': PredictTask.query.filter_by(taskstatus='pending').count(),
                'unhandled_alerts': AlertRecord.query.filter_by(handlestatus=0).count()
            }
            
            return success(stats, '系统统计信息获取成功')
        
        except Exception as e:
            return error(f'获取系统统计信息失败: {str(e)}')
    
    @staticmethod
    def create_alert(alerttype, alertlevel, alertmessage):
        """创建告警"""
        try:
            alert = AlertRecord(
                alerttype=alerttype,
                alertlevel=alertlevel,
                alertmessage=alertmessage
            )
            db.session.add(alert)
            db.session.commit()
            
            return success({'alertid': alert.alertid}, '告警创建成功')
        
        except Exception as e:
            db.session.rollback()
            return error(f'告警创建失败: {str(e)}')

