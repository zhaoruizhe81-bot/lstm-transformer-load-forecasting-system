# 认证服务
from models.database import db, User, OperationLog
from utils.response import success, error
from datetime import datetime

class AuthService:
    """认证服务类"""
    
    @staticmethod
    def login(username, password, ipaddress=''):
        """用户登录"""
        try:
            user = User.query.filter_by(username=username).first()
            
            if not user:
                return error('用户名不存在')
            
            if user.password != password:
                return error('密码错误')
            
            if user.status != 1:
                return error('账号已被禁用')
            
            # 记录登录日志
            log = OperationLog(
                userid=user.userid,
                operation='login',
                module='auth',
                detail=f'用户{username}登录系统',
                ipaddress=ipaddress
            )
            db.session.add(log)
            db.session.commit()
            
            return success({
                'userid': user.userid,
                'username': user.username,
                'realname': user.realname,
                'role': user.role,
                'email': user.email,
                'phone': user.phone
            }, '登录成功')
        
        except Exception as e:
            return error(f'登录失败: {str(e)}')
    
    @staticmethod
    def reset_password(username, security_answer, new_password):
        """重置密码"""
        try:
            user = User.query.filter_by(username=username).first()
            
            if not user:
                return error('用户名不存在')
            
            if user.securityanswer != security_answer:
                return error('安全问题答案错误')
            
            user.password = new_password
            user.updatetime = datetime.now()
            db.session.commit()
            
            return success(None, '密码重置成功')
        
        except Exception as e:
            db.session.rollback()
            return error(f'密码重置失败: {str(e)}')
    
    @staticmethod
    def change_password(userid, old_password, new_password):
        """修改密码"""
        try:
            user = User.query.get(userid)
            
            if not user:
                return error('用户不存在')
            
            if user.password != old_password:
                return error('原密码错误')
            
            user.password = new_password
            user.updatetime = datetime.now()
            db.session.commit()
            
            return success(None, '密码修改成功')
        
        except Exception as e:
            db.session.rollback()
            return error(f'密码修改失败: {str(e)}')
    
    @staticmethod
    def get_user_info(userid):
        """获取用户信息"""
        try:
            user = User.query.get(userid)
            
            if not user:
                return error('用户不存在')
            
            return success({
                'userid': user.userid,
                'username': user.username,
                'realname': user.realname,
                'email': user.email,
                'phone': user.phone,
                'role': user.role,
                'status': user.status,
                'securityquestion': user.securityquestion,
                'createtime': user.createtime.strftime('%Y-%m-%d %H:%M:%S') if user.createtime else None
            })
        
        except Exception as e:
            return error(f'获取用户信息失败: {str(e)}')
    
    @staticmethod
    def update_user_info(userid, realname=None, email=None, phone=None):
        """更新用户信息"""
        try:
            user = User.query.get(userid)
            
            if not user:
                return error('用户不存在')
            
            if realname:
                user.realname = realname
            if email:
                user.email = email
            if phone:
                user.phone = phone
            
            user.updatetime = datetime.now()
            db.session.commit()
            
            return success(None, '用户信息更新成功')
        
        except Exception as e:
            db.session.rollback()
            return error(f'用户信息更新失败: {str(e)}')

