# 统一响应工具类

def success(data=None, message="操作成功"):
    """成功响应"""
    return {
        'code': 200,
        'message': message,
        'data': data
    }

def error(message="操作失败", code=400):
    """错误响应"""
    return {
        'code': code,
        'message': message,
        'data': None
    }

