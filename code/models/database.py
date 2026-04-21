# 数据库模型定义
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    """用户表"""
    __tablename__ = 'users'
    userid = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    realname = db.Column(db.String(50))
    email = db.Column(db.String(100))
    phone = db.Column(db.String(20))
    role = db.Column(db.Enum('admin', 'analyst', 'engineer', 'business'), nullable=False)
    securityquestion = db.Column(db.String(200))
    securityanswer = db.Column(db.String(200))
    status = db.Column(db.Integer, default=1)
    createtime = db.Column(db.DateTime, default=datetime.now)
    updatetime = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

class LoadData(db.Model):
    """负荷数据表"""
    __tablename__ = 'loaddata'
    dataid = db.Column(db.Integer, primary_key=True)
    recordtime = db.Column(db.DateTime, nullable=False)
    loadvalue = db.Column(db.Numeric(12, 2), nullable=False)
    temperature = db.Column(db.Numeric(5, 2))
    humidity = db.Column(db.Numeric(5, 2))
    holiday = db.Column(db.Integer, default=0)
    weekday = db.Column(db.Integer)
    datasource = db.Column(db.String(50))
    uploaduserid = db.Column(db.Integer)
    uploadtime = db.Column(db.DateTime, default=datetime.now)

class DataQuality(db.Model):
    """数据质量记录表"""
    __tablename__ = 'dataquality'
    qualityid = db.Column(db.Integer, primary_key=True)
    dataid = db.Column(db.Integer)
    checktime = db.Column(db.DateTime, default=datetime.now)
    checktype = db.Column(db.String(50))
    issuecount = db.Column(db.Integer, default=0)
    issuedetail = db.Column(db.Text)
    checkuserid = db.Column(db.Integer)

class DataProcess(db.Model):
    """数据处理历史表"""
    __tablename__ = 'dataprocess'
    processid = db.Column(db.Integer, primary_key=True)
    processtype = db.Column(db.String(50))
    processmethod = db.Column(db.String(100))
    inputdatarange = db.Column(db.String(200))
    outputdatarange = db.Column(db.String(200))
    processparams = db.Column(db.Text)
    processresult = db.Column(db.Text)
    processuserid = db.Column(db.Integer)
    processtime = db.Column(db.DateTime, default=datetime.now)

class ModelConfig(db.Model):
    """模型配置表"""
    __tablename__ = 'modelconfig'
    configid = db.Column(db.Integer, primary_key=True)
    modelname = db.Column(db.String(100), nullable=False)
    modeltype = db.Column(db.Enum('lstm', 'transformer', 'hybrid'), nullable=False)
    hyperparams = db.Column(db.Text)
    architecture = db.Column(db.Text)
    createuserid = db.Column(db.Integer)
    createtime = db.Column(db.DateTime, default=datetime.now)

class TrainRecord(db.Model):
    """训练记录表"""
    __tablename__ = 'trainrecord'
    trainid = db.Column(db.Integer, primary_key=True)
    configid = db.Column(db.Integer)
    traindata = db.Column(db.String(200))
    validdata = db.Column(db.String(200))
    epochs = db.Column(db.Integer)
    batchsize = db.Column(db.Integer)
    trainloss = db.Column(db.Numeric(10, 6))
    validloss = db.Column(db.Numeric(10, 6))
    trainstatus = db.Column(db.String(20))
    modelpath = db.Column(db.String(200))
    trainuserid = db.Column(db.Integer)
    starttime = db.Column(db.DateTime)
    endtime = db.Column(db.DateTime)

class ModelVersion(db.Model):
    """模型版本表"""
    __tablename__ = 'modelversion'
    versionid = db.Column(db.Integer, primary_key=True)
    trainid = db.Column(db.Integer)
    versionnumber = db.Column(db.String(50))
    versiondesc = db.Column(db.Text)
    performance = db.Column(db.Text)
    isactive = db.Column(db.Integer, default=0)
    createtime = db.Column(db.DateTime, default=datetime.now)

class PredictTask(db.Model):
    """预测任务表"""
    __tablename__ = 'predicttask'
    taskid = db.Column(db.Integer, primary_key=True)
    taskname = db.Column(db.String(100), nullable=False)
    versionid = db.Column(db.Integer)
    predictstart = db.Column(db.DateTime)
    predictend = db.Column(db.DateTime)
    taskstatus = db.Column(db.String(20))
    createuserid = db.Column(db.Integer)
    createtime = db.Column(db.DateTime, default=datetime.now)
    executetime = db.Column(db.DateTime)

class PredictResult(db.Model):
    """预测结果表"""
    __tablename__ = 'predictresult'
    resultid = db.Column(db.Integer, primary_key=True)
    taskid = db.Column(db.Integer)
    predicttime = db.Column(db.DateTime)
    predictvalue = db.Column(db.Numeric(12, 2))
    actualvalue = db.Column(db.Numeric(12, 2))
    createtime = db.Column(db.DateTime, default=datetime.now)

class ErrorMetric(db.Model):
    """误差指标表"""
    __tablename__ = 'errormetric'
    metricid = db.Column(db.Integer, primary_key=True)
    taskid = db.Column(db.Integer)
    mae = db.Column(db.Numeric(10, 4))
    rmse = db.Column(db.Numeric(10, 4))
    mape = db.Column(db.Numeric(10, 4))
    r2score = db.Column(db.Numeric(10, 6))
    calculatetime = db.Column(db.DateTime, default=datetime.now)

class OperationLog(db.Model):
    """操作日志表"""
    __tablename__ = 'operationlog'
    logid = db.Column(db.Integer, primary_key=True)
    userid = db.Column(db.Integer)
    operation = db.Column(db.String(100))
    module = db.Column(db.String(50))
    detail = db.Column(db.Text)
    ipaddress = db.Column(db.String(50))
    operationtime = db.Column(db.DateTime, default=datetime.now)

class SystemConfig(db.Model):
    """系统配置表"""
    __tablename__ = 'systemconfig'
    configkey = db.Column(db.String(100), primary_key=True)
    configvalue = db.Column(db.Text)
    configdesc = db.Column(db.String(200))
    updatetime = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

class AlertRecord(db.Model):
    """告警记录表"""
    __tablename__ = 'alertrecord'
    alertid = db.Column(db.Integer, primary_key=True)
    alerttype = db.Column(db.String(50))
    alertlevel = db.Column(db.String(20))
    alertmessage = db.Column(db.Text)
    alerttime = db.Column(db.DateTime, default=datetime.now)
    handlestatus = db.Column(db.Integer, default=0)
    handleuserid = db.Column(db.Integer)
    handletime = db.Column(db.DateTime)

