-- ============================================================
-- 电力负荷预测系统数据库
-- Database: electric_load_forecasting_system_with_deep_learning_2026
-- ============================================================

DROP DATABASE IF EXISTS electric_load_forecasting_system_with_deep_learning_2026;
CREATE DATABASE electric_load_forecasting_system_with_deep_learning_2026 DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE electric_load_forecasting_system_with_deep_learning_2026;

-- ============================================================
-- 用户表
-- ============================================================
CREATE TABLE users (
  userid INT PRIMARY KEY AUTO_INCREMENT COMMENT '用户ID',
  username VARCHAR(50) NOT NULL UNIQUE COMMENT '用户名',
  password VARCHAR(100) NOT NULL COMMENT '密码明文',
  realname VARCHAR(50) COMMENT '真实姓名',
  email VARCHAR(100) COMMENT '邮箱',
  phone VARCHAR(20) COMMENT '手机号',
  role ENUM('admin', 'analyst', 'engineer', 'business') NOT NULL COMMENT '角色:admin系统管理员/analyst数据分析师/engineer模型工程师/business业务用户',
  securityquestion VARCHAR(200) COMMENT '安全问题',
  securityanswer VARCHAR(200) COMMENT '安全答案',
  status TINYINT DEFAULT 1 COMMENT '状态:1启用/0禁用',
  createtime DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  updatetime DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间'
) COMMENT='用户表';

-- ============================================================
-- 负荷数据表
-- ============================================================
CREATE TABLE loaddata (
  dataid INT PRIMARY KEY AUTO_INCREMENT COMMENT '数据ID',
  recordtime DATETIME NOT NULL COMMENT '记录时间',
  loadvalue DECIMAL(12,2) NOT NULL COMMENT '负荷值(MW)',
  temperature DECIMAL(5,2) COMMENT '温度(℃)',
  humidity DECIMAL(5,2) COMMENT '湿度(%)',
  holiday TINYINT DEFAULT 0 COMMENT '是否节假日:1是/0否',
  weekday TINYINT COMMENT '星期几:1-7',
  datasource VARCHAR(50) COMMENT '数据来源',
  uploaduserid INT COMMENT '上传用户ID',
  uploadtime DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '上传时间',
  INDEX idx_recordtime (recordtime),
  INDEX idx_uploaduserid (uploaduserid)
) COMMENT='负荷数据表';

-- ============================================================
-- 数据质量记录表
-- ============================================================
CREATE TABLE dataquality (
  qualityid INT PRIMARY KEY AUTO_INCREMENT COMMENT '质量记录ID',
  dataid INT COMMENT '数据ID',
  checktime DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '检查时间',
  checktype VARCHAR(50) COMMENT '检查类型:missing缺失值/outlier异常值/duplicate重复值',
  issuecount INT DEFAULT 0 COMMENT '问题数量',
  issuedetail TEXT COMMENT '问题详情JSON',
  checkuserid INT COMMENT '检查用户ID',
  INDEX idx_dataid (dataid)
) COMMENT='数据质量记录表';

-- ============================================================
-- 数据处理历史表
-- ============================================================
CREATE TABLE dataprocess (
  processid INT PRIMARY KEY AUTO_INCREMENT COMMENT '处理ID',
  processtype VARCHAR(50) COMMENT '处理类型:fillmissing填充缺失值/removeoutlier去除异常值/normalize归一化/feature特征工程',
  processmethod VARCHAR(100) COMMENT '处理方法',
  inputdatarange VARCHAR(200) COMMENT '输入数据范围',
  outputdatarange VARCHAR(200) COMMENT '输出数据范围',
  processparams TEXT COMMENT '处理参数JSON',
  processresult TEXT COMMENT '处理结果JSON',
  processuserid INT COMMENT '处理用户ID',
  processtime DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '处理时间',
  INDEX idx_processuserid (processuserid)
) COMMENT='数据处理历史表';

-- ============================================================
-- 模型配置表
-- ============================================================
CREATE TABLE modelconfig (
  configid INT PRIMARY KEY AUTO_INCREMENT COMMENT '配置ID',
  modelname VARCHAR(100) NOT NULL COMMENT '模型名称',
  modeltype ENUM('lstm', 'transformer', 'hybrid') NOT NULL COMMENT '模型类型:lstm/transformer/hybrid混合',
  hyperparams TEXT COMMENT '超参数JSON',
  architecture TEXT COMMENT '模型架构描述',
  createuserid INT COMMENT '创建用户ID',
  createtime DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  INDEX idx_createuserid (createuserid)
) COMMENT='模型配置表';

-- ============================================================
-- 训练记录表
-- ============================================================
CREATE TABLE trainrecord (
  trainid INT PRIMARY KEY AUTO_INCREMENT COMMENT '训练ID',
  configid INT COMMENT '配置ID',
  traindata VARCHAR(200) COMMENT '训练数据范围',
  validdata VARCHAR(200) COMMENT '验证数据范围',
  epochs INT COMMENT '训练轮数',
  batchsize INT COMMENT '批次大小',
  trainloss DECIMAL(10,6) COMMENT '训练损失',
  validloss DECIMAL(10,6) COMMENT '验证损失',
  trainstatus VARCHAR(20) COMMENT '训练状态:running运行中/completed完成/failed失败',
  modelpath VARCHAR(200) COMMENT '模型文件路径',
  trainuserid INT COMMENT '训练用户ID',
  starttime DATETIME COMMENT '开始时间',
  endtime DATETIME COMMENT '结束时间',
  INDEX idx_configid (configid),
  INDEX idx_trainuserid (trainuserid)
) COMMENT='训练记录表';

-- ============================================================
-- 模型版本表
-- ============================================================
CREATE TABLE modelversion (
  versionid INT PRIMARY KEY AUTO_INCREMENT COMMENT '版本ID',
  trainid INT COMMENT '训练ID',
  versionnumber VARCHAR(50) COMMENT '版本号',
  versiondesc TEXT COMMENT '版本描述',
  performance TEXT COMMENT '性能指标JSON',
  isactive TINYINT DEFAULT 0 COMMENT '是否激活:1是/0否',
  createtime DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  INDEX idx_trainid (trainid)
) COMMENT='模型版本表';

-- ============================================================
-- 预测任务表
-- ============================================================
CREATE TABLE predicttask (
  taskid INT PRIMARY KEY AUTO_INCREMENT COMMENT '任务ID',
  taskname VARCHAR(100) NOT NULL COMMENT '任务名称',
  versionid INT COMMENT '模型版本ID',
  predictstart DATETIME COMMENT '预测开始时间',
  predictend DATETIME COMMENT '预测结束时间',
  taskstatus VARCHAR(20) COMMENT '任务状态:pending待执行/running运行中/completed完成/failed失败',
  createuserid INT COMMENT '创建用户ID',
  createtime DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  executetime DATETIME COMMENT '执行时间',
  INDEX idx_versionid (versionid),
  INDEX idx_createuserid (createuserid)
) COMMENT='预测任务表';

-- ============================================================
-- 预测结果表
-- ============================================================
CREATE TABLE predictresult (
  resultid INT PRIMARY KEY AUTO_INCREMENT COMMENT '结果ID',
  taskid INT COMMENT '任务ID',
  predicttime DATETIME COMMENT '预测时间点',
  predictvalue DECIMAL(12,2) COMMENT '预测值',
  actualvalue DECIMAL(12,2) COMMENT '实际值',
  createtime DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  INDEX idx_taskid (taskid),
  INDEX idx_predicttime (predicttime)
) COMMENT='预测结果表';

-- ============================================================
-- 误差指标表
-- ============================================================
CREATE TABLE errormetric (
  metricid INT PRIMARY KEY AUTO_INCREMENT COMMENT '指标ID',
  taskid INT COMMENT '任务ID',
  mae DECIMAL(10,4) COMMENT '平均绝对误差',
  rmse DECIMAL(10,4) COMMENT '均方根误差',
  mape DECIMAL(10,4) COMMENT '平均绝对百分比误差',
  r2score DECIMAL(10,6) COMMENT 'R2决定系数',
  calculatetime DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '计算时间',
  INDEX idx_taskid (taskid)
) COMMENT='误差指标表';

-- ============================================================
-- 操作日志表
-- ============================================================
CREATE TABLE operationlog (
  logid INT PRIMARY KEY AUTO_INCREMENT COMMENT '日志ID',
  userid INT COMMENT '用户ID',
  operation VARCHAR(100) COMMENT '操作类型',
  module VARCHAR(50) COMMENT '模块名称',
  detail TEXT COMMENT '操作详情',
  ipaddress VARCHAR(50) COMMENT 'IP地址',
  operationtime DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '操作时间',
  INDEX idx_userid (userid),
  INDEX idx_operationtime (operationtime)
) COMMENT='操作日志表';

-- ============================================================
-- 系统配置表
-- ============================================================
CREATE TABLE systemconfig (
  configkey VARCHAR(100) PRIMARY KEY COMMENT '配置键',
  configvalue TEXT COMMENT '配置值',
  configdesc VARCHAR(200) COMMENT '配置描述',
  updatetime DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间'
) COMMENT='系统配置表';

-- ============================================================
-- 告警记录表
-- ============================================================
CREATE TABLE alertrecord (
  alertid INT PRIMARY KEY AUTO_INCREMENT COMMENT '告警ID',
  alerttype VARCHAR(50) COMMENT '告警类型:dataanomaly数据异常/modelfailure模型失败/systemerror系统错误',
  alertlevel VARCHAR(20) COMMENT '告警级别:info信息/warning警告/error错误/critical严重',
  alertmessage TEXT COMMENT '告警消息',
  alerttime DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '告警时间',
  handlestatus TINYINT DEFAULT 0 COMMENT '处理状态:0未处理/1已处理',
  handleuserid INT COMMENT '处理用户ID',
  handletime DATETIME COMMENT '处理时间',
  INDEX idx_alerttime (alerttime),
  INDEX idx_handlestatus (handlestatus)
) COMMENT='告警记录表';

-- ============================================================
-- 插入测试数据
-- ============================================================

-- 插入用户数据
INSERT INTO users (username, password, realname, email, phone, role, securityquestion, securityanswer, status) VALUES
('admin', '123456', '系统管理员', 'admin@electric.com', '13800000001', 'admin', '你的出生地是哪里?', '北京', 1),
('analyst01', '123456', '张数据', 'analyst01@electric.com', '13800000002', 'analyst', '你的母校是哪里?', '清华大学', 1),
('analyst02', '123456', '李分析', 'analyst02@electric.com', '13800000003', 'analyst', '你的宠物叫什么?', '小白', 1),
('engineer01', '123456', '王工程', 'engineer01@electric.com', '13800000004', 'engineer', '你最喜欢的城市?', '上海', 1),
('engineer02', '123456', '赵模型', 'engineer02@electric.com', '13800000005', 'engineer', '你的第一份工作?', '算法工程师', 1),
('business01', '123456', '刘业务', 'business01@electric.com', '13800000006', 'business', '你的爱好是什么?', '阅读', 1),
('business02', '123456', '陈用户', 'business02@electric.com', '13800000007', 'business', '你的座右铭?', '精益求精', 1);

-- 插入2016年7月~2016年9月负荷数据（ETTh1数据集时间范围内的示例数据）
INSERT INTO loaddata (recordtime, loadvalue, temperature, humidity, holiday, weekday, datasource, uploaduserid) VALUES
-- 2016年7月数据
('2016-07-01 00:00:00', 7520.50, 32.5, 75.0, 0, 5, 'ett_small_etth1', 2),
('2016-07-01 06:00:00', 6230.80, 28.8, 72.0, 0, 5, 'ett_small_etth1', 2),
('2016-07-01 12:00:00', 8850.30, 35.3, 68.0, 0, 5, 'ett_small_etth1', 2),
('2016-07-01 18:00:00', 8120.60, 33.5, 70.0, 0, 5, 'ett_small_etth1', 2),
('2016-07-02 00:00:00', 7380.20, 31.2, 74.0, 0, 6, 'ett_small_etth1', 2),
('2016-07-02 06:00:00', 6450.70, 29.0, 71.0, 0, 6, 'ett_small_etth1', 2),
('2016-07-02 12:00:00', 9020.40, 36.5, 66.0, 0, 6, 'ett_small_etth1', 2),
('2016-07-02 18:00:00', 8350.90, 34.5, 69.0, 0, 6, 'ett_small_etth1', 2),
('2016-07-03 00:00:00', 7150.80, 30.0, 76.0, 0, 7, 'ett_small_etth1', 2),
('2016-07-03 06:00:00', 6120.30, 27.5, 73.0, 0, 7, 'ett_small_etth1', 2),
('2016-07-03 12:00:00', 8780.50, 34.8, 67.0, 0, 7, 'ett_small_etth1', 2),
('2016-07-03 18:00:00', 7950.20, 32.2, 70.0, 0, 7, 'ett_small_etth1', 2),
('2016-07-04 00:00:00', 6980.60, 29.5, 75.0, 0, 1, 'ett_small_etth1', 2),
('2016-07-04 06:00:00', 6650.40, 28.8, 72.0, 0, 1, 'ett_small_etth1', 2),
('2016-07-04 12:00:00', 9120.70, 37.0, 65.0, 0, 1, 'ett_small_etth1', 2),
('2016-07-04 18:00:00', 8480.30, 35.5, 68.0, 0, 1, 'ett_small_etth1', 2),
('2016-07-05 00:00:00', 7220.90, 30.8, 74.0, 0, 2, 'ett_small_etth1', 2),
('2016-07-05 06:00:00', 6680.50, 29.5, 71.0, 0, 2, 'ett_small_etth1', 2),
('2016-07-05 12:00:00', 9250.20, 37.2, 64.0, 0, 2, 'ett_small_etth1', 2),
('2016-07-05 18:00:00', 8680.80, 35.3, 67.0, 0, 2, 'ett_small_etth1', 2),
('2016-07-06 00:00:00', 7320.30, 31.8, 73.0, 0, 3, 'ett_small_etth1', 2),
('2016-07-06 06:00:00', 6780.60, 29.5, 70.0, 0, 3, 'ett_small_etth1', 2),
('2016-07-06 12:00:00', 9380.40, 38.0, 63.0, 0, 3, 'ett_small_etth1', 2),
('2016-07-06 18:00:00', 8820.70, 36.5, 66.0, 0, 3, 'ett_small_etth1', 2),
('2016-07-07 00:00:00', 7450.80, 32.8, 72.0, 0, 4, 'ett_small_etth1', 2),
('2016-07-07 06:00:00', 6920.40, 30.5, 69.0, 0, 4, 'ett_small_etth1', 2),
('2016-07-07 12:00:00', 9520.90, 38.2, 62.0, 0, 4, 'ett_small_etth1', 2),
('2016-07-07 18:00:00', 8950.30, 36.5, 65.0, 0, 4, 'ett_small_etth1', 2),
-- 2016年8月数据
('2016-08-01 00:00:00', 7350.20, 31.5, 74.0, 0, 1, 'ett_small_etth1', 2),
('2016-08-01 06:00:00', 6580.70, 29.8, 71.0, 0, 1, 'ett_small_etth1', 2),
('2016-08-01 12:00:00', 9120.50, 36.5, 66.0, 0, 1, 'ett_small_etth1', 2),
('2016-08-01 18:00:00', 8580.90, 34.2, 69.0, 0, 1, 'ett_small_etth1', 2),
('2016-08-02 00:00:00', 7480.60, 32.2, 73.0, 0, 2, 'ett_small_etth1', 2),
('2016-08-02 06:00:00', 6720.30, 30.5, 70.0, 0, 2, 'ett_small_etth1', 2),
('2016-08-02 12:00:00', 9350.80, 37.8, 65.0, 0, 2, 'ett_small_etth1', 2),
('2016-08-02 18:00:00', 8790.40, 35.5, 68.0, 0, 2, 'ett_small_etth1', 2),
('2016-08-03 00:00:00', 7590.50, 33.0, 72.0, 0, 3, 'ett_small_etth1', 2),
('2016-08-03 06:00:00', 6850.90, 31.2, 69.0, 0, 3, 'ett_small_etth1', 2),
('2016-08-03 12:00:00', 9480.20, 38.5, 64.0, 0, 3, 'ett_small_etth1', 2),
('2016-08-03 18:00:00', 8920.60, 36.8, 67.0, 0, 3, 'ett_small_etth1', 2),
('2016-08-04 00:00:00', 7650.80, 33.8, 71.0, 0, 4, 'ett_small_etth1', 2),
('2016-08-04 06:00:00', 6950.40, 31.0, 68.0, 0, 4, 'ett_small_etth1', 2),
('2016-08-04 12:00:00', 9620.70, 39.2, 63.0, 0, 4, 'ett_small_etth1', 2),
('2016-08-04 18:00:00', 9050.30, 37.5, 66.0, 0, 4, 'ett_small_etth1', 2),
('2016-08-05 00:00:00', 7720.30, 34.5, 70.0, 0, 5, 'ett_small_etth1', 2),
('2016-08-05 06:00:00', 7020.80, 31.8, 67.0, 0, 5, 'ett_small_etth1', 2),
('2016-08-05 12:00:00', 9750.50, 39.0, 62.0, 0, 5, 'ett_small_etth1', 2),
('2016-08-05 18:00:00', 9180.90, 37.2, 65.0, 0, 5, 'ett_small_etth1', 2),
-- 2016年9月数据
('2016-09-01 00:00:00', 6580.70, 26.5, 68.0, 0, 4, 'ett_small_etth1', 2),
('2016-09-01 06:00:00', 5820.40, 24.2, 65.0, 0, 4, 'ett_small_etth1', 2),
('2016-09-01 12:00:00', 8250.90, 30.8, 60.0, 0, 4, 'ett_small_etth1', 2),
('2016-09-01 18:00:00', 7620.50, 28.5, 63.0, 0, 4, 'ett_small_etth1', 2),
('2016-09-02 00:00:00', 6420.30, 25.2, 67.0, 0, 5, 'ett_small_etth1', 2),
('2016-09-02 06:00:00', 5680.80, 23.0, 64.0, 0, 5, 'ett_small_etth1', 2),
('2016-09-02 12:00:00', 8080.60, 29.5, 59.0, 0, 5, 'ett_small_etth1', 2),
('2016-09-02 18:00:00', 7420.20, 27.2, 62.0, 0, 5, 'ett_small_etth1', 2),
('2016-09-03 00:00:00', 6330.90, 24.0, 66.0, 0, 6, 'ett_small_etth1', 2),
('2016-09-03 06:00:00', 5590.50, 22.8, 63.0, 0, 6, 'ett_small_etth1', 2),
('2016-09-03 12:00:00', 7920.80, 28.2, 58.0, 0, 6, 'ett_small_etth1', 2),
('2016-09-03 18:00:00', 7250.40, 26.0, 61.0, 0, 6, 'ett_small_etth1', 2),
('2016-09-04 00:00:00', 6220.60, 23.8, 65.0, 0, 7, 'ett_small_etth1', 2),
('2016-09-04 06:00:00', 5480.20, 21.5, 62.0, 0, 7, 'ett_small_etth1', 2),
('2016-09-04 12:00:00', 7750.30, 27.0, 57.0, 0, 7, 'ett_small_etth1', 2),
('2016-09-04 18:00:00', 7080.70, 25.8, 60.0, 0, 7, 'ett_small_etth1', 2),
('2016-09-05 00:00:00', 6510.40, 25.5, 64.0, 0, 1, 'ett_small_etth1', 2),
('2016-09-05 06:00:00', 5870.90, 23.2, 61.0, 0, 1, 'ett_small_etth1', 2),
('2016-09-05 12:00:00', 8180.50, 29.8, 56.0, 0, 1, 'ett_small_etth1', 2),
('2016-09-05 18:00:00', 7510.30, 27.5, 59.0, 0, 1, 'ett_small_etth1', 2),
('2016-09-10 00:00:00', 6350.20, 24.0, 63.0, 0, 6, 'ett_small_etth1', 2),
('2016-09-10 06:00:00', 5720.80, 22.8, 60.0, 0, 6, 'ett_small_etth1', 2),
('2016-09-10 12:00:00', 7920.40, 28.5, 55.0, 0, 6, 'ett_small_etth1', 2),
('2016-09-10 18:00:00', 7250.90, 26.2, 58.0, 0, 6, 'ett_small_etth1', 2),
('2016-09-14 00:00:00', 6480.50, 25.5, 62.0, 0, 3, 'ett_small_etth1', 2),
('2016-09-14 06:00:00', 5850.30, 23.0, 59.0, 0, 3, 'ett_small_etth1', 2),
('2016-09-14 12:00:00', 8050.80, 29.2, 54.0, 0, 3, 'ett_small_etth1', 2),
('2016-09-14 18:00:00', 7380.60, 27.5, 57.0, 0, 3, 'ett_small_etth1', 2);

-- 插入数据质量记录
INSERT INTO dataquality (dataid, checktype, issuecount, issuedetail, checkuserid) VALUES
(15, 'outlier', 1, '{"method":"zscore","threshold":3,"outliers":[15]}', 2),
(28, 'missing', 2, '{"fields":["temperature","humidity"]}', 2),
(45, 'duplicate', 1, '{"duplicate_ids":[44,45]}', 3);

-- 插入数据处理历史
INSERT INTO dataprocess (processtype, processmethod, inputdatarange, outputdatarange, processparams, processresult, processuserid) VALUES
('fillmissing', 'linear_interpolation', '2016-07-01 to 2016-07-31', '2016-07-01 to 2016-07-31', '{"method":"linear"}', '{"filled_count":5,"success":true}', 2),
('removeoutlier', 'zscore', '2016-08-01 to 2016-08-31', '2016-08-01 to 2016-08-31', '{"threshold":3}', '{"removed_count":3,"success":true}', 2),
('normalize', 'minmax', '2016-07-01 to 2016-09-14', '2016-07-01 to 2016-09-14', '{"min":0,"max":1}', '{"normalized_count":248,"success":true}', 3),
('feature', 'correlation_analysis', '2016-07-01 to 2016-09-14', 'N/A', '{"features":["temperature","humidity","weekday"]}', '{"correlations":{"temperature":0.85,"humidity":-0.32,"weekday":0.15}}', 3);

-- 插入模型配置
INSERT INTO modelconfig (modelname, modeltype, hyperparams, architecture, createuserid) VALUES
('LSTM_v1', 'lstm', '{"hidden_size":128,"num_layers":2,"dropout":0.2,"learning_rate":0.001}', '2层LSTM,隐藏层128单元', 4),
('Transformer_v1', 'transformer', '{"d_model":64,"nhead":4,"num_layers":2,"dropout":0.1,"learning_rate":0.0001}', '2层Transformer,4个注意力头', 4),
('Hybrid_v1', 'hybrid', '{"lstm_hidden":64,"transformer_layers":2,"nhead":4,"dropout":0.15,"learning_rate":0.0005}', 'LSTM+Transformer混合模型', 5);

-- 插入训练记录
INSERT INTO trainrecord (configid, traindata, validdata, epochs, batchsize, trainloss, validloss, trainstatus, modelpath, trainuserid, starttime, endtime) VALUES
(1, '2016-07-01 to 2017-10-31', '2017-11-01 to 2018-06-26', 100, 32, 0.0245, 0.0312, 'completed', '/models/lstm_v1_20180701.pth', 4, '2018-07-01 10:00:00', '2018-07-01 12:35:00'),
(2, '2016-07-01 to 2017-10-31', '2017-11-01 to 2018-06-26', 80, 16, 0.0198, 0.0276, 'completed', '/models/transformer_v1_20180702.pth', 4, '2018-07-02 09:00:00', '2018-07-02 14:20:00'),
(3, '2016-07-01 to 2017-10-31', '2017-11-01 to 2018-06-26', 120, 24, 0.0156, 0.0223, 'completed', '/models/hybrid_v1_20180703.pth', 5, '2018-07-03 08:00:00', '2018-07-03 15:45:00');

-- 插入模型版本
INSERT INTO modelversion (trainid, versionnumber, versiondesc, performance, isactive) VALUES
(1, 'v1.0.0', 'LSTM基础版本', '{"mae":125.8,"rmse":168.3,"mape":2.15,"r2":0.92}', 0),
(2, 'v1.0.0', 'Transformer基础版本', '{"mae":98.5,"rmse":142.6,"mape":1.68,"r2":0.95}', 0),
(3, 'v1.0.0', 'LSTM-Transformer混合模型', '{"mae":76.2,"rmse":115.4,"mape":1.32,"r2":0.97}', 1);

-- 插入预测任务
INSERT INTO predicttask (taskname, versionid, predictstart, predictend, taskstatus, createuserid, executetime) VALUES
('2016年9月第二周预测', 3, '2016-09-05 00:00:00', '2016-09-11 23:59:59', 'completed', 6, '2016-09-04 22:00:00'),
('2016年9月第三周预测', 3, '2016-09-12 00:00:00', '2016-09-18 23:59:59', 'pending', 6, NULL),
('2016年8月负荷预测', 2, '2016-08-01 00:00:00', '2016-08-07 23:59:59', 'completed', 7, '2016-07-31 20:00:00');

-- 插入预测结果（部分示例数据）
INSERT INTO predictresult (taskid, predicttime, predictvalue, actualvalue) VALUES
(1, '2016-09-10 00:00:00', 6320.30, 6350.20),
(1, '2016-09-10 06:00:00', 5695.60, 5720.80),
(1, '2016-09-10 12:00:00', 7885.20, 7920.40),
(1, '2016-09-10 18:00:00', 7220.50, 7250.90),
(1, '2016-09-14 00:00:00', 6455.80, 6480.50),
(1, '2016-09-14 06:00:00', 5825.70, 5850.30),
(1, '2016-09-14 12:00:00', 8018.40, 8050.80),
(1, '2016-09-14 18:00:00', 7352.30, 7380.60);

-- 插入误差指标
INSERT INTO errormetric (taskid, mae, rmse, mape, r2score) VALUES
(1, 32.45, 45.82, 0.58, 0.9845),
(3, 68.32, 92.15, 1.12, 0.9623);

-- 插入操作日志
INSERT INTO operationlog (userid, operation, module, detail, ipaddress) VALUES
(1, 'login', 'auth', '管理员登录系统', '192.168.1.100'),
(2, 'upload_data', 'data', '上传ETT-small(ETTh1)负荷数据', '192.168.1.101'),
(4, 'train_model', 'model', '训练LSTM模型v1', '192.168.1.102'),
(5, 'train_model', 'model', '训练混合模型v1', '192.168.1.103'),
(6, 'create_predict_task', 'predict', '创建9月第二周预测任务', '192.168.1.104'),
(1, 'view_log', 'system', '查看系统日志', '192.168.1.100');

-- 插入系统配置
INSERT INTO systemconfig (configkey, configvalue, configdesc) VALUES
('system_name', '电力负荷预测系统', '系统名称'),
('max_upload_size', '100', '最大上传文件大小(MB)'),
('model_save_path', '/models/', '模型保存路径'),
('data_retention_days', '365', '数据保留天数'),
('predict_interval', '6', '预测时间间隔(小时)'),
('alert_email', 'admin@electric.com', '告警邮箱地址');

-- 插入告警记录
INSERT INTO alertrecord (alerttype, alertlevel, alertmessage, handlestatus, handleuserid, handletime) VALUES
('dataanomaly', 'warning', '检测到2016-07-15数据存在3个异常值', 1, 2, '2016-07-16 09:30:00'),
('modelfailure', 'error', 'LSTM模型训练失败:内存不足', 1, 4, '2016-08-20 14:20:00'),
('systemerror', 'critical', '数据库连接超时', 1, 1, '2016-09-01 08:15:00'),
('dataanomaly', 'info', '9月10日数据质量检查通过', 1, 2, '2016-09-11 10:00:00');
