# 模型管理服务
from models.database import db, ModelConfig, TrainRecord, ModelVersion, OperationLog
from algorithms.preprocessing import DataPreprocessor
from utils.response import success, error
# algorithms/visualization 懒加载，避免模块级 matplotlib/PyTorch 初始化冲突
from datetime import datetime
import json
import os

class ModelService:
    """模型管理服务类"""
    
    def __init__(self):
        import torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.preprocessor = DataPreprocessor()
        self.model_save_dir = './models'
        os.makedirs(self.model_save_dir, exist_ok=True)
    
    def create_model_config(self, modelname, modeltype, hyperparams, architecture, userid):
        """创建模型配置"""
        try:
            config = ModelConfig(
                modelname=modelname,
                modeltype=modeltype,
                hyperparams=json.dumps(hyperparams, ensure_ascii=False),
                architecture=architecture,
                createuserid=userid
            )
            db.session.add(config)
            db.session.commit()
            
            # 记录操作日志
            log = OperationLog(
                userid=userid,
                operation='create_model_config',
                module='model',
                detail=f'创建模型配置: {modelname}'
            )
            db.session.add(log)
            db.session.commit()
            
            return success({'configid': config.configid}, '模型配置创建成功')
        
        except Exception as e:
            db.session.rollback()
            return error(f'模型配置创建失败: {str(e)}')
    
    def get_model_configs(self, modeltype=None):
        """获取模型配置列表"""
        try:
            query = ModelConfig.query
            if modeltype:
                query = query.filter_by(modeltype=modeltype)
            
            configs = query.order_by(ModelConfig.createtime.desc()).all()
            
            result = []
            for config in configs:
                result.append({
                    'configid': config.configid,
                    'modelname': config.modelname,
                    'modeltype': config.modeltype,
                    'hyperparams': json.loads(config.hyperparams) if config.hyperparams else {},
                    'architecture': config.architecture,
                    'createtime': config.createtime.strftime('%Y-%m-%d %H:%M:%S') if config.createtime else None
                })
            
            return success({'configs': result, 'count': len(result)})
        
        except Exception as e:
            return error(f'获取模型配置失败: {str(e)}')
    
    def train_model(self, configid, train_data, valid_data, epochs, batchsize, userid):
        """训练模型"""
        try:
            # 获取模型配置
            config = ModelConfig.query.get(configid)
            if not config:
                return error('模型配置不存在')
            
            hyperparams = json.loads(config.hyperparams)

            # 导入 create_dataloader（各模型文件中均有定义，从 lstm_model 导入即可）
            from algorithms.lstm_model import create_dataloader

            # 准备训练数据
            X_train, y_train = train_data
            X_valid, y_valid = valid_data

            train_loader = create_dataloader(X_train, y_train, batchsize, shuffle=True)
            valid_loader = create_dataloader(X_valid, y_valid, batchsize, shuffle=False)
            
            # 创建训练记录
            train_record = TrainRecord(
                configid=configid,
                traindata=f'{len(X_train)} samples',
                validdata=f'{len(X_valid)} samples',
                epochs=epochs,
                batchsize=batchsize,
                trainstatus='running',
                trainuserid=userid,
                starttime=datetime.now()
            )
            db.session.add(train_record)
            db.session.commit()
            
            # 根据模型类型创建模型
            if config.modeltype == 'lstm':
                from algorithms.lstm_model import LSTMModel, LSTMTrainer
                model = LSTMModel(
                    input_size=1,
                    hidden_size=hyperparams.get('hidden_size', 128),
                    num_layers=hyperparams.get('num_layers', 2),
                    dropout=hyperparams.get('dropout', 0.2)
                )
                trainer = LSTMTrainer(model, hyperparams.get('learning_rate', 0.001), self.device)

            elif config.modeltype == 'transformer':
                from algorithms.transformer_model import TransformerModel, TransformerTrainer
                model = TransformerModel(
                    input_size=1,
                    d_model=hyperparams.get('d_model', 256),
                    nhead=hyperparams.get('nhead', 8),
                    num_layers=hyperparams.get('num_layers', 4),
                    dropout=hyperparams.get('dropout', 0.1)
                )
                trainer = TransformerTrainer(model, hyperparams.get('learning_rate', 0.0001), self.device)

            elif config.modeltype == 'hybrid':
                from algorithms.hybrid_model import HybridModel, HybridTrainer
                model = HybridModel(
                    input_size=1,
                    lstm_hidden=hyperparams.get('lstm_hidden', 64),
                    transformer_layers=hyperparams.get('transformer_layers', 2),
                    nhead=hyperparams.get('nhead', 4),
                    dropout=hyperparams.get('dropout', 0.15)
                )
                trainer = HybridTrainer(model, hyperparams.get('learning_rate', 0.0005), self.device)

            else:
                return error('不支持的模型类型')
            
            # 训练模型
            train_losses, valid_losses = trainer.train(train_loader, valid_loader, epochs)
            
            # 保存模型
            model_filename = f'{config.modeltype}_{configid}_{datetime.now().strftime("%Y%m%d%H%M%S")}.pth'
            model_path = os.path.join(self.model_save_dir, model_filename)
            trainer.save_model(model_path)
            
            # 更新训练记录
            train_record.trainloss = train_losses[-1]
            train_record.validloss = valid_losses[-1]
            train_record.trainstatus = 'completed'
            train_record.modelpath = model_path
            train_record.endtime = datetime.now()
            db.session.commit()
            
            # 生成训练历史图表
            epochs_list = list(range(1, epochs + 1))
            from utils.visualization import plot_training_history
            chart = plot_training_history(epochs_list, train_losses, valid_losses, '训练历史')
            
            # 记录操作日志
            log = OperationLog(
                userid=userid,
                operation='train_model',
                module='model',
                detail=f'训练模型: {config.modelname}'
            )
            db.session.add(log)
            db.session.commit()
            
            return success({
                'trainid': train_record.trainid,
                'train_loss': float(train_losses[-1]),
                'valid_loss': float(valid_losses[-1]),
                'model_path': model_path,
                'chart': chart
            }, '模型训练完成')
        
        except Exception as e:
            db.session.rollback()
            if 'train_record' in locals():
                train_record.trainstatus = 'failed'
                train_record.endtime = datetime.now()
                db.session.commit()
            return error(f'模型训练失败: {str(e)}')

    def get_train_records(self, configid=None):
        """获取训练记录"""
        try:
            query = TrainRecord.query
            if configid:
                query = query.filter_by(configid=configid)

            records = query.order_by(TrainRecord.starttime.desc()).all()

            result = []
            for record in records:
                result.append({
                    'trainid': record.trainid,
                    'configid': record.configid,
                    'traindata': record.traindata,
                    'validdata': record.validdata,
                    'epochs': record.epochs,
                    'batchsize': record.batchsize,
                    'trainloss': float(record.trainloss) if record.trainloss else None,
                    'validloss': float(record.validloss) if record.validloss else None,
                    'trainstatus': record.trainstatus,
                    'modelpath': record.modelpath,
                    'starttime': record.starttime.strftime('%Y-%m-%d %H:%M:%S') if record.starttime else None,
                    'endtime': record.endtime.strftime('%Y-%m-%d %H:%M:%S') if record.endtime else None
                })

            return success({'records': result, 'count': len(result)})

        except Exception as e:
            return error(f'获取训练记录失败: {str(e)}')

    def create_model_version(self, trainid, versionnumber, versiondesc, performance, isactive=0):
        """创建模型版本"""
        try:
            # 如果设置为激活版本，先将其他版本设为非激活
            if isactive == 1:
                ModelVersion.query.update({'isactive': 0})

            version = ModelVersion(
                trainid=trainid,
                versionnumber=versionnumber,
                versiondesc=versiondesc,
                performance=json.dumps(performance, ensure_ascii=False),
                isactive=isactive
            )
            db.session.add(version)
            db.session.commit()

            return success({'versionid': version.versionid}, '模型版本创建成功')

        except Exception as e:
            db.session.rollback()
            return error(f'模型版本创建失败: {str(e)}')

    def get_model_versions(self, trainid=None):
        """获取模型版本列表"""
        try:
            query = ModelVersion.query
            if trainid:
                query = query.filter_by(trainid=trainid)

            versions = query.order_by(ModelVersion.createtime.desc()).all()

            result = []
            for version in versions:
                result.append({
                    'versionid': version.versionid,
                    'trainid': version.trainid,
                    'versionnumber': version.versionnumber,
                    'versiondesc': version.versiondesc,
                    'performance': json.loads(version.performance) if version.performance else {},
                    'isactive': version.isactive,
                    'createtime': version.createtime.strftime('%Y-%m-%d %H:%M:%S') if version.createtime else None
                })

            return success({'versions': result, 'count': len(result)})

        except Exception as e:
            return error(f'获取模型版本失败: {str(e)}')

    def activate_model_version(self, versionid):
        """激活模型版本"""
        try:
            # 将所有版本设为非激活
            ModelVersion.query.update({'isactive': 0})

            # 激活指定版本
            version = ModelVersion.query.get(versionid)
            if not version:
                return error('模型版本不存在')

            version.isactive = 1
            db.session.commit()

            return success(None, '模型版本激活成功')

        except Exception as e:
            db.session.rollback()
            return error(f'模型版本激活失败: {str(e)}')

    def load_model_for_prediction(self, versionid):
        """加载模型用于预测"""
        try:
            version = ModelVersion.query.get(versionid)
            if not version:
                return None, '模型版本不存在'

            train_record = TrainRecord.query.get(version.trainid)
            if not train_record or not train_record.modelpath:
                return None, '模型文件不存在'

            config = ModelConfig.query.get(train_record.configid)
            if not config:
                return None, '模型配置不存在'

            hyperparams = json.loads(config.hyperparams)

            # 根据模型类型创建模型
            if config.modeltype == 'lstm':
                from algorithms.lstm_model import LSTMModel, LSTMTrainer
                model = LSTMModel(
                    input_size=1,
                    hidden_size=hyperparams.get('hidden_size', 128),
                    num_layers=hyperparams.get('num_layers', 2),
                    dropout=hyperparams.get('dropout', 0.2)
                )
                trainer = LSTMTrainer(model, hyperparams.get('learning_rate', 0.001), self.device)

            elif config.modeltype == 'transformer':
                from algorithms.transformer_model import TransformerModel, TransformerTrainer
                model = TransformerModel(
                    input_size=1,
                    d_model=hyperparams.get('d_model', 256),
                    nhead=hyperparams.get('nhead', 8),
                    num_layers=hyperparams.get('num_layers', 4),
                    dropout=hyperparams.get('dropout', 0.1)
                )
                trainer = TransformerTrainer(model, hyperparams.get('learning_rate', 0.0001), self.device)

            elif config.modeltype == 'hybrid':
                from algorithms.hybrid_model import HybridModel, HybridTrainer
                model = HybridModel(
                    input_size=1,
                    lstm_hidden=hyperparams.get('lstm_hidden', 64),
                    transformer_layers=hyperparams.get('transformer_layers', 2),
                    nhead=hyperparams.get('nhead', 4),
                    dropout=hyperparams.get('dropout', 0.15)
                )
                trainer = HybridTrainer(model, hyperparams.get('learning_rate', 0.0005), self.device)

            else:
                return None, '不支持的模型类型'

            # 加载模型权重
            trainer.load_model(train_record.modelpath)

            return trainer, None

        except Exception as e:
            return None, f'模型加载失败: {str(e)}'

