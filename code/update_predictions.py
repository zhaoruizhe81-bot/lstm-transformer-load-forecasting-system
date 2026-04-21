# 更新预测数据库：清理旧数据 -> 创建预测任务 -> 执行预测 -> 填充实际值 -> 计算误差
import sys, os
import numpy as np
import torch
import json
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from models.database import db, LoadData, ModelVersion, TrainRecord, ModelConfig
from models.database import PredictTask, PredictResult, ErrorMetric, OperationLog
from algorithms.preprocessing import DataPreprocessor, calculate_metrics
from algorithms.lstm_model import LSTMModel, LSTMTrainer, create_dataloader
from algorithms.transformer_model import TransformerModel, TransformerTrainer
from algorithms.hybrid_model import HybridModel, HybridTrainer

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:123456@localhost:3306/electric_load_forecasting_system_with_deep_learning_2026'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


DEVICE = get_device()
SEQ_LENGTH = 24

# 预测范围：选验证集内一段，确保有实际值可匹配
# 数据范围 2016-07-01 ~ 2018-06-26 19:00:00
# 我们选 2018-06-01 ~ 2018-06-07（一周）作为预测范围，逐小时预测
PREDICT_START = '2018-06-01 00:00:00'
PREDICT_END   = '2018-06-07 23:00:00'

# 用于准备输入的历史窗口（预测起点前7天）
HISTORY_START = '2018-05-24 00:00:00'


def load_model_by_version(versionid):
    """根据versionid加载模型"""
    version = ModelVersion.query.get(versionid)
    train_record = TrainRecord.query.get(version.trainid)
    config = ModelConfig.query.get(train_record.configid)
    hyperparams = json.loads(config.hyperparams)

    if config.modeltype == 'lstm':
        model = LSTMModel(input_size=1, hidden_size=hyperparams.get('hidden_size', 128),
                          num_layers=hyperparams.get('num_layers', 2), dropout=hyperparams.get('dropout', 0.2))
        trainer = LSTMTrainer(model, hyperparams.get('learning_rate', 0.001), DEVICE)
    elif config.modeltype == 'transformer':
        model = TransformerModel(input_size=1, d_model=hyperparams.get('d_model', 64),
                                 nhead=hyperparams.get('nhead', 4), num_layers=hyperparams.get('num_layers', 2),
                                 dropout=hyperparams.get('dropout', 0.1))
        trainer = TransformerTrainer(model, hyperparams.get('learning_rate', 0.0001), DEVICE)
    elif config.modeltype == 'hybrid':
        model = HybridModel(input_size=1, lstm_hidden=hyperparams.get('lstm_hidden', 64),
                            transformer_layers=hyperparams.get('transformer_layers', 2),
                            nhead=hyperparams.get('nhead', 4), dropout=hyperparams.get('dropout', 0.15))
        trainer = HybridTrainer(model, hyperparams.get('learning_rate', 0.0005), DEVICE)
    else:
        raise ValueError(f'不支持的模型类型: {config.modeltype}')

    trainer.load_model(train_record.modelpath)
    return trainer, config.modeltype, config.modelname


def main():
    print("🔄 更新预测数据库...\n")

    with app.app_context():
        # ===== 1. 清理旧的预测数据 =====
        print("🗑️  清理旧的预测数据...")
        ErrorMetric.query.delete()
        PredictResult.query.delete()
        PredictTask.query.delete()
        db.session.commit()
        print("   ✅ 清理完成\n")

        # ===== 2. 获取所有模型版本 =====
        versions = (
            ModelVersion.query
            .join(TrainRecord, ModelVersion.trainid == TrainRecord.trainid)
            .join(ModelConfig, TrainRecord.configid == ModelConfig.configid)
            .order_by(ModelConfig.configid)
            .all()
        )
        print(f"📋 找到 {len(versions)} 个模型版本")
        for v in versions:
            tr = TrainRecord.query.get(v.trainid)
            cfg = ModelConfig.query.get(tr.configid)
            print(f"   versionid={v.versionid}, model={cfg.modelname}, type={cfg.modeltype}, active={v.isactive}")

        # ===== 3. 准备归一化器（用训练集数据拟合） =====
        print(f"\n📊 加载历史数据用于归一化...")
        train_records = LoadData.query.filter(
            LoadData.recordtime >= '2016-07-01',
            LoadData.recordtime <= '2017-10-31'
        ).order_by(LoadData.recordtime).all()
        train_values = [float(r.loadvalue) for r in train_records]

        preprocessor = DataPreprocessor()
        preprocessor.normalize_minmax(train_values)
        print(f"   训练集 {len(train_values)} 条数据已拟合scaler")

        # ===== 4. 获取预测所需历史数据 =====
        print(f"\n📊 加载预测输入历史: {HISTORY_START} ~ {PREDICT_START}")
        history_records = LoadData.query.filter(
            LoadData.recordtime >= HISTORY_START,
            LoadData.recordtime < PREDICT_START
        ).order_by(LoadData.recordtime).all()
        history_values = [float(r.loadvalue) for r in history_records]
        print(f"   历史数据: {len(history_values)} 条")

        # 归一化历史数据
        history_array = np.array(history_values).reshape(-1, 1)
        history_norm = preprocessor.scaler.transform(history_array).flatten().tolist()

        # ===== 5. 生成预测时间点（逐小时） =====
        predict_start = datetime.strptime(PREDICT_START, '%Y-%m-%d %H:%M:%S')
        predict_end = datetime.strptime(PREDICT_END, '%Y-%m-%d %H:%M:%S')
        predict_times = []
        t = predict_start
        while t <= predict_end:
            predict_times.append(t)
            t += timedelta(hours=1)
        print(f"   预测时间点: {len(predict_times)} 个 ({predict_start} ~ {predict_end})")

        # ===== 6. 对每个模型版本执行预测 =====
        for v in versions:
            trainer, model_type, model_name = load_model_by_version(v.versionid)
            print(f"\n{'='*60}")
            print(f"🚀 模型: {model_name} (versionid={v.versionid})")
            print(f"{'='*60}")

            # 创建预测任务
            task = PredictTask(
                taskname=f'{model_name} 验证集预测 (2018-06-01~06-07)',
                versionid=v.versionid,
                predictstart=predict_start,
                predictend=predict_end,
                taskstatus='running',
                createuserid=4,
                executetime=datetime.now()
            )
            db.session.add(task)
            db.session.flush()
            print(f"   创建任务 taskid={task.taskid}")

            # 逐小时滚动预测
            input_sequence = history_norm[-SEQ_LENGTH:]
            predictions = []
            for pt in predict_times:
                X_input = np.array([input_sequence])
                pred_norm = trainer.predict(X_input)[0]
                pred_value = preprocessor.denormalize_minmax([pred_norm])[0]
                predictions.append(pred_value)

                # 保存预测结果
                result = PredictResult(
                    taskid=task.taskid,
                    predicttime=pt,
                    predictvalue=round(pred_value, 2)
                )
                db.session.add(result)

                # 滚动更新输入序列
                input_sequence = input_sequence[1:] + [pred_norm]

            task.taskstatus = 'completed'
            db.session.commit()
            print(f"   ✅ 预测完成，共 {len(predictions)} 条")

            # ===== 7. 填充实际值 =====
            print(f"   📝 填充实际值...")
            results = PredictResult.query.filter_by(taskid=task.taskid).all()
            updated = 0
            for r in results:
                actual = LoadData.query.filter_by(recordtime=r.predicttime).first()
                if actual:
                    r.actualvalue = actual.loadvalue
                    updated += 1
            db.session.commit()
            print(f"   ✅ 填充了 {updated}/{len(results)} 条实际值")

            # ===== 8. 计算误差指标 =====
            valid_results = [r for r in results if r.actualvalue is not None]
            if valid_results:
                y_true = [float(r.actualvalue) for r in valid_results]
                y_pred = [float(r.predictvalue) for r in valid_results]
                metrics = calculate_metrics(y_true, y_pred)

                em = ErrorMetric(
                    taskid=task.taskid,
                    mae=round(metrics['mae'], 4),
                    rmse=round(metrics['rmse'], 4),
                    mape=round(metrics['mape'], 4),
                    r2score=round(metrics['r2score'], 6)
                )
                db.session.add(em)
                db.session.commit()
                print(f"   📊 MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  MAPE={metrics['mape']:.2f}%  R2={metrics['r2score']:.4f}")
            else:
                print(f"   ⚠️ 无实际值，跳过误差计算")

        # ===== 9. 汇总 =====
        print(f"\n{'='*60}")
        print(f"🎉 全部完成！")
        print(f"{'='*60}")
        tasks = PredictTask.query.all()
        for t in tasks:
            total = PredictResult.query.filter_by(taskid=t.taskid).count()
            with_actual = PredictResult.query.filter(
                PredictResult.taskid == t.taskid,
                PredictResult.actualvalue.isnot(None)
            ).count()
            em = ErrorMetric.query.filter_by(taskid=t.taskid).first()
            mae_str = f"MAE={float(em.mae):.4f}" if em else "无指标"
            print(f"  taskid={t.taskid} [{t.taskname}] 预测{total}条, 实际值{with_actual}条, {mae_str}")


if __name__ == '__main__':
    main()
