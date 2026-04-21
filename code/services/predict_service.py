# 预测服务
from models.database import db, PredictTask, PredictResult, ErrorMetric, LoadData, OperationLog
from services.model_service import ModelService
from algorithms.preprocessing import DataPreprocessor, calculate_metrics
from utils.response import success, error
from utils.visualization import plot_prediction_comparison, plot_scatter
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import json

class PredictService:
    """预测服务类"""
    
    def __init__(self):
        self.model_service = ModelService()
        self.preprocessor = DataPreprocessor()
    
    def create_predict_task(self, taskname, versionid, predictstart, predictend, userid):
        """创建预测任务"""
        try:
            task = PredictTask(
                taskname=taskname,
                versionid=versionid,
                predictstart=datetime.strptime(predictstart, '%Y-%m-%d %H:%M:%S'),
                predictend=datetime.strptime(predictend, '%Y-%m-%d %H:%M:%S'),
                taskstatus='pending',
                createuserid=userid
            )
            db.session.add(task)
            db.session.commit()
            
            # 记录操作日志
            log = OperationLog(
                userid=userid,
                operation='create_predict_task',
                module='predict',
                detail=f'创建预测任务: {taskname}'
            )
            db.session.add(log)
            db.session.commit()
            
            return success({'taskid': task.taskid}, '预测任务创建成功')
        
        except Exception as e:
            db.session.rollback()
            return error(f'预测任务创建失败: {str(e)}')
    
    def get_predict_tasks(self, userid=None, status=None):
        """获取预测任务列表"""
        try:
            query = PredictTask.query
            if userid:
                query = query.filter_by(createuserid=userid)
            if status:
                query = query.filter_by(taskstatus=status)
            
            tasks = query.order_by(PredictTask.createtime.desc()).all()
            
            result = []
            for task in tasks:
                result.append({
                    'taskid': task.taskid,
                    'taskname': task.taskname,
                    'versionid': task.versionid,
                    'predictstart': task.predictstart.strftime('%Y-%m-%d %H:%M:%S') if task.predictstart else None,
                    'predictend': task.predictend.strftime('%Y-%m-%d %H:%M:%S') if task.predictend else None,
                    'taskstatus': task.taskstatus,
                    'createtime': task.createtime.strftime('%Y-%m-%d %H:%M:%S') if task.createtime else None,
                    'executetime': task.executetime.strftime('%Y-%m-%d %H:%M:%S') if task.executetime else None
                })
            
            return success({'tasks': result, 'count': len(result)})
        
        except Exception as e:
            return error(f'获取预测任务失败: {str(e)}')
    
    def execute_predict_task(self, taskid, userid):
        """执行预测任务"""
        try:
            task = PredictTask.query.get(taskid)
            if not task:
                return error('预测任务不存在')
            
            # 更新任务状态
            task.taskstatus = 'running'
            task.executetime = datetime.now()
            db.session.commit()
            
            # 加载模型
            trainer, err_msg = self.model_service.load_model_for_prediction(task.versionid)
            if err_msg:
                task.taskstatus = 'failed'
                db.session.commit()
                return error(err_msg)
            
            # 获取历史数据用于预测
            history_start = task.predictstart - timedelta(days=7)
            history_data = LoadData.query.filter(
                LoadData.recordtime >= history_start,
                LoadData.recordtime < task.predictstart
            ).order_by(LoadData.recordtime).all()
            
            if len(history_data) < 24:
                task.taskstatus = 'failed'
                db.session.commit()
                return error('历史数据不足，无法进行预测')
            
            # 准备输入数据
            history_values = [float(item.loadvalue) for item in history_data]
            normalized_history = self.preprocessor.normalize_minmax(history_values)
            
            # 生成预测时间点（逐小时）
            predict_times = []
            current_time = task.predictstart
            while current_time <= task.predictend:
                predict_times.append(current_time)
                current_time += timedelta(hours=1)

            # 执行预测
            predictions = []
            input_sequence = normalized_history[-24:]

            for predict_time in predict_times:
                # 使用最近24个时间步作为输入
                X_input = np.array([input_sequence])
                pred_normalized = trainer.predict(X_input)[0]

                # 反归一化
                pred_value = self.preprocessor.denormalize_minmax([pred_normalized])[0]
                predictions.append(pred_value)

                # 更新输入序列
                input_sequence = input_sequence[1:] + [pred_normalized]

                # 保存预测结果
                result = PredictResult(
                    taskid=taskid,
                    predicttime=predict_time,
                    predictvalue=round(pred_value, 2)
                )
                db.session.add(result)

            # 更新任务状态
            task.taskstatus = 'completed'
            db.session.commit()

            # 自动填充实际值
            results = PredictResult.query.filter_by(taskid=taskid).all()
            updated_count = 0
            for r in results:
                actual_data = LoadData.query.filter_by(recordtime=r.predicttime).first()
                if actual_data:
                    r.actualvalue = actual_data.loadvalue
                    updated_count += 1
            db.session.commit()

            # 自动计算误差指标（如果有实际值）
            valid_results = [r for r in results if r.actualvalue is not None]
            metrics_data = None
            if valid_results:
                y_true = [float(r.actualvalue) for r in valid_results]
                y_pred = [float(r.predictvalue) for r in valid_results]
                metrics_data = calculate_metrics(y_true, y_pred)

                error_metric = ErrorMetric(
                    taskid=taskid,
                    mae=round(metrics_data['mae'], 4),
                    rmse=round(metrics_data['rmse'], 4),
                    mape=round(metrics_data['mape'], 4),
                    r2score=round(metrics_data['r2score'], 6)
                )
                db.session.add(error_metric)
                db.session.commit()

            # 记录操作日志
            log = OperationLog(
                userid=userid,
                operation='execute_predict',
                module='predict',
                detail=f'执行预测任务: {task.taskname}, 预测{len(predictions)}条, 填充实际值{updated_count}条'
            )
            db.session.add(log)
            db.session.commit()

            return success({
                'taskid': taskid,
                'predictions': predictions,
                'predict_times': [t.strftime('%Y-%m-%d %H:%M:%S') for t in predict_times],
                'actual_count': updated_count,
                'metrics': metrics_data
            }, '预测任务执行成功')
        
        except Exception as e:
            db.session.rollback()
            if 'task' in locals():
                task.taskstatus = 'failed'
                db.session.commit()
            return error(f'预测任务执行失败: {str(e)}')

    def get_predict_results(self, taskid):
        """获取预测结果"""
        try:
            results = PredictResult.query.filter_by(taskid=taskid).order_by(PredictResult.predicttime).all()

            result_list = []
            for result in results:
                result_list.append({
                    'resultid': result.resultid,
                    'predicttime': result.predicttime.strftime('%Y-%m-%d %H:%M:%S') if result.predicttime else None,
                    'predictvalue': float(result.predictvalue) if result.predictvalue else None,
                    'actualvalue': float(result.actualvalue) if result.actualvalue else None
                })

            return success({'results': result_list, 'count': len(result_list)})

        except Exception as e:
            return error(f'获取预测结果失败: {str(e)}')

    def update_actual_values(self, taskid):
        """更新实际值（从数据库中获取）"""
        try:
            results = PredictResult.query.filter_by(taskid=taskid).all()

            updated_count = 0
            for result in results:
                actual_data = LoadData.query.filter_by(recordtime=result.predicttime).first()
                if actual_data:
                    result.actualvalue = actual_data.loadvalue
                    updated_count += 1

            db.session.commit()

            return success({'updated_count': updated_count}, '实际值更新成功')

        except Exception as e:
            db.session.rollback()
            return error(f'实际值更新失败: {str(e)}')

    def calculate_error_metrics(self, taskid):
        """计算误差指标"""
        try:
            results = PredictResult.query.filter_by(taskid=taskid).all()

            # 筛选有实际值的结果
            valid_results = [r for r in results if r.actualvalue is not None]

            if not valid_results:
                return error('没有可用的实际值进行误差计算')

            y_true = [float(r.actualvalue) for r in valid_results]
            y_pred = [float(r.predictvalue) for r in valid_results]

            metrics = calculate_metrics(y_true, y_pred)

            # 保存误差指标
            error_metric = ErrorMetric(
                taskid=taskid,
                mae=metrics['mae'],
                rmse=metrics['rmse'],
                mape=metrics['mape'],
                r2score=metrics['r2score']
            )
            db.session.add(error_metric)
            db.session.commit()

            return success(metrics, '误差指标计算完成')

        except Exception as e:
            db.session.rollback()
            return error(f'误差指标计算失败: {str(e)}')

    def visualize_prediction(self, taskid):
        """可视化预测结果"""
        try:
            results = PredictResult.query.filter_by(taskid=taskid).order_by(PredictResult.predicttime).all()

            if not results:
                return error('预测结果不存在')

            times = [r.predicttime.strftime('%Y-%m-%d %H:%M') for r in results]
            predicted = [float(r.predictvalue) for r in results]
            actual = [float(r.actualvalue) if r.actualvalue else None for r in results]

            # 如果有实际值，生成对比图
            if any(a is not None for a in actual):
                valid_indices = [i for i, a in enumerate(actual) if a is not None]
                valid_times = [times[i] for i in valid_indices]
                valid_actual = [actual[i] for i in valid_indices]
                valid_predicted = [predicted[i] for i in valid_indices]

                comparison_chart = plot_prediction_comparison(
                    valid_times, valid_actual, valid_predicted, '预测值与实际值对比'
                )

                scatter_chart = plot_scatter(
                    valid_actual, valid_predicted, '实际值 (MW)', '预测值 (MW)', '预测散点图'
                )

                return success({
                    'comparison_chart': comparison_chart,
                    'scatter_chart': scatter_chart
                }, '预测结果可视化完成')
            else:
                # 只有预测值，生成预测曲线
                from utils.visualization import plot_load_curve
                chart = plot_load_curve(times, predicted, '预测负荷曲线')
                return success({'chart': chart}, '预测结果可视化完成')

        except Exception as e:
            return error(f'预测结果可视化失败: {str(e)}')

    def export_predict_results(self, taskid):
        """导出预测结果"""
        try:
            task = PredictTask.query.get(taskid)
            if not task:
                return error('预测任务不存在')

            results = PredictResult.query.filter_by(taskid=taskid).order_by(PredictResult.predicttime).all()

            if not results:
                return error('预测结果不存在')

            # 构建导出数据
            export_data = {
                'task_info': {
                    'taskid': task.taskid,
                    'taskname': task.taskname,
                    'predictstart': task.predictstart.strftime('%Y-%m-%d %H:%M:%S') if task.predictstart else None,
                    'predictend': task.predictend.strftime('%Y-%m-%d %H:%M:%S') if task.predictend else None,
                    'taskstatus': task.taskstatus
                },
                'results': []
            }

            for result in results:
                export_data['results'].append({
                    'predicttime': result.predicttime.strftime('%Y-%m-%d %H:%M:%S') if result.predicttime else None,
                    'predictvalue': float(result.predictvalue) if result.predictvalue else None,
                    'actualvalue': float(result.actualvalue) if result.actualvalue else None
                })

            # 获取误差指标
            metrics = ErrorMetric.query.filter_by(taskid=taskid).first()
            if metrics:
                export_data['metrics'] = {
                    'mae': float(metrics.mae) if metrics.mae else None,
                    'rmse': float(metrics.rmse) if metrics.rmse else None,
                    'mape': float(metrics.mape) if metrics.mape else None,
                    'r2score': float(metrics.r2score) if metrics.r2score else None
                }

            return success(export_data, '预测结果导出成功')

        except Exception as e:
            return error(f'预测结果导出失败: {str(e)}')

