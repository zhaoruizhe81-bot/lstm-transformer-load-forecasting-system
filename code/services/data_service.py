# 数据管理服务
from models.database import db, LoadData, DataQuality, DataProcess, User, OperationLog
from algorithms.preprocessing import DataPreprocessor, calculate_metrics
from utils.response import success, error
# utils.visualization 懒加载，避免模块级 matplotlib 初始化冲突
from datetime import datetime
import pandas as pd
import numpy as np
import json

class DataService:
    """数据管理服务类"""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
    
    def upload_data(self, data_list, userid):
        """批量上传负荷数据"""
        try:
            for item in data_list:
                load_data = LoadData(
                    recordtime=datetime.strptime(item['recordtime'], '%Y-%m-%d %H:%M:%S'),
                    loadvalue=item['loadvalue'],
                    temperature=item.get('temperature'),
                    humidity=item.get('humidity'),
                    holiday=item.get('holiday', 0),
                    weekday=item.get('weekday'),
                    datasource=item.get('datasource', 'manual_upload'),
                    uploaduserid=userid
                )
                db.session.add(load_data)
            
            db.session.commit()
            
            # 记录操作日志
            log = OperationLog(
                userid=userid,
                operation='upload_data',
                module='data',
                detail=f'上传{len(data_list)}条负荷数据'
            )
            db.session.add(log)
            db.session.commit()
            
            return success({'count': len(data_list)}, '数据上传成功')
        
        except Exception as e:
            db.session.rollback()
            return error(f'数据上传失败: {str(e)}')
    
    def query_data(self, start_time=None, end_time=None, page=1, page_size=100):
        """查询负荷数据"""
        try:
            query = LoadData.query
            
            if start_time:
                query = query.filter(LoadData.recordtime >= start_time)
            if end_time:
                query = query.filter(LoadData.recordtime <= end_time)
            
            total = query.count()
            data_list = query.order_by(LoadData.recordtime.desc()).paginate(
                page=page, per_page=page_size, error_out=False
            )
            
            result = []
            for item in data_list.items:
                result.append({
                    'dataid': item.dataid,
                    'recordtime': item.recordtime.strftime('%Y-%m-%d %H:%M:%S'),
                    'loadvalue': float(item.loadvalue),
                    'temperature': float(item.temperature) if item.temperature else None,
                    'humidity': float(item.humidity) if item.humidity else None,
                    'holiday': item.holiday,
                    'weekday': item.weekday,
                    'datasource': item.datasource
                })
            
            return success({
                'total': total,
                'page': page,
                'page_size': page_size,
                'data': result
            })
        
        except Exception as e:
            return error(f'数据查询失败: {str(e)}')
    
    def detect_outliers(self, start_time, end_time, method='zscore', threshold=3, userid=None):
        """异常检测"""
        try:
            data_list = LoadData.query.filter(
                LoadData.recordtime >= start_time,
                LoadData.recordtime <= end_time
            ).order_by(LoadData.recordtime).all()
            
            if not data_list:
                return error('指定时间范围内无数据')
            
            load_values = [float(item.loadvalue) for item in data_list]
            
            if method == 'zscore':
                outliers, z_scores = self.preprocessor.detect_outliers_zscore(load_values, threshold)
                detail = {'method': 'zscore', 'threshold': threshold, 'outliers': outliers}
            elif method == 'iqr':
                outliers, iqr_info = self.preprocessor.detect_outliers_iqr(load_values)
                detail = {'method': 'iqr', 'outliers': outliers, 'iqr_info': iqr_info}
            elif method == 'boxplot':
                outliers, iqr_info = self.preprocessor.detect_outliers_boxplot(load_values)
                detail = {'method': 'boxplot', 'outliers': outliers, 'iqr_info': iqr_info}
            else:
                return error('不支持的异常检测方法')
            
            # 生成箱型图
            from utils.visualization import plot_boxplot
            chart = plot_boxplot(load_values, f'{method}异常检测')
            
            # 记录质量检查
            if userid:
                quality = DataQuality(
                    checktype='outlier',
                    issuecount=len(outliers),
                    issuedetail=json.dumps(detail, ensure_ascii=False),
                    checkuserid=userid
                )
                db.session.add(quality)
                db.session.commit()
            
            return success({
                'outliers': outliers,
                'outlier_count': len(outliers),
                'detail': detail,
                'chart': chart
            }, '异常检测完成')
        
        except Exception as e:
            return error(f'异常检测失败: {str(e)}')
    
    def fill_missing(self, start_time, end_time, method='linear', userid=None):
        """填充缺失值"""
        try:
            data_list = LoadData.query.filter(
                LoadData.recordtime >= start_time,
                LoadData.recordtime <= end_time
            ).order_by(LoadData.recordtime).all()
            
            if not data_list:
                return error('指定时间范围内无数据')
            
            load_values = [float(item.loadvalue) if item.loadvalue else None for item in data_list]
            missing_indices = [i for i, v in enumerate(load_values) if v is None]
            
            if not missing_indices:
                return success({'filled_count': 0}, '无缺失值需要填充')
            
            # 填充缺失值
            if method == 'linear':
                filled_values = self.preprocessor.fill_missing_linear(load_values, missing_indices)
            elif method == 'spline':
                filled_values = self.preprocessor.fill_missing_spline(load_values, missing_indices)
            else:
                return error('不支持的填充方法')
            
            # 更新数据库
            for i in missing_indices:
                data_list[i].loadvalue = filled_values[i]
            db.session.commit()
            
            # 记录处理历史
            if userid:
                process = DataProcess(
                    processtype='fillmissing',
                    processmethod=method,
                    inputdatarange=f'{start_time} to {end_time}',
                    outputdatarange=f'{start_time} to {end_time}',
                    processparams=json.dumps({'method': method}),
                    processresult=json.dumps({'filled_count': len(missing_indices), 'success': True}),
                    processuserid=userid
                )
                db.session.add(process)
                db.session.commit()
            
            return success({'filled_count': len(missing_indices)}, '缺失值填充完成')

        except Exception as e:
            db.session.rollback()
            return error(f'缺失值填充失败: {str(e)}')

    def normalize_data(self, start_time, end_time, method='minmax', userid=None):
        """数据归一化"""
        try:
            data_list = LoadData.query.filter(
                LoadData.recordtime >= start_time,
                LoadData.recordtime <= end_time
            ).order_by(LoadData.recordtime).all()

            if not data_list:
                return error('指定时间范围内无数据')

            load_values = [float(item.loadvalue) for item in data_list]

            if method == 'minmax':
                normalized = self.preprocessor.normalize_minmax(load_values)
            else:
                return error('不支持的归一化方法')

            # 记录处理历史
            if userid:
                process = DataProcess(
                    processtype='normalize',
                    processmethod=method,
                    inputdatarange=f'{start_time} to {end_time}',
                    outputdatarange=f'{start_time} to {end_time}',
                    processparams=json.dumps({'method': method}),
                    processresult=json.dumps({'normalized_count': len(normalized), 'success': True}),
                    processuserid=userid
                )
                db.session.add(process)
                db.session.commit()

            return success({
                'normalized_data': normalized,
                'count': len(normalized)
            }, '数据归一化完成')

        except Exception as e:
            return error(f'数据归一化失败: {str(e)}')

    def correlation_analysis(self, start_time, end_time, userid=None):
        """相关性分析"""
        try:
            data_list = LoadData.query.filter(
                LoadData.recordtime >= start_time,
                LoadData.recordtime <= end_time
            ).order_by(LoadData.recordtime).all()

            if not data_list:
                return error('指定时间范围内无数据')

            # 构建DataFrame
            df_data = []
            for item in data_list:
                df_data.append({
                    'loadvalue': float(item.loadvalue),
                    'temperature': float(item.temperature) if item.temperature else 0,
                    'humidity': float(item.humidity) if item.humidity else 0,
                    'weekday': item.weekday if item.weekday else 0,
                    'holiday': item.holiday
                })

            df = pd.DataFrame(df_data)
            result = self.preprocessor.correlation_analysis(df, 'loadvalue')

            # 生成相关性热力图
            from utils.visualization import plot_correlation_matrix
            corr_matrix = df.corr()
            chart = plot_correlation_matrix(
                corr_matrix.values,
                corr_matrix.columns.tolist(),
                '特征相关性分析'
            )

            # 记录处理历史
            if userid:
                process = DataProcess(
                    processtype='feature',
                    processmethod='correlation_analysis',
                    inputdatarange=f'{start_time} to {end_time}',
                    outputdatarange='N/A',
                    processparams=json.dumps({'features': list(df.columns)}),
                    processresult=json.dumps({'correlations': result['target_correlation']}),
                    processuserid=userid
                )
                db.session.add(process)
                db.session.commit()

            return success({
                'correlation': result['target_correlation'],
                'chart': chart
            }, '相关性分析完成')

        except Exception as e:
            return error(f'相关性分析失败: {str(e)}')

    def visualize_load_curve(self, start_time, end_time):
        """可视化负荷曲线"""
        try:
            data_list = LoadData.query.filter(
                LoadData.recordtime >= start_time,
                LoadData.recordtime <= end_time
            ).order_by(LoadData.recordtime).all()

            if not data_list:
                return error('指定时间范围内无数据')

            times = [item.recordtime.strftime('%Y-%m-%d %H:%M') for item in data_list]
            values = [float(item.loadvalue) for item in data_list]

            from utils.visualization import plot_load_curve
            chart = plot_load_curve(times, values, '负荷曲线')

            return success({'chart': chart}, '负荷曲线生成成功')

        except Exception as e:
            return error(f'负荷曲线生成失败: {str(e)}')

    def get_data_statistics(self, start_time, end_time):
        """获取数据统计信息"""
        try:
            data_list = LoadData.query.filter(
                LoadData.recordtime >= start_time,
                LoadData.recordtime <= end_time
            ).all()

            if not data_list:
                return error('指定时间范围内无数据')

            load_values = [float(item.loadvalue) for item in data_list]

            stats = {
                'count': len(load_values),
                'mean': float(np.mean(load_values)),
                'std': float(np.std(load_values)),
                'min': float(np.min(load_values)),
                'max': float(np.max(load_values)),
                'median': float(np.median(load_values))
            }

            return success(stats, '统计信息获取成功')

        except Exception as e:
            return error(f'统计信息获取失败: {str(e)}')

