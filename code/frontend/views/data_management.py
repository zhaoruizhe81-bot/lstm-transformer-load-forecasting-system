# 数据管理页面（数据分析师）
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, date
from ui_utils.api_client import APIClient
from ui_utils.helpers import show_success, show_error, show_api_response, display_base64_image, dataframe_to_csv
from config import OUTLIER_METHODS, FILL_METHODS

def _get_data_range():
    """从数据库获取实际数据的时间范围，缓存到session_state"""
    if 'data_range' not in st.session_state:
        try:
            api_client = APIClient()
            resp = api_client.query_data(page=1, page_size=1)
            # 查最早一条
            resp_asc = api_client.get_data_date_range()
            if resp_asc and resp_asc.get('code') == 200 and resp_asc['data']:
                st.session_state.data_range = (
                    datetime.strptime(resp_asc['data']['min_date'], '%Y-%m-%d %H:%M:%S').date(),
                    datetime.strptime(resp_asc['data']['max_date'], '%Y-%m-%d %H:%M:%S').date(),
                )
            else:
                st.session_state.data_range = (date(2016, 7, 1), date(2018, 6, 26))
        except Exception:
            st.session_state.data_range = (date(2016, 7, 1), date(2018, 6, 26))
    return st.session_state.data_range

def _default_start():
    return _get_data_range()[0]

def _default_end():
    mn, mx = _get_data_range()
    # 默认结束 = 开始后7天，但不超过最大值
    end = mn + timedelta(days=7)
    return min(end, mx)

def show_data_management_page():
    """数据管理主页面"""
    st.title('📊 数据管理')
    tab1, tab2, tab3, tab4, tab5 = st.tabs(['数据查询', '数据上传', '异常检测', '数据处理', '数据可视化'])
    api_client = APIClient()
    
    with tab1:
        show_data_query(api_client)
    with tab2:
        show_data_upload(api_client)
    with tab3:
        show_outlier_detection(api_client)
    with tab4:
        show_data_processing(api_client)
    with tab5:
        show_data_visualization(api_client)

def show_data_query(api_client):
    """数据查询"""
    st.subheader('🔍 负荷数据查询')
    mn, mx = _get_data_range()
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input('开始日期', value=mn, min_value=mn, max_value=mx, key='q_sd')
    with col2:
        end_date = st.date_input('结束日期', value=min(mn + timedelta(days=7), mx), min_value=mn, max_value=mx, key='q_ed')
    
    col3, col4 = st.columns(2)
    with col3:
        page = st.number_input('页码', min_value=1, value=1, key='q_page')
    with col4:
        page_size = st.selectbox('每页条数', [20, 50, 100, 200], index=1, key='q_ps')
    
    if st.button('查询数据', type='primary', key='q_btn'):
        start_dt = f"{start_date} 00:00:00"
        end_dt = f"{end_date} 23:59:59"
        with st.spinner('查询中...'):
            response = api_client.query_data(start_dt, end_dt, page, page_size)
        if response['code'] == 200:
            data = response['data']
            st.success(f"共查询到 {data['total']} 条数据")
            if data['data']:
                df = pd.DataFrame(data['data'])
                st.dataframe(df, use_container_width=True)
                csv = dataframe_to_csv(df)
                st.download_button('📥 下载CSV', data=csv, file_name=f'load_data_{start_date}_{end_date}.csv', mime='text/csv')
                with st.expander('📈 数据统计'):
                    stat_resp = api_client.get_data_statistics(start_dt, end_dt)
                    if stat_resp['code'] == 200:
                        s = stat_resp['data']
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric('数据量', s['count']); st.metric('最小值', f"{s['min']:.2f} MW")
                        with c2:
                            st.metric('平均值', f"{s['mean']:.2f} MW"); st.metric('最大值', f"{s['max']:.2f} MW")
                        with c3:
                            st.metric('标准差', f"{s['std']:.2f}"); st.metric('中位数', f"{s['median']:.2f} MW")
            else:
                st.info('未查询到数据')
        else:
            show_error(response['message'])

def show_data_upload(api_client):
    """数据上传"""
    st.subheader('📤 上传负荷数据')
    st.info('支持三种CSV格式：\n- **标准格式**：列名 recordtime, loadvalue（temperature/humidity/holiday/weekday 可选）\n- **COMED格式**：列名 Datetime, COMED_MW（自动转换）\n- **ETT-small格式**：列名 date, OT（自动转换，OT作为负荷值）')
    uploaded_file = st.file_uploader('选择CSV文件', type=['csv'], key='upload_csv')
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write('原始预览：'); st.dataframe(df.head(5), use_container_width=True)

            # 自动检测并转换 ETT-small 格式
            if 'date' in df.columns and 'OT' in df.columns:
                st.info('✅ 检测到 ETT-small 格式，自动转换中...')
                df = df.rename(columns={'date': 'recordtime', 'OT': 'loadvalue'})
                df['recordtime'] = pd.to_datetime(df['recordtime']).dt.strftime('%Y-%m-%d %H:%M:%S')
                df['loadvalue'] = pd.to_numeric(df['loadvalue'], errors='coerce')
                # 缩放到合理电网负荷范围 (3000-10000 MW)
                lmin, lmax = df['loadvalue'].min(), df['loadvalue'].max()
                df['loadvalue'] = (df['loadvalue'] - lmin) / (lmax - lmin) * 7000 + 3000
                df = df.dropna(subset=['recordtime', 'loadvalue'])
                if 'holiday' not in df.columns: df['holiday'] = 0
                if 'weekday' not in df.columns:
                    df['weekday'] = pd.to_datetime(df['recordtime']).dt.dayofweek + 1
                if 'datasource' not in df.columns: df['datasource'] = 'ett_small'
                st.success(f'转换完成，共 {len(df)} 条数据')
                st.dataframe(df.head(5), use_container_width=True)
            # 自动检测并转换 COMED 格式
            elif 'Datetime' in df.columns and 'COMED_MW' in df.columns:
                st.info('✅ 检测到 COMED 格式，自动转换中...')
                df = df.rename(columns={'Datetime': 'recordtime', 'COMED_MW': 'loadvalue'})
                df['recordtime'] = pd.to_datetime(df['recordtime']).dt.strftime('%Y-%m-%d %H:%M:%S')
                df['loadvalue'] = pd.to_numeric(df['loadvalue'], errors='coerce')
                df = df.dropna(subset=['recordtime', 'loadvalue'])
                if 'holiday' not in df.columns: df['holiday'] = 0
                if 'weekday' not in df.columns:
                    df['weekday'] = pd.to_datetime(df['recordtime']).dt.dayofweek + 1
                if 'datasource' not in df.columns: df['datasource'] = 'COMED_hourly'
                st.success(f'转换完成，共 {len(df)} 条数据')
                st.dataframe(df.head(5), use_container_width=True)
            elif 'recordtime' not in df.columns or 'loadvalue' not in df.columns:
                show_error('CSV 缺少必要列。需要 recordtime+loadvalue 或 Datetime+COMED_MW')
                return

            try:
                t_min = pd.to_datetime(df['recordtime']).min()
                t_max = pd.to_datetime(df['recordtime']).max()
                st.info(f'数据时间范围：{t_min} ~ {t_max}，共 {len(df)} 条')
            except Exception:
                pass

            max_upload = 5000
            if len(df) > max_upload:
                st.warning(f'数据量较大（{len(df)} 条），每次最多上传 {max_upload} 条，请分批上传')
                batch_size = st.number_input('每批条数', min_value=100, max_value=max_upload, value=1000, step=100, key='batch_sz')
                batch_idx = st.number_input('批次序号（从0开始）', min_value=0, value=0, key='batch_idx')
                start_i = int(batch_idx) * int(batch_size)
                end_i = min(start_i + int(batch_size), len(df))
                df_batch = df.iloc[start_i:end_i]
                st.info(f'当前批次：第 {start_i+1} ~ {end_i} 条')
                if st.button(f'上传此批次 ({len(df_batch)} 条)', type='primary', key='upload_btn'):
                    _do_upload(api_client, df_batch)
            else:
                if st.button(f'确认上传（{len(df)} 条）', type='primary', key='upload_btn'):
                    _do_upload(api_client, df)
        except Exception as e:
            show_error(f'文件解析失败: {str(e)}')

    with st.expander('✏️ 手动输入单条数据'):
        with st.form('manual_input'):
            c1, c2 = st.columns(2)
            with c1:
                record_date = st.date_input('日期', value=date(2016, 7, 1), min_value=date(2000,1,1), max_value=date(2030,12,31), key='mi_date')
                record_time = st.time_input('时间', key='mi_time')
                loadvalue = st.number_input('负荷值(MW)', min_value=0.0, step=0.1, key='mi_lv')
            with c2:
                temperature = st.number_input('温度(℃)', value=20.0, step=0.1, key='mi_temp')
                humidity = st.number_input('湿度(%)', min_value=0.0, max_value=100.0, value=50.0, key='mi_hum')
                holiday = st.checkbox('节假日', key='mi_hol')
                weekday = st.selectbox('星期几', [1,2,3,4,5,6,7], key='mi_wd')
            if st.form_submit_button('提交'):
                data_list = [{'recordtime': f"{record_date} {record_time}", 'loadvalue': loadvalue,
                              'temperature': temperature, 'humidity': humidity,
                              'holiday': 1 if holiday else 0, 'weekday': weekday}]
                response = api_client.upload_data(data_list, st.session_state.user_info['userid'])
                show_api_response(response)

def _do_upload(api_client, df):
    userid = st.session_state.user_info['userid']
    data_list = df.to_dict('records')
    with st.spinner(f'上传 {len(data_list)} 条数据中...'):
        response = api_client.upload_data(data_list, userid)
    if show_api_response(response):
        st.balloons()
        if 'data_range' in st.session_state:
            del st.session_state['data_range']

def show_outlier_detection(api_client):
    """异常检测"""
    st.subheader('🔎 异常值检测')
    mn, mx = _get_data_range()
    c1, c2 = st.columns(2)
    with c1:
        od_start = st.date_input('开始日期', value=mn, min_value=mn, max_value=mx, key='od_s')
    with c2:
        od_end = st.date_input('结束日期', value=min(mn + timedelta(days=30), mx), min_value=mn, max_value=mx, key='od_e')
    c3, c4 = st.columns(2)
    with c3:
        method = st.selectbox('检测方法', list(OUTLIER_METHODS.keys()), format_func=lambda x: OUTLIER_METHODS[x], key='od_m')
    with c4:
        threshold = st.slider('Z-score阈值', 1.0, 5.0, 3.0, 0.1, key='od_t')

    if st.button('开始检测', type='primary', key='od_btn'):
        userid = st.session_state.user_info['userid']
        with st.spinner('检测中...'):
            response = api_client.detect_outliers(f"{od_start} 00:00:00", f"{od_end} 23:59:59", method, threshold, userid)
        if response['code'] == 200:
            data = response['data']
            show_success(f"检测完成，发现 {data['outlier_count']} 个异常值")
            if data['outlier_count'] > 0:
                st.write('异常值索引：', data['outliers'])
            if 'chart' in data:
                display_base64_image(data['chart'])
        else:
            show_error(response['message'])

def show_data_processing(api_client):
    """数据处理"""
    st.subheader('⚙️ 数据处理')
    mn, mx = _get_data_range()
    st.markdown('#### 缺失值填充')
    c1, c2 = st.columns(2)
    with c1:
        fill_sd = st.date_input('开始日期', value=mn, min_value=mn, max_value=mx, key='fill_s')
    with c2:
        fill_ed = st.date_input('结束日期', value=min(mn + timedelta(days=30), mx), min_value=mn, max_value=mx, key='fill_e')
    fill_method = st.selectbox('填充方法', list(FILL_METHODS.keys()), format_func=lambda x: FILL_METHODS[x], key='fill_m')
    if st.button('执行填充', key='fill_btn'):
        userid = st.session_state.user_info['userid']
        with st.spinner('处理中...'):
            response = api_client.fill_missing(f"{fill_sd} 00:00:00", f"{fill_ed} 23:59:59", fill_method, userid)
        if response['code'] == 200:
            show_success(f"填充完成，共填充 {response['data']['filled_count']} 个缺失值")
        else:
            show_error(response['message'])
    st.markdown('---')
    st.markdown('#### 数据归一化')
    c1, c2 = st.columns(2)
    with c1:
        norm_sd = st.date_input('开始日期', value=mn, min_value=mn, max_value=mx, key='norm_s')
    with c2:
        norm_ed = st.date_input('结束日期', value=min(mn + timedelta(days=30), mx), min_value=mn, max_value=mx, key='norm_e')
    if st.button('执行归一化', key='norm_btn'):
        userid = st.session_state.user_info['userid']
        with st.spinner('处理中...'):
            response = api_client.normalize_data(f"{norm_sd} 00:00:00", f"{norm_ed} 23:59:59", 'minmax', userid)
        if response['code'] == 200:
            show_success(f"归一化完成，共处理 {response['data']['count']} 条数据")
        else:
            show_error(response['message'])
    st.markdown('---')
    st.markdown('#### 相关性分析')
    c1, c2 = st.columns(2)
    with c1:
        corr_sd = st.date_input('开始日期', value=mn, min_value=mn, max_value=mx, key='corr_s')
    with c2:
        corr_ed = st.date_input('结束日期', value=min(mn + timedelta(days=30), mx), min_value=mn, max_value=mx, key='corr_e')
    if st.button('执行分析', type='primary', key='corr_btn'):
        userid = st.session_state.user_info['userid']
        with st.spinner('分析中...'):
            response = api_client.correlation_analysis(f"{corr_sd} 00:00:00", f"{corr_ed} 23:59:59", userid)
        if response['code'] == 200:
            show_success('相关性分析完成')
            corr_data = response['data']['correlation']
            df_corr = pd.DataFrame(list(corr_data.items()), columns=['特征', '相关系数'])
            st.dataframe(df_corr, use_container_width=True)
            if 'chart' in response['data']:
                display_base64_image(response['data']['chart'])
        else:
            show_error(response['message'])

def show_data_visualization(api_client):
    """数据可视化"""
    st.subheader('📈 数据可视化')
    mn, mx = _get_data_range()
    c1, c2 = st.columns(2)
    with c1:
        vis_sd = st.date_input('开始日期', value=mn, min_value=mn, max_value=mx, key='vis_s')
    with c2:
        vis_ed = st.date_input('结束日期', value=min(mn + timedelta(days=7), mx), min_value=mn, max_value=mx, key='vis_e')
    if st.button('生成负荷曲线', type='primary', key='vis_btn'):
        with st.spinner('生成中...'):
            response = api_client.visualize_load_curve(f"{vis_sd} 00:00:00", f"{vis_ed} 23:59:59")
        if response['code'] == 200:
            show_success('负荷曲线生成成功')
            display_base64_image(response['data']['chart'])
        else:
            show_error(response['message'])

