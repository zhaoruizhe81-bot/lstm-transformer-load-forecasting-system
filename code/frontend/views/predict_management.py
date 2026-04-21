# 预测管理页面（业务用户）
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, date
import json
from ui_utils.api_client import APIClient
from ui_utils.helpers import show_success, show_error, show_api_response, display_base64_image, create_download_link

def _get_predict_date_defaults():
    """获取预测日期默认值：数据范围末尾往后推一周"""
    try:
        api_client = APIClient()
        resp = api_client.get_data_date_range()
        if resp and resp.get('code') == 200 and resp['data']:
            max_date = datetime.strptime(resp['data']['max_date'], '%Y-%m-%d %H:%M:%S').date()
            pred_start = max_date + timedelta(days=1)
            pred_end = max_date + timedelta(days=7)
            return pred_start, pred_end
    except Exception:
        pass
    return date(2018, 8, 3), date(2018, 8, 9)

def show_predict_management_page():
    """预测管理主页面"""
    st.title('🔮 预测管理')
    tab1, tab2, tab3, tab4 = st.tabs(['创建任务', '任务列表', '预测结果', '结果分析'])
    api_client = APIClient()
    with tab1: show_create_task(api_client)
    with tab2: show_task_list(api_client)
    with tab3: show_predict_results(api_client)
    with tab4: show_result_analysis(api_client)

def show_create_task(api_client):
    """创建预测任务"""
    st.subheader('➕ 创建预测任务')
    response = api_client.get_model_versions()
    if response['code'] != 200:
        show_error(response['message']); return
    versions = response['data']['versions']
    active_versions = [v for v in versions if v['isactive'] == 1]
    if not active_versions:
        st.warning('⚠️ 没有激活的模型版本，请先在模型管理中激活一个版本'); return
    pred_start_def, pred_end_def = _get_predict_date_defaults()
    with st.form('create_task_form'):
        taskname = st.text_input('任务名称', placeholder='例如：2018年8月第一周预测', key='ct_name')
        vopts = {f"版本 {v['versionnumber']} (ID:{v['versionid']})": v['versionid'] for v in active_versions}
        sel = st.selectbox('选择模型版本', list(vopts.keys()), key='ct_ver')
        versionid = vopts[sel]
        c1, c2 = st.columns(2)
        with c1:
            psd = st.date_input('预测开始日期', value=pred_start_def,
                                min_value=date(2000, 1, 1), max_value=date(2030, 12, 31), key='ct_psd')
        with c2:
            ped = st.date_input('预测结束日期', value=pred_end_def,
                                min_value=date(2000, 1, 1), max_value=date(2030, 12, 31), key='ct_ped')
        st.info('💡 预测时间间隔为6小时，系统将自动生成预测时间点')
        if st.form_submit_button('创建任务', type='primary'):
            if not taskname:
                show_error('请输入任务名称')
            else:
                userid = st.session_state.user_info['userid']
                resp = api_client.create_predict_task(taskname, versionid, f"{psd} 00:00:00", f"{ped} 23:59:59", userid)
                if show_api_response(resp): st.balloons()

def show_task_list(api_client):
    """任务列表"""
    st.subheader('📋 预测任务列表')
    status_map = {'全部':None,'pending':'pending','running':'running','completed':'completed','failed':'failed'}
    label_map = {'全部':'全部','pending':'待执行','running':'运行中','completed':'已完成','failed':'失败'}
    sf = st.selectbox('任务状态', list(status_map.keys()), format_func=lambda x: label_map[x], key='tl_sf')
    userid = st.session_state.user_info['userid']
    response = api_client.get_predict_tasks(userid, status_map[sf])
    if response['code'] != 200:
        show_error(response['message']); return
    tasks = response['data']['tasks']
    if not tasks:
        st.info('暂无预测任务'); return
    icon_map = {'pending':'⏳','running':'🔄','completed':'✅','failed':'❌'}
    for t in tasks:
        ic = icon_map.get(t['taskstatus'],'❓')
        txt = label_map.get(t['taskstatus'], t['taskstatus'])
        with st.expander(f"{ic} {t['taskname']} - {txt}"):
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**任务ID:** {t['taskid']}　|　**模型版本:** {t['versionid']}　|　**状态:** {txt}")
                st.write(f"**预测范围:** {t['predictstart']} ~ {t['predictend']}")
            with c2:
                st.write(f"**创建时间:** {t['createtime']}")
                if t['executetime']: st.write(f"**执行时间:** {t['executetime']}")
                if t['taskstatus'] == 'pending':
                    if st.button('▶️ 执行任务', key=f"exec_{t['taskid']}"):
                        with st.spinner('执行中...'):
                            r = api_client.execute_predict_task(t['taskid'], userid)
                        if show_api_response(r): st.rerun()

def show_predict_results(api_client):
    """预测结果"""
    st.subheader('📊 预测结果查看')
    userid = st.session_state.user_info['userid']
    response = api_client.get_predict_tasks(userid, 'completed')
    if response['code'] != 200:
        show_error(response['message']); return
    tasks = response['data']['tasks']
    if not tasks:
        st.info('暂无已完成的预测任务'); return
    topts = {f"{t['taskname']} (ID:{t['taskid']})": t['taskid'] for t in tasks}
    sel = st.selectbox('选择任务', list(topts.keys()), key='pr_sel')
    taskid = topts[sel]
    rr = api_client.get_predict_results(taskid)
    if rr['code'] != 200:
        show_error(rr['message']); return
    results = rr['data']['results']
    if not results:
        st.info('该任务暂无预测结果'); return
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button('🔄 更新实际值', key='pr_upd'):
            with st.spinner('更新中...'):
                ur = api_client.update_actual_values(taskid)
            show_api_response(ur)
    with c2:
        if st.button('📈 计算误差指标', key='pr_met'):
            with st.spinner('计算中...'):
                mr = api_client.calculate_error_metrics(taskid)
            if mr['code'] == 200:
                show_success('误差指标计算完成')
                m = mr['data']
                a,b,c,d = st.columns(4)
                with a: st.metric('MAE', f"{m['mae']:.2f}")
                with b: st.metric('RMSE', f"{m['rmse']:.2f}")
                with c: st.metric('MAPE', f"{m['mape']:.2f}%")
                with d: st.metric('R²', f"{m['r2score']:.4f}")
            else:
                show_error(mr['message'])
    with c3:
        if st.button('📥 导出结果', key='pr_exp'):
            er = api_client.export_predict_results(taskid)
            if er['code'] == 200:
                show_success('结果导出成功')
                create_download_link(er['data'], f'predict_{taskid}.json', '点击下载JSON')
            else:
                show_error(er['message'])

def show_result_analysis(api_client):
    """结果分析"""
    st.subheader('📈 预测结果分析')
    userid = st.session_state.user_info['userid']
    response = api_client.get_predict_tasks(userid, 'completed')
    if response['code'] != 200:
        show_error(response['message']); return
    tasks = response['data']['tasks']
    if not tasks:
        st.info('暂无已完成的预测任务'); return
    topts = {f"{t['taskname']} (ID:{t['taskid']})": t['taskid'] for t in tasks}
    sel = st.selectbox('选择任务', list(topts.keys()), key='ra_sel')
    taskid = topts[sel]
    if st.button('生成可视化分析', type='primary', key='ra_btn'):
        with st.spinner('生成中...'):
            vr = api_client.visualize_prediction(taskid)
        if vr['code'] == 200:
            show_success('可视化分析生成成功')
            data = vr['data']
            if 'comparison_chart' in data:
                st.write('**预测值与实际值对比：**')
                display_base64_image(data['comparison_chart'])
            if 'scatter_chart' in data:
                st.write('**预测散点图：**')
                display_base64_image(data['scatter_chart'])
            if 'chart' in data and 'comparison_chart' not in data:
                st.write('**预测负荷曲线：**')
                display_base64_image(data['chart'])
                st.info('💡 更新实际值后可查看对比分析')
        else:
            show_error(vr['message'])

