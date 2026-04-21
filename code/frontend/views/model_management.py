# 模型管理页面（模型工程师）
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, date
import json
from ui_utils.api_client import APIClient
from ui_utils.helpers import show_success, show_error, show_api_response, display_base64_image
from config import MODEL_TYPES

def _get_train_date_defaults():
    """从数据库获取训练日期默认值"""
    try:
        api_client = APIClient()
        resp = api_client.get_data_date_range()
        if resp and resp.get('code') == 200 and resp['data']:
            mn = datetime.strptime(resp['data']['min_date'], '%Y-%m-%d %H:%M:%S').date()
            mx = datetime.strptime(resp['data']['max_date'], '%Y-%m-%d %H:%M:%S').date()
            total_days = (mx - mn).days
            # 训练集：前80%，验证集：后20%
            split = mn + timedelta(days=int(total_days * 0.8))
            return mn, split - timedelta(days=1), split, mx
    except Exception:
        pass
    return date(2011,1,1), date(2017,6,30), date(2017,7,1), date(2018,8,2)

def show_model_management_page():
    """模型管理主页面"""
    st.title('🤖 模型管理')
    tab1, tab2, tab3, tab4 = st.tabs(['模型配置', '模型训练', '训练记录', '模型版本'])
    api_client = APIClient()
    with tab1:
        show_model_config(api_client)
    with tab2:
        show_model_training(api_client)
    with tab3:
        show_train_records(api_client)
    with tab4:
        show_model_versions(api_client)

def show_model_config(api_client):
    """模型配置"""
    st.subheader('⚙️ 模型配置管理')
    filter_type = st.selectbox('筛选模型类型', ['全部'] + list(MODEL_TYPES.keys()),
                               format_func=lambda x: '全部' if x == '全部' else MODEL_TYPES[x], key='mc_filter')
    modeltype = None if filter_type == '全部' else filter_type
    response = api_client.get_model_configs(modeltype)
    if response['code'] == 200:
        configs = response['data']['configs']
        if configs:
            for config in configs:
                with st.expander(f"📋 {config['modelname']} ({MODEL_TYPES[config['modeltype']]})"):
                    st.write(f"**配置ID:** {config['configid']}　|　**模型类型:** {MODEL_TYPES[config['modeltype']]}　|　**创建时间:** {config['createtime']}")
                    st.json(config['hyperparams'])
        else:
            st.info('暂无模型配置')
    st.markdown('---')
    st.markdown('### 创建新模型配置')
    with st.form('create_config_form'):
        modelname = st.text_input('模型名称', placeholder='例如：LSTM_v2', key='mc_name')
        modeltype = st.selectbox('模型类型', list(MODEL_TYPES.keys()), format_func=lambda x: MODEL_TYPES[x], key='mc_type')
        architecture = st.text_area('架构描述', key='mc_arch')
        st.write('**超参数配置：**')
        if modeltype == 'lstm':
            c1, c2 = st.columns(2)
            with c1:
                hs = st.number_input('隐藏层大小', 32, 512, 128, 32, key='mc_hs')
                nl = st.number_input('LSTM层数', 1, 5, 2, key='mc_nl')
            with c2:
                dp = st.slider('Dropout', 0.0, 0.5, 0.2, 0.05, key='mc_dp')
                lr = st.number_input('学习率', 0.0001, 0.01, 0.001, format='%.4f', key='mc_lr')
            hyperparams = {'hidden_size': hs, 'num_layers': nl, 'dropout': dp, 'learning_rate': lr}
        elif modeltype == 'transformer':
            c1, c2 = st.columns(2)
            with c1:
                dm = st.number_input('模型维度', 64, 512, 256, 64, key='mc_dm')
                nh = st.selectbox('注意力头数', [2,4,8,16], index=2, key='mc_nh')
            with c2:
                nl = st.number_input('Transformer层数', 1, 8, 4, key='mc_tnl')
                dp = st.slider('Dropout', 0.0, 0.5, 0.1, 0.05, key='mc_tdp')
            lr = st.number_input('学习率', 0.00001, 0.001, 0.0001, format='%.5f', key='mc_tlr')
            hyperparams = {'d_model': dm, 'nhead': nh, 'num_layers': nl, 'dropout': dp, 'learning_rate': lr}
        else:
            c1, c2 = st.columns(2)
            with c1:
                lh = st.number_input('LSTM隐藏层', 32, 256, 64, 32, key='mc_lh')
                tl = st.number_input('Transformer层数', 1, 4, 2, key='mc_htl')
            with c2:
                nh = st.selectbox('注意力头数', [2,4,8], index=1, key='mc_hnh')
                dp = st.slider('Dropout', 0.0, 0.5, 0.15, 0.05, key='mc_hdp')
            lr = st.number_input('学习率', 0.0001, 0.01, 0.0005, format='%.4f', key='mc_hlr')
            hyperparams = {'lstm_hidden': lh, 'transformer_layers': tl, 'nhead': nh, 'dropout': dp, 'learning_rate': lr}
        if st.form_submit_button('创建配置', type='primary'):
            if not modelname or not architecture:
                show_error('请填写模型名称和架构描述')
            else:
                userid = st.session_state.user_info['userid']
                resp = api_client.create_model_config(modelname, modeltype, hyperparams, architecture, userid)
                if show_api_response(resp):
                    st.balloons()

def show_model_training(api_client):
    """模型训练"""
    st.subheader('🎯 模型训练')
    response = api_client.get_model_configs()
    if response['code'] != 200 or not response['data']['configs']:
        st.warning('请先创建模型配置'); return
    configs = response['data']['configs']
    config_options = {f"{c['modelname']} (ID:{c['configid']})": c['configid'] for c in configs}
    selected = st.selectbox('选择模型配置', list(config_options.keys()), key='mt_sel')
    configid = config_options[selected]
    ts_def, te_def, vs_def, ve_def = _get_train_date_defaults()
    st.markdown('---')
    c1, c2 = st.columns(2)
    with c1:
        ts = st.date_input('训练开始', value=ts_def, min_value=date(2000,1,1), max_value=date(2030,12,31), key='mt_ts')
        te = st.date_input('训练结束', value=te_def, min_value=date(2000,1,1), max_value=date(2030,12,31), key='mt_te')
    with c2:
        vs = st.date_input('验证开始', value=vs_def, min_value=date(2000,1,1), max_value=date(2030,12,31), key='mt_vs')
        ve = st.date_input('验证结束', value=ve_def, min_value=date(2000,1,1), max_value=date(2030,12,31), key='mt_ve')
    c3, c4, c5 = st.columns(3)
    with c3:
        epochs = st.number_input('训练轮数', 10, 500, 100, 10, key='mt_ep')
    with c4:
        bs = st.number_input('批次大小', 8, 128, 32, 8, key='mt_bs')
    with c5:
        sl = st.number_input('序列长度', 12, 72, 24, 12, key='mt_sl')
    st.info('⚠️ 模型训练可能需要较长时间，请耐心等待')
    if st.button('开始训练', type='primary', key='mt_btn'):
        userid = st.session_state.user_info['userid']
        with st.spinner('模型训练中...'):
            resp = api_client.train_model(configid, f"{ts} 00:00:00", f"{te} 23:59:59", f"{vs} 00:00:00", f"{ve} 23:59:59", epochs, bs, sl, userid)
        if resp['code'] == 200:
            d = resp['data']
            show_success('模型训练完成！')
            c1, c2, c3 = st.columns(3)
            with c1: st.metric('训练ID', d['trainid'])
            with c2: st.metric('训练损失', f"{d['train_loss']:.6f}")
            with c3: st.metric('验证损失', f"{d['valid_loss']:.6f}")
            if 'chart' in d:
                display_base64_image(d['chart'])
        else:
            show_error(resp['message'])

def show_train_records(api_client):
    """训练记录"""
    st.subheader('📊 训练记录')
    response = api_client.get_train_records()
    if response['code'] == 200:
        records = response['data']['records']
        if records:
            for r in records:
                icon = {'completed':'✅','running':'🔄','failed':'❌'}.get(r['trainstatus'],'❓')
                with st.expander(f"{icon} 训练ID:{r['trainid']} - {r['trainstatus']}"):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.write(f"**配置ID:** {r['configid']}"); st.write(f"**训练数据:** {r['traindata']}")
                    with c2:
                        st.write(f"**轮数:** {r['epochs']}"); st.write(f"**批次:** {r['batchsize']}")
                    with c3:
                        if r['trainloss']: st.metric('训练损失', f"{r['trainloss']:.6f}")
                        if r['validloss']: st.metric('验证损失', f"{r['validloss']:.6f}")
                    st.write(f"**开始:** {r['starttime']}　**结束:** {r['endtime']}")
        else:
            st.info('暂无训练记录')
    else:
        show_error(response['message'])

def show_model_versions(api_client):
    """模型版本管理"""
    st.subheader('📦 模型版本管理')
    response = api_client.get_model_versions()
    if response['code'] == 200:
        versions = response['data']['versions']
        if versions:
            for v in versions:
                badge = '🟢 激活' if v['isactive'] == 1 else '⚪ 未激活'
                with st.expander(f"{badge} 版本 {v['versionnumber']} (ID:{v['versionid']})"):
                    st.write(f"**训练ID:** {v['trainid']}　|　**描述:** {v['versiondesc']}　|　**创建:** {v['createtime']}")
                    if v['performance']:
                        p = v['performance']
                        c1,c2,c3,c4 = st.columns(4)
                        with c1: st.metric('MAE', f"{p.get('mae',0):.2f}")
                        with c2: st.metric('RMSE', f"{p.get('rmse',0):.2f}")
                        with c3: st.metric('MAPE', f"{p.get('mape',0):.2f}%")
                        with c4: st.metric('R²', f"{p.get('r2',0):.4f}")
                    if v['isactive'] == 0:
                        if st.button('激活此版本', key=f"act_{v['versionid']}"):
                            r2 = api_client.activate_model_version(v['versionid'])
                            if show_api_response(r2): st.rerun()
        else:
            st.info('暂无模型版本')
    st.markdown('---')
    st.markdown('### 创建新版本')
    with st.form('create_ver_form'):
        train_resp = api_client.get_train_records()
        if train_resp['code'] == 200:
            records = [r for r in train_resp['data']['records'] if r['trainstatus'] == 'completed']
            if records:
                opts = {f"训练ID:{r['trainid']} - {r['starttime']}": r['trainid'] for r in records}
                sel = st.selectbox('选择训练记录', list(opts.keys()), key='cv_sel')
                trainid = opts[sel]
                vn = st.text_input('版本号', placeholder='v1.0.0', key='cv_vn')
                vd = st.text_area('版本描述', key='cv_vd')
                c1,c2,c3,c4 = st.columns(4)
                with c1: mae = st.number_input('MAE', 0.0, value=100.0, step=0.1, key='cv_mae')
                with c2: rmse = st.number_input('RMSE', 0.0, value=150.0, step=0.1, key='cv_rmse')
                with c3: mape = st.number_input('MAPE(%)', 0.0, value=2.0, step=0.1, key='cv_mape')
                with c4: r2s = st.number_input('R²', 0.0, 1.0, 0.95, 0.01, key='cv_r2')
                ia = st.checkbox('设为激活版本', key='cv_ia')
                if st.form_submit_button('创建版本', type='primary'):
                    if not vn or not vd:
                        show_error('请填写版本号和描述')
                    else:
                        perf = {'mae':mae,'rmse':rmse,'mape':mape,'r2':r2s}
                        resp = api_client.create_model_version(trainid, vn, vd, perf, 1 if ia else 0)
                        if show_api_response(resp): st.balloons()
            else:
                st.warning('没有已完成的训练记录')

