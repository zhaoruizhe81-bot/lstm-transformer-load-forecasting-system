# 系统管理页面（系统管理员）
import streamlit as st
import pandas as pd
from datetime import datetime
from ui_utils.api_client import APIClient
from ui_utils.helpers import show_success, show_error, show_api_response
from config import ROLE_NAMES

def show_admin_page():
    """系统管理主页面"""
    if st.session_state.user_info['role'] != 'admin':
        st.error('⛔ 您没有权限访问此页面'); return
    st.title('⚙️ 系统管理')
    tab1, tab2, tab3, tab4 = st.tabs(['用户管理', '操作日志', '告警管理', '系统配置'])
    api_client = APIClient()
    with tab1: show_user_management(api_client)
    with tab2: show_operation_logs(api_client)
    with tab3: show_alert_management(api_client)
    with tab4: show_system_config(api_client)

def show_user_management(api_client):
    """用户管理"""
    st.subheader('👥 用户管理')
    response = api_client.get_all_users()
    if response['code'] == 200:
        users = response['data']['users']
        st.write(f'**当前用户数：{len(users)}**')
        for u in users:
            si = '🟢' if u['status'] == 1 else '🔴'
            rn = ROLE_NAMES.get(u['role'], u['role'])
            with st.expander(f"{si} {u['username']} - {u['realname']} ({rn})"):
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"**ID:** {u['userid']}　**用户名:** {u['username']}　**姓名:** {u['realname']}")
                    st.write(f"**邮箱:** {u['email']}　**手机:** {u['phone']}")
                with c2:
                    st.write(f"**角色:** {rn}　**状态:** {'启用' if u['status']==1 else '禁用'}")
                    st.write(f"**创建时间:** {u['createtime']}")
                b1, b2, b3 = st.columns(3)
                with b1:
                    if u['status'] == 1:
                        if st.button('🔒 禁用', key=f"dis_{u['userid']}"):
                            r = api_client.update_user_status(u['userid'], status=0)
                            if show_api_response(r): st.rerun()
                    else:
                        if st.button('🔓 启用', key=f"en_{u['userid']}"):
                            r = api_client.update_user_status(u['userid'], status=1)
                            if show_api_response(r): st.rerun()
                with b3:
                    if st.button('🗑️ 删除', key=f"del_{u['userid']}"):
                        if u['userid'] == st.session_state.user_info['userid']:
                            show_error('不能删除当前登录用户')
                        else:
                            r = api_client.delete_user(u['userid'])
                            if show_api_response(r): st.rerun()
    else:
        show_error(response['message'])
    st.markdown('---')
    st.markdown('### ➕ 创建新用户')
    with st.form('create_user_form'):
        c1, c2 = st.columns(2)
        with c1:
            nu = st.text_input('用户名*', key='cu_un')
            np_ = st.text_input('密码*', value='123456', type='password', key='cu_pw')
            nrn = st.text_input('真实姓名*', key='cu_rn')
            ne = st.text_input('邮箱', key='cu_em')
        with c2:
            nph = st.text_input('手机号', key='cu_ph')
            nr = st.selectbox('角色*', list(ROLE_NAMES.keys()), format_func=lambda x: ROLE_NAMES[x], key='cu_rl')
            nsq = st.text_input('安全问题', key='cu_sq')
            nsa = st.text_input('安全答案', key='cu_sa')
        if st.form_submit_button('创建用户', type='primary'):
            if not all([nu, np_, nrn]):
                show_error('请填写必填字段')
            else:
                aid = st.session_state.user_info['userid']
                r = api_client.create_user(nu, np_, nrn, ne, nph, nr, nsq, nsa, aid)
                if show_api_response(r): st.balloons()

def show_operation_logs(api_client):
    """操作日志"""
    st.subheader('📝 操作日志')
    c1, c2 = st.columns([3, 1])
    with c1:
        page = st.number_input('页码', min_value=1, value=1, key='log_p')
    with c2:
        ps = st.selectbox('每页条数', [20,50,100], index=1, key='log_ps')
    if st.button('查询日志', type='primary', key='log_btn'):
        response = api_client.get_operation_logs(page, ps)
        if response['code'] == 200:
            d = response['data']
            st.write(f"**共 {d['total']} 条日志，当前第 {d['page']} 页**")
            if d['logs']:
                st.dataframe(pd.DataFrame(d['logs']), use_container_width=True)
            else:
                st.info('暂无日志记录')
        else:
            show_error(response['message'])

def show_alert_management(api_client):
    """告警管理"""
    st.subheader('🚨 告警管理')
    sf = st.selectbox('告警状态', ['全部','未处理','已处理'], key='al_sf')
    status = None if sf == '全部' else (0 if sf == '未处理' else 1)
    response = api_client.get_alert_records(status)
    if response['code'] == 200:
        alerts = response['data']['alerts']
        if alerts:
            for a in alerts:
                li = {'info':'ℹ️','warning':'⚠️','error':'❌','critical':'🔥'}.get(a['alertlevel'],'❓')
                st2 = '✅ 已处理' if a['handlestatus']==1 else '⏳ 未处理'
                with st.expander(f"{li} {a['alerttype']} - {st2}"):
                    st.write(f"**ID:** {a['alertid']}　**类型:** {a['alerttype']}　**级别:** {a['alertlevel']}　**时间:** {a['alerttime']}")
                    st.info(a['alertmessage'])
                    if a['handlestatus'] == 0:
                        if st.button('✅ 标记已处理', key=f"ha_{a['alertid']}"):
                            uid = st.session_state.user_info['userid']
                            r = api_client.handle_alert(a['alertid'], uid)
                            if show_api_response(r): st.rerun()
        else:
            st.info('暂无告警记录')
    else:
        show_error(response['message'])

def show_system_config(api_client):
    """系统配置"""
    st.subheader('🔧 系统配置')
    response = api_client.get_system_config()
    if response['code'] == 200:
        configs = response['data']
        rows = [{'配置项': k, '配置值': v['value'], '说明': v['desc']} for k, v in configs.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        show_error(response['message'])

