# 登录页面
import streamlit as st
from ui_utils.api_client import APIClient
from ui_utils.helpers import show_success, show_error
from config import ROLE_NAMES

def show_login_page():
    """显示登录页面"""
    st.title('⚡ 电力负荷预测系统')
    st.markdown('---')
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.subheader('🔐 用户登录')
        
        with st.form('login_form'):
            username = st.text_input('用户名', placeholder='请输入用户名')
            password = st.text_input('密码', type='password', placeholder='请输入密码')
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                login_btn = st.form_submit_button('登录', use_container_width=True)
            with col_btn2:
                reset_btn = st.form_submit_button('忘记密码', use_container_width=True)
            
            if login_btn:
                if not username or not password:
                    show_error('请输入用户名和密码')
                else:
                    api_client = APIClient()
                    response = api_client.login(username, password)
                    
                    if response['code'] == 200:
                        st.session_state.logged_in = True
                        st.session_state.user_info = response['data']
                        st.session_state.current_page = 'home'
                        show_success(f"欢迎回来，{response['data']['realname']}！")
                        st.rerun()
                    else:
                        show_error(response['message'])
            
            if reset_btn:
                st.session_state.current_page = 'reset_password'
                st.rerun()
        
        with st.expander('📋 测试账号'):
            st.markdown("""
            | 用户名 | 密码 | 角色 |
            |--------|------|------|
            | admin | 123456 | 系统管理员 |
            | analyst01 | 123456 | 数据分析师 |
            | engineer01 | 123456 | 模型工程师 |
            | business01 | 123456 | 业务用户 |
            """)

def show_reset_password_page():
    """显示重置密码页面"""
    st.title('🔑 重置密码')
    st.markdown('---')
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form('reset_form'):
            username = st.text_input('用户名')
            st.info('请回答您设置的安全问题')
            security_answer = st.text_input('安全问题答案')
            new_password = st.text_input('新密码', type='password')
            confirm_password = st.text_input('确认新密码', type='password')
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                submit_btn = st.form_submit_button('重置密码', use_container_width=True)
            with col_btn2:
                back_btn = st.form_submit_button('返回登录', use_container_width=True)
            
            if submit_btn:
                if not all([username, security_answer, new_password, confirm_password]):
                    show_error('请填写所有字段')
                elif new_password != confirm_password:
                    show_error('两次输入的密码不一致')
                else:
                    api_client = APIClient()
                    response = api_client.reset_password(username, security_answer, new_password)
                    
                    if response['code'] == 200:
                        show_success('密码重置成功，请使用新密码登录')
                        st.session_state.current_page = 'login'
                        st.rerun()
                    else:
                        show_error(response['message'])
            
            if back_btn:
                st.session_state.current_page = 'login'
                st.rerun()

