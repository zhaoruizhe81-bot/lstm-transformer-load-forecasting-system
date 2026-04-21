# Streamlit工具函数
import streamlit as st
import base64
from io import BytesIO
import pandas as pd

def init_session_state():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'login'

def check_login():
    if not st.session_state.get('logged_in', False):
        st.warning('请先登录')
        st.stop()

def logout():
    st.session_state.logged_in = False
    st.session_state.user_info = None
    st.session_state.current_page = 'login'
    st.rerun()

def display_base64_image(base64_str):
    if base64_str and base64_str.startswith('data:image'):
        st.image(base64_str)
    else:
        st.error('图片格式错误')

def format_datetime(dt_str):
    if dt_str:
        return dt_str.replace('T', ' ')
    return ''

def show_success(message):
    st.success(f'✅ {message}')

def show_error(message):
    st.error(f'❌ {message}')

def show_info(message):
    st.info(f'ℹ️ {message}')

def show_warning(message):
    st.warning(f'⚠️ {message}')

def create_download_link(data, filename, file_label='下载文件'):
    import json
    json_str = json.dumps(data, ensure_ascii=False, indent=2)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:application/json;base64,{b64}" download="{filename}">{file_label}</a>'
    st.markdown(href, unsafe_allow_html=True)

def dataframe_to_csv(df):
    return df.to_csv(index=False).encode('utf-8-sig')

def show_api_response(response):
    if response['code'] == 200:
        show_success(response['message'])
        return True
    else:
        show_error(response['message'])
        return False

