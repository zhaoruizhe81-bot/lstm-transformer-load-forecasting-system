# Streamlit主入口文件
import streamlit as st
import sys
import os

# 确保路径正确
_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from config import PAGE_TITLE, PAGE_ICON, ROLE_NAMES

# 页面配置（必须最先调用）
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_info' not in st.session_state:
    st.session_state.user_info = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'login'

# 自定义CSS
st.markdown("""
<style>
    .user-info-box {background:#f0f2f6;padding:1rem;border-radius:.5rem;margin-bottom:1rem}
</style>
""", unsafe_allow_html=True)


# ========== 未登录：显示登录页 ==========
if not st.session_state.logged_in:
    from views.login import show_login_page, show_reset_password_page
    if st.session_state.current_page == 'reset_password':
        show_reset_password_page()
    else:
        show_login_page()
    st.stop()


# ========== 已登录：显示侧边栏+主内容 ==========
user_info = st.session_state.user_info
role = user_info['role']
role_name = ROLE_NAMES.get(role, role)

# --- 侧边栏 ---
with st.sidebar:
    st.markdown(f"## {PAGE_ICON} {PAGE_TITLE}")
    st.markdown('---')
    st.markdown(f"""
<div class="user-info-box">
<b>👤 {user_info['realname']}</b><br>
角色：{role_name}<br>
用户名：{user_info['username']}
</div>
""", unsafe_allow_html=True)
    st.markdown('---')

    # 根据角色生成菜单
    if role == 'admin':
        menu_options = {'🏠 首页':'home','👥 用户管理':'admin','📊 数据管理':'data','🤖 模型管理':'model','🔮 预测管理':'predict'}
    elif role == 'analyst':
        menu_options = {'🏠 首页':'home','📊 数据管理':'data'}
    elif role == 'engineer':
        menu_options = {'🏠 首页':'home','🤖 模型管理':'model'}
    elif role == 'business':
        menu_options = {'🏠 首页':'home','🔮 预测管理':'predict'}
    else:
        menu_options = {'🏠 首页':'home'}

    for label, page_key in menu_options.items():
        if st.button(label, key=f'nav_{page_key}', use_container_width=True):
            st.session_state.current_page = page_key

    st.markdown('---')
    if st.button('🚪 退出登录', type='primary', use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.user_info = None
        st.session_state.current_page = 'login'
        st.rerun()

# --- 主内容路由 ---
page = st.session_state.get('current_page', 'home')

if page == 'home':
    # 首页内联显示
    st.title('🏠 欢迎使用电力负荷预测系统')
    st.markdown('---')
    st.markdown(f"### 👋 欢迎，{user_info['realname']}！您当前的角色是：**{role_name}**")
    st.markdown('本系统基于 LSTM-Transformer 深度学习技术，提供准确的城市电网短期负荷预测服务。')
    st.markdown('---')
    st.subheader('📚 系统功能')
    c1, c2 = st.columns(2)
    with c1:
        st.info("**📊 数据管理** — 数据上传、查询、异常检测、预处理、可视化")
        st.info("**🤖 模型管理** — LSTM/Transformer/混合模型配置、训练、版本管理")
    with c2:
        st.info("**🔮 预测管理** — 创建任务、执行预测、结果分析、误差评估")
        st.info("**⚙️ 系统管理** — 用户管理、操作日志、告警处理、系统配置")
    st.markdown('---')
    st.subheader('ℹ️ 系统信息')
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric('系统版本','v1.0.0')
    with c2: st.metric('后端框架','Flask')
    with c3: st.metric('前端框架','Streamlit')
    with c4: st.metric('深度学习','PyTorch')

elif page == 'admin':
    try:
        from views.admin import show_admin_page
        show_admin_page()
    except Exception as e:
        import traceback
        st.error(f'页面加载失败: {e}')
        st.code(traceback.format_exc())

elif page == 'data':
    try:
        from views.data_management import show_data_management_page
        show_data_management_page()
    except Exception as e:
        import traceback
        st.error(f'页面加载失败: {e}')
        st.code(traceback.format_exc())

elif page == 'model':
    try:
        from views.model_management import show_model_management_page
        show_model_management_page()
    except Exception as e:
        import traceback
        st.error(f'页面加载失败: {e}')
        st.code(traceback.format_exc())

elif page == 'predict':
    try:
        from views.predict_management import show_predict_management_page
        show_predict_management_page()
    except Exception as e:
        import traceback
        st.error(f'页面加载失败: {e}')
        st.code(traceback.format_exc())

else:
    st.title('🏠 首页')
    st.write('请从左侧菜单选择功能')

