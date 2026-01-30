import streamlit as st
import pandas as pd
import numpy as np
from modules.data_loader import DataLoader
from modules.data_analyzer import DataAnalyzer
from modules.code_generator import CodeGenerator
from modules.result_summarizer import ResultSummarizer
import os

# 设置页面配置
st.set_page_config(
    page_title="中西循真 - 临床疗效评价智能体",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 应用常量
APP_NAME = '中西循真临床疗效评价智能体'
VERSION = "version 0.01"
# 初始化用户数据库（简单实现，实际应用中应使用数据库）
if "users_db" not in st.session_state:
    st.session_state.users_db = {
        "wmq1009": "12345"
    }
# 参考ollama模型选项，更新可用模型列表
AVAILABLE_MODELS = [
    "gpt-4o-mini", 
    "gpt-4o", 
    "gpt-3.5-turbo",
    "qwen3",
    "qwen3-large",
    "qwen3-vl",
    "qwen3-coder",
    "deepseek-r1",
    "deepseek-chat",
    "gemma3",
    "glm-4.6"
]

# 自动安装缺失的模块
from modules.auto_install import install_missing_modules, ensure_matplotlib_pyplot

# 确保matplotlib.pyplot可以正常导入
ensure_matplotlib_pyplot()

# 安装必需的模块
required_modules = ["pandas", "numpy", "matplotlib", "seaborn", "scipy", "openai", "requests"]
install_missing_modules(required_modules)

# 初始化模块
loader = DataLoader()
analyzer = DataAnalyzer()
code_gen = CodeGenerator()
result_summarizer = ResultSummarizer()

# 初始化会话状态
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "api_key" not in st.session_state:
    st.session_state.api_key = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = AVAILABLE_MODELS[3]
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "df" not in st.session_state:
    st.session_state.df = None
if "data_types" not in st.session_state:
    st.session_state.data_types = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "show_register" not in st.session_state:
    st.session_state.show_register = False

# 自定义CSS样式，增加科技感
st.markdown("""
<style>
    /* 重置默认样式 */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    /* 主容器样式 - 改为白色到浅蓝色的渐变 */
    .main {
        background: linear-gradient(135deg, #ffffff 0%, #e3f2fd 50%, #bbdefb 100%);
        color: #333333;
        overflow: hidden;
    }
    
    /* 标题样式 */
    h1, h2, h3, h4, h5, h6 {
        color: #1976d2;
        font-weight: bold;
        text-shadow: 0 0 5px rgba(25, 118, 210, 0.3);
    }
    
    /* 卡片样式 */
    .stCard {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(25, 118, 210, 0.2);
    }
    
    /* 按钮样式 */
    .stButton > button {
        background: linear-gradient(45deg, #1976d2 0%, #2196f3 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        box-shadow: 0 2px 10px rgba(25, 118, 210, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(25, 118, 210, 0.4);
    }
    
    /* 输入框样式 */
    .stTextInput > div > input,
    .stTextArea > div > textarea,
    .stFileUploader > div > div,
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(25, 118, 210, 0.3);
        border-radius: 8px;
        color: #333333;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* 数据框样式 */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 8px;
        border: 1px solid rgba(25, 118, 210, 0.2);
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    /* 聊天容器样式 */
    .chat-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 15px;
        height: 400px;
        overflow-y: auto;
        border: 1px solid rgba(25, 118, 210, 0.2);
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    /* 滚动条样式 */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(25, 118, 210, 0.1);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(25, 118, 210, 0.5);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(25, 118, 210, 0.7);
    }
    
    /* 版本号样式 */
    .version {
        position: fixed;
        bottom: 10px;
        right: 10px;
        color: rgba(0, 0, 0, 0.5);
        font-size: 12px;
        z-index: 1000;
    }
    
    /* 登录容器样式 - 修改为白色底色 */
    .login-container {
        max-width: 400px;
        margin: 50px auto;
        background: white;
        padding: 40px;
        border-radius: 15px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.15);
        text-align: center;
        color: #333333;
        z-index: 10;
    }
    
    /* 隐藏Streamlit默认的页脚和菜单 */
    #MainMenu {
        visibility: hidden;
    }
    footer {
        visibility: hidden;
    }
    
    /* 修复白色方块问题 - 移除可能导致问题的样式 */
    .st-emotion-cache-12fmjuu,
    .st-emotion-cache-13ln4jf,
    .st-emotion-cache-1wmy9hl,
    .st-emotion-cache-16txtl3 {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0 !important;
        width: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* 修复容器间距 */
    .stContainer {
        margin: 0;
        padding: 0;
    }
    
    /* 修复Streamlit默认的padding和margin */
    .stApp {
        padding: 0;
        margin: 0;
    }
    
    /* 修复卡片和容器的默认样式 */
    [data-testid="stCard"] {
        background: transparent;
        box-shadow: none;
        border: none;
    }
    
    /* 修复文件上传器样式 */
    [data-testid="stFileUploader"] {
        background: transparent;
    }
    
    /* 修复选择框样式 */
    [data-testid="stSelectbox"] {
        background: transparent;
    }
</style>
""", unsafe_allow_html=True)

# 显示版本号
st.markdown(f'<div class="version">{VERSION}</div>', unsafe_allow_html=True)

# 登录和注册界面
if not st.session_state.logged_in:
    # 创建一个居中的容器
    col1, center_col, col3 = st.columns([1, 1, 1])
    
    with center_col:
        # 显示应用标题
        st.title(f"🏥 {APP_NAME}")
        
        # 根据状态显示登录或注册表单
        if not st.session_state.show_register:
            # 登录表单
            with st.container():
                st.markdown('<div class="login-container">', unsafe_allow_html=True)
                st.subheader("🔐 登录")
                
                username = st.text_input("用户名", key="username_input")
                password = st.text_input("密码", type="password", key="password_input")
                
                # 登录按钮
                if st.button("登录"):
                    if username in st.session_state.users_db and st.session_state.users_db[username] == password:
                        st.session_state.logged_in = True
                        st.session_state.current_user = username
                        st.success("登录成功！")
                        st.rerun()
                    else:
                        st.error("用户名或密码错误")
                
                # 注册链接
                st.markdown("---")
                st.write("还没有账户？")
                if st.button("注册新账户"):
                    st.session_state.show_register = True
                    st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            # 注册表单
            with st.container():
                st.markdown('<div class="login-container">', unsafe_allow_html=True)
                st.subheader("📝 注册")
                
                new_username = st.text_input("新用户名", key="new_username_input")
                new_password = st.text_input("新密码", type="password", key="new_password_input")
                confirm_password = st.text_input("确认密码", type="password", key="confirm_password_input")
                
                # 注册按钮
                if st.button("注册"):
                    if new_username and new_password and confirm_password:
                        if new_username in st.session_state.users_db:
                            st.error("用户名已存在")
                        elif new_password != confirm_password:
                            st.error("两次输入的密码不一致")
                        else:
                            # 添加新用户
                            st.session_state.users_db[new_username] = new_password
                            st.success("注册成功！")
                            st.session_state.show_register = False
                            st.rerun()
                    else:
                        st.error("请填写所有字段")
                
                # 返回登录链接
                st.markdown("---")
                st.write("已有账户？")
                if st.button("返回登录"):
                    st.session_state.show_register = False
                    st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)

# API配置界面
elif not st.session_state.api_key:
    # 创建一个居中的容器
    col1, center_col, col3 = st.columns([1, 1, 1])
    
    with center_col:
        # 显示应用标题
        st.title(f"🏥 {APP_NAME}")
        
        # API配置表单
        with st.container():
            st.markdown('<div class="login-container">', unsafe_allow_html=True)
            st.subheader("⚙️ API配置")
            
            st.session_state.api_key = st.text_input(
                "请输入OpenAI API密钥", 
                type="password", 
                key="api_key_input"
            )
            
            st.session_state.selected_model = st.selectbox(
                "选择大模型", 
                AVAILABLE_MODELS, 
                key="model_select"
            )
            
            if st.button("保存配置"):
                if st.session_state.api_key:
                    # 设置环境变量
                    os.environ["OPENAI_API_KEY"] = st.session_state.api_key
                    os.environ["OPENAI_MODEL"] = st.session_state.selected_model
                    st.success("配置成功！")
                    st.rerun()
                else:
                    st.error("请输入API密钥")
            
            st.markdown('</div>', unsafe_allow_html=True)

# 主应用界面
else:
    # 主页面标题
    st.title(f"🏥 {APP_NAME}")
    
    # 创建左右分栏布局
    left_col, right_col = st.columns([1, 1], gap="medium")
    
    # 左侧对话窗口
    with left_col:
        st.header("💬 对话窗口")
        
        # 文件上传组件
        st.subheader("上传数据")
        uploaded_file = st.file_uploader(
            "选择您的数据文件", 
            type=["csv", "xlsx", "xls"],
            key="file_uploader"
        )
        
        # 如果上传了新文件，更新会话状态
        if uploaded_file is not None and uploaded_file != st.session_state.uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            st.session_state.df = loader.load_data(uploaded_file)
            if st.session_state.df is not None:
                st.session_state.data_types = analyzer.determine_data_types(st.session_state.df)
                st.success("数据上传成功！")
            else:
                st.error("数据加载失败，请检查文件格式")
        
        # 对话历史显示区域
        st.subheader("对话历史")
        chat_container = st.container(height=400)
        
        with chat_container:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            # 显示系统欢迎消息
            if not st.session_state.chat_history:
                st.markdown("**系统**: 您好！请上传数据文件，然后告诉我您的分析需求。")
            
            # 显示对话历史
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"**您**: {msg['content']}")
                else:
                    st.markdown(f"**系统**: {msg['content']}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 用户输入区域
        st.subheader("输入需求")
        user_input = st.text_area(
            "请输入您的分析需求（例如：比较两组患者的年龄差异，分析血糖与血压的相关性等）",
            height=100
        )
        
        # 发送按钮
        if st.button("发送"):
            if user_input:
                # 添加用户消息到对话历史
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                if st.session_state.df is not None:
                    # 生成分析代码
                    code = code_gen.generate_code(st.session_state.df, user_input, st.session_state.data_types)
                    
                    # 运行代码
                    with st.spinner("正在分析..."):
                        max_retries = 3
                        iteration = 0
                        success = False
                        final_error = ""
                        
                        while iteration <= max_retries and not success:
                            try:
                                iteration += 1
                                
                                # 在执行代码前确保matplotlib.pyplot可以正常导入
                                ensure_matplotlib_pyplot()
                                
                                # 自动安装代码中可能使用的缺失模块
                                import re
                                # 提取代码中导入的模块
                                imported_modules = re.findall(r'import\s+(\w+)', code)
                                imported_modules += re.findall(r'from\s+(\w+)\s+import', code)
                                # 去重并过滤掉已导入的模块
                                installed_modules = set()
                                for module in imported_modules:
                                    if module not in ['pandas', 'numpy', 'matplotlib', 'seaborn', 'scipy', 'openai', 'requests']:
                                        installed_modules.add(module)
                                # 安装缺失的模块
                                if installed_modules:
                                    install_missing_modules(list(installed_modules))
                                
                                # 运行代码
                                exec_globals = {"df": st.session_state.df, "pd": pd, "np": np}
                                exec(code, exec_globals)
                                
                                # 获取结果
                                result = exec_globals.get("result", None)
                                
                                # 总结结果
                                summary = result_summarizer.summarize_result(result, user_input)
                                
                                # 保存结果到会话状态
                                st.session_state.analysis_result = {
                                    "code": code,
                                    "result": result,
                                    "summary": summary,
                                    "plt": exec_globals.get("plt", None),
                                    "retries": iteration - 1
                                }
                                
                                # 添加系统回复到对话历史
                                st.session_state.chat_history.append({"role": "assistant", "content": summary})
                                
                                # 关闭图表对象
                                if "plt" in exec_globals:
                                    exec_globals["plt"].close()
                                    
                                success = True
                                
                            except Exception as e:
                                error_msg = str(e)
                                
                                # 如果达到最大重试次数，记录最终错误
                                if iteration > max_retries:
                                    final_error = f"分析失败：经过{max_retries}次修复尝试后仍无法运行代码。\n\n原错误：{error_msg}\n\n最后尝试的代码：\n{code}"
                                    break
                                
                                # 显示修复尝试进度
                                st.info(f"代码执行出错，正在尝试修复 ({iteration}/{max_retries})...")
                                
                                # 使用大模型修复代码
                                code = code_gen.fix_code(
                                    code=code,
                                    error_msg=error_msg,
                                    df=st.session_state.df,
                                    data_types=st.session_state.data_types,
                                    iteration=iteration
                                )
                        
                        # 处理最终结果
                        if not success:
                            st.error(final_error)
                else:
                    st.error("请先上传数据文件")
    
    # 右侧预览窗口
    with right_col:
        st.header("📊 数据与结果预览")
        
        # 数据预览区域
        st.subheader("数据预览")
        if st.session_state.df is not None:
            st.dataframe(st.session_state.df.head(), use_container_width=True)
            
            # 数据特征总结
            st.subheader("数据特征总结")
            features_summary = analyzer.summarize_features(st.session_state.df)
            st.write(features_summary)
            
            # 数据类型判断
            st.subheader("数据类型")
            st.write(st.session_state.data_types)
        else:
            st.info("请先上传数据文件")
        
        # 结果预览区域
        if st.session_state.analysis_result is not None:
            st.subheader("分析结果")
            
            # 使用标签页组织分析结果
            tab1, tab2, tab3 = st.tabs(["分析总结", "生成的代码", "可视化结果"])
            
            with tab1:
                st.markdown(st.session_state.analysis_result["summary"])
                
                # 添加结果下载按钮
                result_text = st.session_state.analysis_result["summary"]
                st.download_button(
                    label="📥 下载分析结果",
                    data=result_text,
                    file_name="analysis_result.txt",
                    mime="text/plain",
                    key="download_result"
                )
            
            with tab2:
                st.code(st.session_state.analysis_result["code"], language='python')
                
                # 添加代码下载按钮
                code_text = st.session_state.analysis_result["code"]
                st.download_button(
                    label="📥 下载代码",
                    data=code_text,
                    file_name="analysis_code.py",
                    mime="text/x-python",
                    key="download_code"
                )
            
            with tab3:
                if st.session_state.analysis_result["plt"] is not None:
                    st.pyplot(st.session_state.analysis_result["plt"])
                else:
                    st.info("本次分析未生成可视化结果")
        else:
            st.info("请输入分析需求")
