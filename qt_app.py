import sys
import os
import pandas as pd
import numpy as np
import sys
import subprocess
import importlib

# 从modules.auto_install导入自动安装函数
from modules.auto_install import install_missing_modules, ensure_matplotlib_pyplot

# 安装必需的模块
required_modules = ["pandas", "numpy", "matplotlib", "seaborn", "scipy", "openai", "requests", "statsmodels", "pingouin", "lifelines"]
install_missing_modules(required_modules)

# 确保matplotlib.pyplot可以正常导入
ensure_matplotlib_pyplot()

# 导入PyQt5模块
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QFileDialog, QTableWidget,
    QTableWidgetItem, QTabWidget, QComboBox, QScrollArea, QGroupBox,
    QMessageBox, QSplitter, QPlainTextEdit, QTabBar, QListWidget
)
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor, QBrush, QLinearGradient
from modules.data_loader import DataLoader
from modules.data_analyzer import DataAnalyzer
from modules.code_generator import CodeGenerator
from modules.result_summarizer import ResultSummarizer
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 应用常量
APP_NAME = '中西循真临床疗效评价智能体'
VERSION = "version 0.01"

# 可用模型列表
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

# 数据分析线程类
class AnalysisThread(QThread):
    result_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    
    def __init__(self, df, user_input, data_types, code_gen, result_summarizer, llm_parser=None):
        super().__init__()
        self.df = df
        self.user_input = user_input
        self.data_types = data_types
        self.code_gen = code_gen
        self.result_summarizer = result_summarizer
        self.llm_parser = llm_parser
    
    def run(self):
        try:
            max_retries = 3  # 最多尝试修复3次
            iteration = 0
            code = None  # 初始化code变量
            error_msg = None  # 初始化error_msg变量
            
            while iteration <= max_retries:
                try:
                    iteration += 1
                    
                    # 生成初始代码或修复后的代码
                    if iteration == 1:
                        # 第一次尝试：生成初始代码
                        code = self.code_gen.generate_code(self.df, self.user_input, self.data_types)
                    else:
                        # 后续尝试：使用大模型重新生成完整代码
                        code = self.code_gen.regenerate_code(
                            self.df, 
                            self.user_input, 
                            self.data_types,
                            previous_code=code,
                            error_msg=error_msg
                        )
                    
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
                    
                    # 在后台线程中设置matplotlib使用非交互式后端
                    import matplotlib
                    original_backend = matplotlib.get_backend()
                    matplotlib.use('Agg')  # 使用非交互式后端
                    
                    # 运行代码
                    exec_globals = {"df": self.df, "pd": pd, "np": np}
                    exec(code, exec_globals)
                    
                    # 获取结果
                    result = exec_globals.get("result", None)
                    
                    # 获取plt对象和figure对象
                    plt = exec_globals.get("plt", None)
                    figure = None
                    if plt is not None:
                        try:
                            figure = plt.gcf()  # 获取当前figure对象
                            # 调整figure大小以确保良好显示
                            figure.set_size_inches(8, 6)
                        except Exception as e:
                            print(f"获取figure对象失败: {e}")
                        finally:
                            # 关闭plt，释放资源
                            plt.close()
                    
                    # 恢复原始后端
                    matplotlib.use(original_backend)
                    
                    # 总结结果
                    summary = self.result_summarizer.summarize_result(result, self.user_input)
                    
                    # 准备结果数据
                    result_data = {
                        "code": code,
                        "result": result,
                        "summary": summary,
                        "figure": figure,  # 传递figure对象而不是plt对象
                        "retries": iteration - 1  # 记录重试次数
                    }
                    
                    self.result_signal.emit(result_data)
                    return  # 成功执行，退出线程
                    
                except Exception as e:
                    error_msg = str(e)
                    
                    # 如果达到最大重试次数，发送最终错误
                    if iteration > max_retries:
                        final_error = f"分析失败：经过{max_retries}次修复尝试后仍无法运行代码。\n\n原错误：{error_msg}\n\n最后尝试的代码：\n{code}"
                        self.error_signal.emit(final_error)
                        return
                    
                    # 记录修复尝试
                    self.error_signal.emit(f"尝试重新生成代码 ({iteration}/{max_retries})...")
        except Exception as e:
            self.error_signal.emit(f"分析错误: {str(e)}")
            return

# 登录对话框
class LoginDialog(QWidget):
    login_success = pyqtSignal(str, str)  # 用户名, 密码
    
    def __init__(self, users_db):
        super().__init__()
        self.users_db = users_db
        self.show_register = False
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle(APP_NAME)
        self.setGeometry(400, 200, 400, 350)
        
        # 创建渐变背景
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, QColor(255, 255, 255))
        gradient.setColorAt(0.5, QColor(227, 242, 253))
        gradient.setColorAt(1.0, QColor(187, 222, 251))
        
        palette = self.palette()
        palette.setBrush(QPalette.Window, QBrush(gradient))
        self.setPalette(palette)
        
        main_layout = QVBoxLayout()
        
        # 应用标题
        title_label = QLabel(f"🏥 {APP_NAME}")
        title_label.setFont(QFont("Arial", 20, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 登录/注册表单容器
        form_container = QWidget()
        form_container.setStyleSheet("""
            QWidget {
                background-color: rgba(255, 255, 255, 0.95);
                border-radius: 10px;
                padding: 20px;
                border: 1px solid rgba(25, 118, 210, 0.2);
            }
        """)
        form_layout = QVBoxLayout(form_container)
        
        # 根据状态显示登录或注册表单
        if not self.show_register:
            self.show_login_form(form_layout)
        else:
            self.show_register_form(form_layout)
        
        main_layout.addWidget(form_container)
        main_layout.setAlignment(Qt.AlignCenter)
        
        self.setLayout(main_layout)
    
    def show_login_form(self, layout):
        # 清空布局
        self.clear_layout(layout)
        
        # 登录表单
        login_label = QLabel("🔐 登录")
        login_label.setFont(QFont("Arial", 14, QFont.Bold))
        login_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(login_label)
        layout.addSpacing(10)
        
        # 用户名输入
        username_label = QLabel("用户名")
        layout.addWidget(username_label)
        self.username_input = QLineEdit()
        self.username_input.setStyleSheet("""
            QLineEdit {
                border: 1px solid rgba(25, 118, 210, 0.3);
                border-radius: 8px;
                padding: 8px;
                background-color: rgba(255, 255, 255, 0.9);
            }
        """)
        layout.addWidget(self.username_input)
        
        # 密码输入
        password_label = QLabel("密码")
        layout.addWidget(password_label)
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setStyleSheet("""
            QLineEdit {
                border: 1px solid rgba(25, 118, 210, 0.3);
                border-radius: 8px;
                padding: 8px;
                background-color: rgba(255, 255, 255, 0.9);
            }
        """)
        layout.addWidget(self.password_input)
        
        layout.addSpacing(15)
        
        # 登录按钮
        login_btn = QPushButton("登录")
        login_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #1976d2, stop:1 #2196f3);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #2196f3, stop:1 #42a5f5);
            }
        """)
        login_btn.clicked.connect(self.login)
        layout.addWidget(login_btn)
        
        layout.addSpacing(10)
        
        # 注册链接
        register_btn = QPushButton("注册新账户")
        register_btn.setStyleSheet("""
            QPushButton {
                background: none;
                border: none;
                color: #1976d2;
                text-decoration: underline;
            }
        """)
        register_btn.clicked.connect(self.show_register_page)
        layout.addWidget(register_btn)
    
    def show_register_form(self, layout):
        # 清空布局
        self.clear_layout(layout)
        
        # 注册表单
        register_label = QLabel("📝 注册")
        register_label.setFont(QFont("Arial", 14, QFont.Bold))
        register_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(register_label)
        layout.addSpacing(10)
        
        # 新用户名输入
        new_username_label = QLabel("新用户名")
        layout.addWidget(new_username_label)
        self.new_username_input = QLineEdit()
        self.new_username_input.setStyleSheet("""
            QLineEdit {
                border: 1px solid rgba(25, 118, 210, 0.3);
                border-radius: 8px;
                padding: 8px;
                background-color: rgba(255, 255, 255, 0.9);
            }
        """)
        layout.addWidget(self.new_username_input)
        
        # 新密码输入
        new_password_label = QLabel("新密码")
        layout.addWidget(new_password_label)
        self.new_password_input = QLineEdit()
        self.new_password_input.setEchoMode(QLineEdit.Password)
        self.new_password_input.setStyleSheet("""
            QLineEdit {
                border: 1px solid rgba(25, 118, 210, 0.3);
                border-radius: 8px;
                padding: 8px;
                background-color: rgba(255, 255, 255, 0.9);
            }
        """)
        layout.addWidget(self.new_password_input)
        
        # 确认密码输入
        confirm_password_label = QLabel("确认密码")
        layout.addWidget(confirm_password_label)
        self.confirm_password_input = QLineEdit()
        self.confirm_password_input.setEchoMode(QLineEdit.Password)
        self.confirm_password_input.setStyleSheet("""
            QLineEdit {
                border: 1px solid rgba(25, 118, 210, 0.3);
                border-radius: 8px;
                padding: 8px;
                background-color: rgba(255, 255, 255, 0.9);
            }
        """)
        layout.addWidget(self.confirm_password_input)
        
        layout.addSpacing(15)
        
        # 注册按钮
        register_btn = QPushButton("注册")
        register_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #1976d2, stop:1 #2196f3);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #2196f3, stop:1 #42a5f5);
            }
        """)
        register_btn.clicked.connect(self.register)
        layout.addWidget(register_btn)
        
        layout.addSpacing(10)
        
        # 返回登录链接
        back_btn = QPushButton("返回登录")
        back_btn.setStyleSheet("""
            QPushButton {
                background: none;
                border: none;
                color: #1976d2;
                text-decoration: underline;
            }
        """)
        back_btn.clicked.connect(self.show_login_page)
        layout.addWidget(back_btn)
    
    def login(self):
        username = self.username_input.text()
        password = self.password_input.text()
        
        if username in self.users_db and self.users_db[username] == password:
            self.login_success.emit(username, password)
            self.close()
        else:
            QMessageBox.warning(self, "登录失败", "用户名或密码错误")
    
    def register(self):
        new_username = self.new_username_input.text()
        new_password = self.new_password_input.text()
        confirm_password = self.confirm_password_input.text()
        
        if new_username and new_password and confirm_password:
            if new_username in self.users_db:
                QMessageBox.warning(self, "注册失败", "用户名已存在")
            elif new_password != confirm_password:
                QMessageBox.warning(self, "注册失败", "两次输入的密码不一致")
            else:
                # 添加新用户
                self.users_db[new_username] = new_password
                QMessageBox.information(self, "注册成功", "注册成功！")
                self.show_login_page()
        else:
            QMessageBox.warning(self, "注册失败", "请填写所有字段")
    
    def show_register_page(self):
        self.show_register = True
        self.init_ui()
    
    def show_login_page(self):
        self.show_register = False
        self.init_ui()
    
    def clear_layout(self, layout):
        while layout.count() > 0:
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
    
    def fill_missing_values(self):
        """处理缺失值"""
        try:
            column = self.fill_missing_column_combo.currentText()
            method = self.fill_missing_method_combo.currentText()
            
            if method == "均值填充":
                self.df[column] = self.df[column].fillna(self.df[column].mean())
            elif method == "中位数填充":
                self.df[column] = self.df[column].fillna(self.df[column].median())
            elif method == "众数填充":
                self.df[column] = self.df[column].fillna(self.df[column].mode()[0])
            elif method == "线性插值":
                self.df[column] = self.df[column].interpolate()
            elif method == "删除缺失值":
                self.df = self.df.dropna(subset=[column])
            
            # 更新数据预览
            self.update_data_preview()
            
            QMessageBox.information(self, "成功", f"已使用{method}方法处理{column}列的缺失值")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理缺失值时发生错误: {str(e)}")
    
    def transform_data(self):
        """数据转换"""
        try:
            column = self.transform_column_combo.currentText()
            method = self.transformation_method.currentText()
            
            # 确保列是数值类型
            self.df[column] = pd.to_numeric(self.df[column], errors='coerce')
            
            if method == "对数转换":
                # 确保所有值为正
                if (self.df[column] <= 0).any():
                    QMessageBox.warning(self, "警告", "对数转换要求所有值为正")
                    return
                self.df[column] = np.log(self.df[column])
            elif method == "平方根转换":
                # 确保所有值非负
                if (self.df[column] < 0).any():
                    QMessageBox.warning(self, "警告", "平方根转换要求所有值非负")
                    return
                self.df[column] = np.sqrt(self.df[column])
            elif method == "平方转换":
                self.df[column] = self.df[column] ** 2
            elif method == "指数转换":
                self.df[column] = np.exp(self.df[column])
            
            # 更新数据预览
            self.update_data_preview()
            
            QMessageBox.information(self, "成功", f"已使用{method}方法转换{column}列")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"数据转换时发生错误: {str(e)}")
    
    def standardize_data(self):
        """数据标准化"""
        try:
            column = self.standardize_column_combo.currentText()
            method = self.normalization_method.currentText()
            
            # 确保列是数值类型
            self.df[column] = pd.to_numeric(self.df[column], errors='coerce')
            
            if method == "Z-score标准化":
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                self.df[column] = scaler.fit_transform(self.df[[column]])
            elif method == "Min-Max标准化":
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                self.df[column] = scaler.fit_transform(self.df[[column]])
            elif method == "Robust标准化":
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
                self.df[column] = scaler.fit_transform(self.df[[column]])
            
            # 更新数据预览
            self.update_data_preview()
            
            QMessageBox.information(self, "成功", f"已使用{method}方法标准化{column}列")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"数据标准化时发生错误: {str(e)}")
    
    def run_regression(self):
        """运行回归分析"""
        try:
            method = self.regression_method_combo.currentText() if hasattr(self, 'regression_method_combo') else self.regression_method.currentText()
            dep_var = self.regression_y.currentText()
            
            # 获取选中的自变量
            if hasattr(self, 'regression_indep_vars_combo'):
                indep_var = self.regression_indep_vars_combo.currentText()
            else:
                # 从列表中获取选中的自变量
                selected_items = self.regression_x.selectedItems()
                if not selected_items:
                    QMessageBox.warning(self, "警告", "请选择至少一个自变量")
                    return
                indep_vars = [item.text() for item in selected_items]
                indep_var = indep_vars[0]  # 暂时只支持一个自变量
            
            # 确保因变量是数值类型
            self.df[dep_var] = pd.to_numeric(self.df[dep_var], errors='coerce')
            # 确保自变量是数值类型
            self.df[indep_var] = pd.to_numeric(self.df[indep_var], errors='coerce')
            
            # 移除包含NaN的行
            df_clean = self.df[[dep_var, indep_var]].dropna()
            
            import statsmodels.api as sm
            
            # 添加常数项
            X = sm.add_constant(df_clean[indep_var])
            y = df_clean[dep_var]
            
            if method == "线性回归":
                model = sm.OLS(y, X).fit()
            elif method == "Logistic回归":
                # 确保因变量是二分类变量
                if len(df_clean[dep_var].unique()) != 2:
                    QMessageBox.warning(self, "警告", "Logistic回归要求因变量是二分类变量")
                    return
                model = sm.Logit(y, X).fit()
            elif method == "Cox回归":
                # Cox回归需要生存分析包
                try:
                    from lifelines import CoxPHFitter
                    # 确保数据包含生存时间和事件指示器
                    # 这里假设indep_var是生存时间，dep_var是事件指示器
                    if not (y.isin([0, 1]).all()):
                        QMessageBox.warning(self, "警告", "Cox回归要求因变量是事件指示器(0/1)")
                        return
                    cph = CoxPHFitter()
                    cph.fit(df_clean[[indep_var, dep_var]], duration_col=indep_var, event_col=dep_var)
                    result_text = str(cph.summary)
                    
                    # 显示结果
                    self.show_analysis_result(f"{method}结果", result_text)
                    return
                except ImportError:
                    QMessageBox.critical(self, "错误", "Cox回归需要lifelines包，请先安装")
                    return
            
            # 显示结果
            result_text = f"{method}结果:\n\n"
            result_text += str(model.summary())
            
            self.show_analysis_result(f"{method}结果", result_text)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"运行回归分析时发生错误: {str(e)}")
    
    def calculate_propensity_score(self):
        """计算倾向性评分"""
        try:
            if hasattr(self, 'propensity_treatment_combo'):
                treatment_var = self.propensity_treatment_combo.currentText()
                covariate_var = self.propensity_covariates_combo.currentText()
            else:
                treatment_var = self.treatment_var.currentText()
                # 从列表中获取选中的协变量
                selected_items = self.covariates_list.selectedItems() if hasattr(self, 'covariates_list') else self.regression_x.selectedItems()
                if not selected_items:
                    QMessageBox.warning(self, "警告", "请选择至少一个协变量")
                    return
                covariate_vars = [item.text() for item in selected_items]
                covariate_var = covariate_vars
            
            # 确保处理变量是二分类变量
            if len(self.df[treatment_var].unique()) != 2:
                QMessageBox.warning(self, "警告", "倾向性评分要求处理变量是二分类变量")
                return
            
            # 确保协变量是数值类型
            for var in covariate_var:
                self.df[var] = pd.to_numeric(self.df[var], errors='coerce')
            
            # 移除包含NaN的行
            df_clean = self.df[[treatment_var] + covariate_var].dropna()
            
            from sklearn.linear_model import LogisticRegression
            
            # 准备数据
            X = df_clean[covariate_var]
            y = df_clean[treatment_var]
            
            # 拟合Logistic回归模型
            model = LogisticRegression()
            model.fit(X, y)
            
            # 计算倾向性评分
            propensity_scores = model.predict_proba(X)[:, 1]
            
            # 显示结果
            result_text = "倾向性评分结果:\n\n"
            result_text += f"处理变量: {treatment_var}\n"
            result_text += f"协变量: {', '.join(covariate_var)}\n"
            result_text += f"模型系数: {model.coef_}\n"
            result_text += f"截距: {model.intercept_}\n"
            result_text += f"倾向性评分示例: {propensity_scores[:5]}\n"
            
            self.show_analysis_result("倾向性评分结果", result_text)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"计算倾向性评分时发生错误: {str(e)}")
    
    def mixed_effects_model(self, data, outcome, treatment, time, subject_id, 
                           random_slope=False, covariance='unstructured', 
                           max_categories=5, min_unique=10, covariates=None):
        """
        拟合混合效应模型
        
        参数:
            data: pandas DataFrame - 包含所有变量的数据集
            outcome: str - 因变量列名
            treatment: str - 处理变量列名
            time: str - 时间变量列名
            subject_id: str - 受试者ID列名
            random_slope: bool - 是否包含随机斜率
            covariance: str - 协方差结构类型
            max_categories: int - 多分类变量的最大类别数
            min_unique: int - 连续变量的最小唯一值数
            covariates: list - 协变量列表
        
        返回:
            result: 拟合结果
        """
        # 检查必需列
        required_columns = [outcome, treatment, time, subject_id]
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            raise ValueError(f"数据中缺少必需列: {missing}")
    
        # 创建数据副本并删除缺失值
        df = data.copy().dropna(subset=required_columns)
    
        # 确定协变量
        key_columns = [outcome, treatment, time, subject_id]
        if covariates is None:
            # 默认使用除关键变量外的所有列作为协变量
            covariates = [col for col in df.columns if col not in key_columns]
        else:
            # 确保用户指定的协变量存在
            missing_cov = [cov for cov in covariates if cov not in df.columns]
            if missing_cov:
                raise ValueError(f"指定的协变量不存在: {missing_cov}")
    
        # 构建公式
        fixed_effects = f"{treatment} * {time}"
    
        # 添加协变量
        if covariates:
            cov_string = " + ".join(covariates)
            formula = f"{outcome} ~ {fixed_effects} + {cov_string}"
        else:
            formula = f"{outcome} ~ {fixed_effects}"
    
        # 构建随机效应结构
        if random_slope:
            re_formula = f"1 + {time}"
        else:
            re_formula = "1"
    
        # 自动检测因变量类型并选择模型
        outcome_series = df[outcome]
        n_unique = outcome_series.nunique()
    
        # 1. 检查是否为二分类变量
        if n_unique == 2:
            # 确认是二元分类 (0/1 或 True/False)
            unique_vals = sorted(outcome_series.dropna().unique())
            if set(unique_vals) in [{0, 1}, {0.0, 1.0}, {False, True}]:
                return self._fit_binary_glmm(df, formula, subject_id, re_formula)
    
        # 2. 检查是否为多分类变量
        if 2 < n_unique <= max_categories:
            return self._fit_multinomial_glmm(df, outcome, treatment, time, subject_id, re_formula, covariance, covariates)
    
        # 3. 检查是否为计数数据
        if outcome_series.min() >= 0 and outcome_series.dtype in [np.int64, np.int32]:
            # 检查过离散 (方差 > 均值)
            variance = outcome_series.var()
            mean_val = outcome_series.mean()
    
            if variance > 1.5 * mean_val:
                return self._fit_count_glmm(df, formula, subject_id, 'negativebinomial', re_formula)
            else:
                return self._fit_count_glmm(df, formula, subject_id, 'poisson', re_formula)
    
        # 4. 检查是否为连续非负偏态数据
        if outcome_series.min() >= 0 and n_unique > min_unique:
            # 检查偏度
            skewness = stats.skew(outcome_series.dropna())
            if skewness > 1.0:  # 显著右偏
                return self._fit_glmm(df, formula, subject_id, 'gamma', 'log', re_formula)
    
        # 5. 默认使用线性混合模型
        return self._fit_linear_mixed_model(df, formula, subject_id, re_formula, covariance)
    
    def _fit_linear_mixed_model(self, df, formula, subject_id, re_formula, covariance):
        """拟合线性混合模型"""
        import statsmodels.formula.api as smf
        
        model = smf.mixedlm(
            formula=formula,
            data=df,
            groups=df[subject_id],
            re_formula=re_formula
        )
    
        # 设置协方差结构
        if covariance != 'unstructured':
            model.set_covariance_type(covariance)
    
        result = model.fit()
        result.model_type = "Linear Mixed Model (Gaussian)"
        return result
    
    def _fit_binary_glmm(self, df, formula, subject_id, re_formula):
        """拟合二分类Logistic混合模型"""
        try:
            # 使用statsmodels的BinomialBayesMixedGLM
            from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
            import patsy
    
            # 准备设计矩阵
            y, X = patsy.dmatrices(formula, df, return_type='dataframe')
    
            # 创建随机效应设计矩阵
            groups = df[subject_id]
            vc = {}
            intercept_dm = patsy.dmatrix(f"0 + C({subject_id})", df)
            vc['intercept'] = np.asarray(intercept_dm)
            n_intercepts = vc['intercept'].shape[1]
    
            # 创建标识符数组
            n_subjects = df[subject_id].nunique()
            ident = np.zeros(len(vc['intercept'].columns))  # 初始化为全0
    
            n_slopes = 0
            if re_formula != "1":
                # 提取时间变量名
                time_var = re_formula.split(" + ")[1]
                slope_dm = patsy.dmatrix(f"0 + C({subject_id}):{time_var}", df)
                vc['slope'] = np.asarray(slope_dm)
                n_slopes = vc['slope'].shape[1]
    
            #ident = np.zeros(n_intercepts)  # 随机截距标识为0
            if n_slopes > 0:
                ident = np.concatenate([ident, np.ones(n_slopes)])  # 随机斜率标识为1
    
            # 将设计矩阵转换为数组
            exog = np.asarray(X)
            endog = np.asarray(y).ravel()
            vc_matrix = np.hstack([v for v in vc.values()])
    
            # 拟合模型
            model = BinomialBayesMixedGLM(endog, exog, vc_matrix, ident)
            result = model.fit_vb()
            result.model_type = "Bayesian Binomial Mixed Model (Logit)"
            return result
        except Exception as e:
            # 如果获取figure对象失败，显示错误信息
            import warnings
            warnings.warn(f"无法使用BinomialBayesMixedGLM: {str(e)}，使用GEE替代")
            return self._fit_glmm(df, formula, subject_id, 'binomial', 'logit', re_formula)
    
    def _fit_count_glmm(self, df, formula, subject_id, family, re_formula):
        """拟合计数数据混合模型（泊松/负二项）"""
        if family == 'poisson':
            return self._fit_glmm(df, formula, subject_id, 'poisson', 'log', re_formula)
        elif family == 'negativebinomial':
            return self._fit_glmm(df, formula, subject_id, 'negativebinomial', 'log', re_formula)
        else:
            raise ValueError(f"不支持的计数分布族: {family}")
    
    def _fit_glmm(self, df, formula, subject_id, family, link, re_formula):
        """拟合广义线性混合模型（使用GEE）"""
        import statsmodels.api as sm
        from statsmodels.formula.api import gee
    
        family_map = {
            'binomial': sm.families.Binomial,
            'poisson': sm.families.Poisson,
            'gamma': sm.families.Gamma,
            'negativebinomial': sm.families.NegativeBinomial,
            'gaussian': sm.families.Gaussian
        }
    
        link_map = {
            'logit': sm.families.links.logit,
            'probit': sm.families.links.probit,
            'log': sm.families.links.log,
            'identity': sm.families.links.identity,
            'cloglog': sm.families.links.cloglog,
            'inverse': sm.families.links.inverse_power
        }
    
        if family.lower() not in family_map:
            raise ValueError(f"不支持的分布族: {family}")
    
        if link.lower() not in link_map:
            raise ValueError(f"不支持的连接函数: {link}")
    
        family_instance = family_map[family.lower()](link=link_map[link.lower()]())
    
        # 使用GEE作为替代方案
        cov_struct = sm.cov_struct.Exchangeable()
    
        model = gee(
            formula=formula,
            groups=df[subject_id],
            cov_struct=cov_struct,
            data=df,
            family=family_instance
        )
    
        result = model.fit()
        result.model_type = f"GEE Model ({family.capitalize()}, {link.capitalize()})"
        return result
    
    def _fit_multinomial_glmm(self, df, outcome, treatment, time, subject_id, re_formula, covariance, covariates):
        """拟合多项Logistic回归模型（固定效应，无随机效应）"""
        import warnings
        import statsmodels.api as sm
        warnings.warn("多项Logistic混合模型在Python中实现有限，使用固定效应多项Logistic回归（忽略随机效应）")
        
        # 确保 outcome 为整数编码（如 0, 1, 2）
        y = df[outcome].astype(int)  # 显式转换为整数类型
        
        # 构建预测变量（固定效应）
        predictors = [treatment, time] + covariates
        X = df[predictors]
        X = pd.get_dummies(X, drop_first=True)  # 自动处理类别变量
        
        # 添加截距项
        X = sm.add_constant(X)
        
        # 使用 statsmodels 的 MNLogit（直接传递 X 和 y）
        model = sm.MNLogit(endog=y, exog=X)  # 注意参数顺序：endog=y, exog=X
        result = model.fit()
        result.model_type = "Multinomial Logistic Regression (No Random Effects)"
        return result
    
    def repeated_measures_anova(self, data, outcome_var, subject_var, group_var, time_var):
        """
        执行混合设计重复测量方差分析（Mixed Design ANOVA）
        
        参数:
            data: pandas DataFrame - 包含所有变量的数据集
            outcome_var: str - 因变量（连续型变量）的列名
            subject_var: str - 被试ID列名
            group_var: str - 组间因素列名（分类变量）
            time_var: str - 组内因素列名（分类变量）
        
        返回:
            result: 包含ANOVA结果和可视化图表的字典
        """
        # 数据检查：确保列存在
        required_cols = [outcome_var, subject_var, group_var, time_var]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"数据中缺少必要列: {missing_cols}")
    
        # 转换为分类变量（确保组间和组内因素是分类变量）
        data[group_var] = data[group_var].astype('category')
        data[time_var] = data[time_var].astype('category')
    
        # 执行混合设计重复测量方差分析
        try:
            import pingouin as pg
            
            anova_results = pg.mixed_anova(
                data=data,
                dv=outcome_var,
                between=group_var,
                within=time_var,
                subject=subject_var
            )
    
            posthoc_time = pg.pairwise_ttests(
                data=data,
                dv=outcome_var,
                within=time_var,
                subject=subject_var,
                parametric=True,
                padjust='bonf'
            )
    
            posthoc_group = pg.pairwise_ttests(
                data=data,
                dv=outcome_var,
                between=group_var,
                parametric=True,
                padjust='bonf'
            )
            
            data['group_time'] = data[group_var].astype(str) + '_' + data[time_var].astype(str)
            posthoc_interaction = pg.pairwise_ttests(
                data=data,
                dv=outcome_var,
                between='group_time',
                parametric=True,
                padjust='bonf'
            )
    
            # 创建可视化图表
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(10, 6))
            sns.pointplot(
                data=data,
                x=time_var,
                y=outcome_var,
                hue=group_var,
                dodge=True,
                errorbar=('ci', 95),
                linestyles=['-', '--'],
                markers=['o', 's']
            )
            plt.title('混合设计重复测量方差分析结果')
            plt.ylabel(outcome_var)
            plt.legend(title=group_var)
            plt.tight_layout()
            plot = plt.gcf()
    
            return {
                'anova_table': anova_results,
                'posthoc_time': posthoc_time,
                'posthoc_group': posthoc_group,
                'posthoc_interaction': posthoc_interaction,
                'plot': plot
            }
        except Exception as e:
            raise ValueError(f"执行重复测量方差分析时出错: {e}")

# API配置对话框
class APIConfigDialog(QWidget):
    config_success = pyqtSignal(str, str)  # API密钥, 模型
    
    def __init__(self):
        super().__init__()
        self.api_status = "off"  # off, testing, on, error
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle(APP_NAME)
        self.setGeometry(400, 200, 400, 350)
        
        # 创建渐变背景
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, QColor(255, 255, 255))
        gradient.setColorAt(0.5, QColor(227, 242, 253))
        gradient.setColorAt(1.0, QColor(187, 222, 251))
        
        palette = self.palette()
        palette.setBrush(QPalette.Window, QBrush(gradient))
        self.setPalette(palette)
        
        main_layout = QVBoxLayout()
        
        # 应用标题
        title_label = QLabel(f"🏥 {APP_NAME}")
        title_label.setFont(QFont("Arial", 20, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # API配置表单容器
        config_container = QWidget()
        config_container.setStyleSheet("""
            QWidget {
                background-color: rgba(255, 255, 255, 0.95);
                border-radius: 10px;
                padding: 20px;
                border: 1px solid rgba(25, 118, 210, 0.2);
            }
        """)
        config_layout = QVBoxLayout(config_container)
        
        # API配置标题
        config_label = QLabel("⚙️ API配置")
        config_label.setFont(QFont("Arial", 14, QFont.Bold))
        config_label.setAlignment(Qt.AlignCenter)
        config_layout.addWidget(config_label)
        config_layout.addSpacing(10)
        
        # API密钥输入
        api_key_label = QLabel("OpenAI API密钥")
        config_layout.addWidget(api_key_label)
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setStyleSheet("""
            QLineEdit {
                border: 1px solid rgba(25, 118, 210, 0.3);
                border-radius: 8px;
                padding: 8px;
                background-color: rgba(255, 255, 255, 0.9);
            }
        """)
        config_layout.addWidget(self.api_key_input)
        
        config_layout.addSpacing(10)
        
        # 模型选择
        model_label = QLabel("选择大模型")
        config_layout.addWidget(model_label)
        self.model_select = QComboBox()
        self.model_select.addItems(AVAILABLE_MODELS)
        # 设置默认选中qwen3
        self.model_select.setCurrentIndex(3)
        self.model_select.setStyleSheet("""
            QComboBox {
                border: 1px solid rgba(25, 118, 210, 0.3);
                border-radius: 8px;
                padding: 8px;
                background-color: rgba(255, 255, 255, 0.9);
            }
        """)
        config_layout.addWidget(self.model_select)
        
        config_layout.addSpacing(15)
        
        # 添加状态灯
        status_layout = QHBoxLayout()
        status_layout.setAlignment(Qt.AlignCenter)
        
        self.status_light = QLabel()
        self.status_light.setFixedSize(20, 20)
        self.status_light.setStyleSheet("""
            QLabel {
                border-radius: 10px;
                background-color: #ff4444;
            }
        """)
        status_layout.addWidget(self.status_light)
        
        self.status_text = QLabel("API连接状态: 未连接")
        self.status_text.setFont(QFont("Arial", 10))
        status_layout.addWidget(self.status_text)
        
        config_layout.addLayout(status_layout)
        
        config_layout.addSpacing(15)
        
        # 保存按钮
        save_btn = QPushButton("保存配置")
        save_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #1976d2, stop:1 #2196f3);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #2196f3, stop:1 #42a5f5);
            }
        """)
        save_btn.clicked.connect(self.save_config)
        config_layout.addWidget(save_btn)
        
        main_layout.addWidget(config_container)
        main_layout.setAlignment(Qt.AlignCenter)
        
        self.setLayout(main_layout)
        
        # 连接信号
        self.api_key_input.textChanged.connect(self.on_input_change)
        self.model_select.currentIndexChanged.connect(self.on_input_change)
    
    def on_input_change(self):
        """当用户输入API密钥或选择模型时自动测试连接"""
        api_key = self.api_key_input.text()
        if api_key:
            self.test_api_connection()
        else:
            self.update_status("off")
    
    def update_status(self, status):
        """更新状态灯和状态文本"""
        self.api_status = status
        
        if status == "off":
            self.status_light.setStyleSheet("""
                QLabel {
                    border-radius: 10px;
                    background-color: #ff4444;
                }
            """)
            self.status_text.setText("API连接状态: 未连接")
        elif status == "testing":
            self.status_light.setStyleSheet("""
                QLabel {
                    border-radius: 10px;
                    background-color: #ffaa00;
                }
            """)
            self.status_text.setText("API连接状态: 测试中...")
        elif status == "on":
            self.status_light.setStyleSheet("""
                QLabel {
                    border-radius: 10px;
                    background-color: #44ff44;
                }
            """)
            self.status_text.setText("API连接状态: 正常")
        elif status == "error":
            self.status_light.setStyleSheet("""
                QLabel {
                    border-radius: 10px;
                    background-color: #ff4444;
                }
            """)
            self.status_text.setText("API连接状态: 错误")
    
    def test_api_connection(self):
        """测试API连接"""
        self.update_status("testing")
        
        # 使用线程测试API连接，避免UI卡顿
        from PyQt5.QtCore import QThread, pyqtSignal
        
        class TestThread(QThread):
            result_signal = pyqtSignal(bool)
            
            def __init__(self, api_key, model):
                super().__init__()
                self.api_key = api_key
                self.model = model
            
            def run(self):
                try:
                    from modules.llm_parser import LLMParser
                    # 使用LLMParser进行连接测试，支持不同模型类型
                    llm_parser = LLMParser(api_key=self.api_key, model=self.model)
                    success, message = llm_parser.test_connection()
                    self.result_signal.emit(success)
                except Exception as e:
                    print(f"API连接测试失败: {e}")
                    self.result_signal.emit(False)
        
        api_key = self.api_key_input.text()
        model = self.model_select.currentText()
        
        self.test_thread = TestThread(api_key, model)
        self.test_thread.result_signal.connect(self.on_test_result)
        self.test_thread.start()
    
    def on_test_result(self, success):
        """处理API连接测试结果"""
        if success:
            self.update_status("on")
        else:
            self.update_status("error")
    
    def save_config(self):
        api_key = self.api_key_input.text()
        selected_model = self.model_select.currentText()
        
        if api_key:
            self.config_success.emit(api_key, selected_model)
            self.close()
        else:
            QMessageBox.warning(self, "配置失败", "请输入API密钥")

# 主应用窗口
class MainApplication(QMainWindow):
    def __init__(self, username):
        super().__init__()
        self.username = username
        self.df = None
        self.original_df = None  # 保存原始数据副本
        self.data_types = None
        self.chat_history = []
        self.analysis_result = None
        self.init_modules()
        self.init_ui()
    
    def init_modules(self):
        # 初始化模块
        self.loader = DataLoader()
        self.analyzer = DataAnalyzer()
        self.code_gen = CodeGenerator()
        self.result_summarizer = ResultSummarizer()
    
    def init_ui(self):
        self.setWindowTitle(f"🏥 {APP_NAME}")
        self.setGeometry(100, 50, 1200, 800)
        
        # 创建中央组件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建渐变背景
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, QColor(255, 255, 255))
        gradient.setColorAt(0.5, QColor(227, 242, 253))
        gradient.setColorAt(1.0, QColor(187, 222, 251))
        
        palette = central_widget.palette()
        palette.setBrush(QPalette.Window, QBrush(gradient))
        central_widget.setPalette(palette)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 标题
        title_label = QLabel(f"🏥 {APP_NAME}")
        title_label.setFont(QFont("Arial", 24, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #1976d2;")
        main_layout.addWidget(title_label)
        main_layout.addSpacing(20)
        
        # 创建三栏布局
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧对话窗口
        left_widget = self.create_left_panel()
        
        # 中间功能模块
        middle_widget = self.create_middle_panel()
        
        # 右侧数据预览
        right_widget = self.create_right_panel()
        
        splitter.addWidget(left_widget)
        splitter.addWidget(middle_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 500, 300])  # 设置初始大小
        
        main_layout.addWidget(splitter)
        
        # 版本号
        version_label = QLabel(VERSION)
        version_label.setAlignment(Qt.AlignRight)
        version_label.setStyleSheet("color: rgba(0, 0, 0, 0.5); font-size: 12px;")
        main_layout.addWidget(version_label)
    
    def create_left_panel(self):
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # 对话窗口标题
        chat_title = QLabel("💬 对话窗口")
        chat_title.setFont(QFont("Arial", 16, QFont.Bold))
        chat_title.setStyleSheet("color: #1976d2;")
        left_layout.addWidget(chat_title)
        
        left_layout.addSpacing(10)
        
        # 文件上传区域
        upload_group = QGroupBox("上传数据")
        upload_group.setStyleSheet("""
            QGroupBox {
                background-color: rgba(255, 255, 255, 0.9);
                border-radius: 8px;
                border: 1px solid rgba(25, 118, 210, 0.2);
                padding: 10px;
            }
            QGroupBox::title {
                color: #1976d2;
                font-weight: bold;
            }
        """)
        upload_layout = QVBoxLayout(upload_group)
        
        upload_btn = QPushButton("选择数据文件")
        upload_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #1976d2, stop:1 #2196f3);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #2196f3, stop:1 #42a5f5);
            }
        """)
        upload_btn.clicked.connect(self.upload_file)
        upload_layout.addWidget(upload_btn)
        
        # 上传状态
        self.upload_status = QLabel("未上传数据文件")
        self.upload_status.setStyleSheet("color: #666;")
        upload_layout.addWidget(self.upload_status)
        
        left_layout.addWidget(upload_group)
        
        left_layout.addSpacing(10)
        
        # 对话历史区域
        chat_group = QGroupBox("对话历史")
        chat_group.setStyleSheet("""
            QGroupBox {
                background-color: rgba(255, 255, 255, 0.9);
                border-radius: 8px;
                border: 1px solid rgba(25, 118, 210, 0.2);
                padding: 10px;
            }
            QGroupBox::title {
                color: #1976d2;
                font-weight: bold;
            }
        """)
        chat_layout = QVBoxLayout(chat_group)
        
        # 对话历史显示
        self.chat_history_text = QTextEdit()
        self.chat_history_text.setReadOnly(True)
        self.chat_history_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid rgba(25, 118, 210, 0.2);
                border-radius: 8px;
                padding: 10px;
                background-color: rgba(255, 255, 255, 0.9);
            }
        """)
        self.chat_history_text.setFixedHeight(300)
        chat_layout.addWidget(self.chat_history_text)
        
        # 显示系统欢迎消息
        self.chat_history_text.append("**系统**: 您好！请上传数据文件，然后告诉我您的分析需求。")
        
        left_layout.addWidget(chat_group)
        
        left_layout.addSpacing(10)
        
        # 用户输入区域
        input_group = QGroupBox("输入需求")
        input_group.setStyleSheet("""
            QGroupBox {
                background-color: rgba(255, 255, 255, 0.9);
                border-radius: 8px;
                border: 1px solid rgba(25, 118, 210, 0.2);
                padding: 10px;
            }
            QGroupBox::title {
                color: #1976d2;
                font-weight: bold;
            }
        """)
        input_layout = QVBoxLayout(input_group)
        
        self.user_input = QTextEdit()
        self.user_input.setPlaceholderText("请输入您的分析需求（例如：比较两组患者的年龄差异，分析血糖与血压的相关性等）")
        self.user_input.setStyleSheet("""
            QTextEdit {
                border: 1px solid rgba(25, 118, 210, 0.2);
                border-radius: 8px;
                padding: 10px;
                background-color: rgba(255, 255, 255, 0.9);
            }
        """)
        self.user_input.setFixedHeight(100)
        input_layout.addWidget(self.user_input)
        
        send_btn = QPushButton("发送")
        send_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #1976d2, stop:1 #2196f3);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #2196f3, stop:1 #42a5f5);
            }
        """)
        send_btn.clicked.connect(self.send_request)
        input_layout.addWidget(send_btn)
        
        left_layout.addWidget(input_group)
        
        left_layout.addStretch()
        
        return left_widget
    
    def create_middle_panel(self):
        """
        创建中间功能模块面板
        """
        middle_widget = QWidget()
        middle_layout = QVBoxLayout(middle_widget)
        
        # 功能模块标题
        module_title = QLabel("⚙️ 功能模块")
        module_title.setFont(QFont("Arial", 16, QFont.Bold))
        module_title.setStyleSheet("color: #1976d2;")
        middle_layout.addWidget(module_title)
        
        middle_layout.addSpacing(10)
        
        # 选项卡窗口
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                background-color: rgba(255, 255, 255, 0.9);
                border-radius: 8px;
                border: 1px solid rgba(25, 118, 210, 0.2);
                padding: 10px;
            }
            QTabBar::tab {
                background-color: rgba(255, 255, 255, 0.9);
                border: 1px solid rgba(25, 118, 210, 0.2);
                border-radius: 8px 8px 0 0;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: rgba(25, 118, 210, 0.1);
                color: #1976d2;
                font-weight: bold;
            }
        """)
        
        # 数据处理选项卡
        self.data_processing_tab = QWidget()
        self.data_processing_layout = QVBoxLayout(self.data_processing_tab)
        self.tab_widget.addTab(self.data_processing_tab, "数据处理")
        self.init_data_processing_tab()
        
        # 数据分析选项卡
        self.data_analysis_tab = QWidget()
        self.data_analysis_layout = QVBoxLayout(self.data_analysis_tab)
        self.tab_widget.addTab(self.data_analysis_tab, "数据分析")
        self.init_data_analysis_tab()
        
        # 倾向性评分分析选项卡
        self.propensity_score_tab = QWidget()
        self.propensity_score_layout = QVBoxLayout(self.propensity_score_tab)
        self.tab_widget.addTab(self.propensity_score_tab, "倾向性评分分析")
        self.init_propensity_score_tab()
        
        # 分析结果选项卡
        self.result_tab = QWidget()
        self.result_layout = QVBoxLayout(self.result_tab)
        self.tab_widget.addTab(self.result_tab, "分析结果")
        
        middle_layout.addWidget(self.tab_widget)
        
        return middle_widget
    
    def create_right_panel(self):
        """
        创建右侧数据预览面板
        """
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # 数据预览标题
        preview_title = QLabel("📊 数据预览")
        preview_title.setFont(QFont("Arial", 16, QFont.Bold))
        preview_title.setStyleSheet("color: #1976d2;")
        right_layout.addWidget(preview_title)
        
        right_layout.addSpacing(10)
        
        # 数据预览选项卡
        self.data_preview_tab = QWidget()
        self.data_preview_layout = QVBoxLayout(self.data_preview_tab)
        
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.data_preview_tab)
        
        right_layout.addWidget(scroll_area)
        
        return right_widget
    
    def init_data_processing_tab(self):
        """
        初始化数据处理选项卡
        """
        # 创建滚动区域的容器
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # 数据处理标题
        processing_title = QLabel("🔧 数据处理功能")
        processing_title.setFont(QFont("Arial", 14, QFont.Bold))
        processing_title.setStyleSheet("color: #1976d2;")
        scroll_layout.addWidget(processing_title)
        
        scroll_layout.addSpacing(10)
        
        # 缺失值填补组
        missing_values_group = QGroupBox("缺失值填补")
        missing_values_layout = QVBoxLayout(missing_values_group)
        
        # 方法选择
        method_label = QLabel("选择填补方法:")
        missing_values_layout.addWidget(method_label)
        
        self.fill_missing_method_combo = QComboBox()
        self.fill_missing_method_combo.addItems(["均值填补", "中位数填补", "众数填补", "线性插值", "删除缺失值"])
        self.fill_missing_method_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid rgba(25, 118, 210, 0.3);
                border-radius: 8px;
                padding: 8px;
                background-color: rgba(255, 255, 255, 0.9);
            }
        """)
        missing_values_layout.addWidget(self.fill_missing_method_combo)
        
        # 列选择
        column_label = QLabel("选择要填补的列:")
        missing_values_layout.addWidget(column_label)
        
        self.fill_missing_column_combo = QComboBox()
        self.fill_missing_column_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid rgba(25, 118, 210, 0.3);
                border-radius: 8px;
                padding: 8px;
                background-color: rgba(255, 255, 255, 0.9);
            }
        """)
        missing_values_layout.addWidget(self.fill_missing_column_combo)
        
        # 填补按钮
        fill_missing_btn = QPushButton("执行缺失值填补")
        fill_missing_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #1976d2, stop:1 #1e88e5);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #2196f3, stop:1 #42a5f5);
            }
        """)
        fill_missing_btn.clicked.connect(self.fill_missing_values)
        missing_values_layout.addWidget(fill_missing_btn)
        
        scroll_layout.addWidget(missing_values_group)
        
        scroll_layout.addSpacing(10)
        
        # 数据转换组
        data_transformation_group = QGroupBox("数据转换")
        data_transformation_layout = QVBoxLayout(data_transformation_group)
        
        # 方法选择
        transform_method_label = QLabel("选择转换方法:")
        data_transformation_layout.addWidget(transform_method_label)
        
        self.transformation_method = QComboBox()
        self.transformation_method.addItems(["对数转换", "平方根转换", "平方转换", "指数转换"])
        self.transformation_method.setStyleSheet("""
            QComboBox {
                border: 1px solid rgba(25, 118, 210, 0.3);
                border-radius: 8px;
                padding: 8px;
                background-color: rgba(255, 255, 255, 0.9);
            }
        """)
        data_transformation_layout.addWidget(self.transformation_method)
        
        # 列选择
        transform_column_label = QLabel("选择要转换的列:")
        data_transformation_layout.addWidget(transform_column_label)
        
        self.transform_column_combo = QComboBox()
        self.transform_column_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid rgba(25, 118, 210, 0.3);
                border-radius: 8px;
                padding: 8px;
                background-color: rgba(255, 255, 255, 0.9);
            }
        """)
        data_transformation_layout.addWidget(self.transform_column_combo)
        
        # 转换按钮
        transform_btn = QPushButton("执行数据转换")
        transform_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #1976d2, stop:1 #1e88e5);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #2196f3, stop:1 #42a5f5);
            }
        """)
        transform_btn.clicked.connect(self.transform_data)
        data_transformation_layout.addWidget(transform_btn)
        
        scroll_layout.addWidget(data_transformation_group)
        
        scroll_layout.addSpacing(10)
        
        # 数据标准化组
        normalization_group = QGroupBox("数据标准化")
        normalization_layout = QVBoxLayout(normalization_group)
        
        # 方法选择
        normalization_method_label = QLabel("选择标准化方法:")
        normalization_layout.addWidget(normalization_method_label)
        
        self.normalization_method = QComboBox()
        self.normalization_method.addItems(["Z-score标准化", "Min-Max标准化", "Robust标准化"])
        self.normalization_method.setStyleSheet("""
            QComboBox {
                border: 1px solid rgba(25, 118, 210, 0.3);
                border-radius: 8px;
                padding: 8px;
                background-color: rgba(255, 255, 255, 0.9);
            }
        """)
        normalization_layout.addWidget(self.normalization_method)
        
        # 列选择
        normalization_column_label = QLabel("选择要标准化的列:")
        normalization_layout.addWidget(normalization_column_label)
        
        self.standardize_column_combo = QComboBox()
        self.standardize_column_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid rgba(25, 118, 210, 0.3);
                border-radius: 8px;
                padding: 8px;
                background-color: rgba(255, 255, 255, 0.9);
            }
        """)
        normalization_layout.addWidget(self.standardize_column_combo)
        
        # 标准化按钮
        normalize_btn = QPushButton("执行数据标准化")
        normalize_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #1976d2, stop:1 #1e88e5);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #2196f3, stop:1 #42a5f5);
            }
        """)
        normalize_btn.clicked.connect(self.standardize_data)
        normalization_layout.addWidget(normalize_btn)
        
        scroll_layout.addWidget(normalization_group)
        
        # 重置数据按钮
        reset_data_btn = QPushButton("重置数据")
        reset_data_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #ef5350, stop:1 #f44336);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #e57373, stop:1 #ef5350);
            }
        """)
        reset_data_btn.clicked.connect(self.reset_data)
        scroll_layout.addWidget(reset_data_btn)
        
        scroll_layout.addSpacing(10)
        
        # 下载处理后数据按钮
        download_data_btn = QPushButton("下载处理后数据")
        download_data_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #66bb6a, stop:1 #43a047);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #81c784, stop:1 #66bb6a);
            }
        """)
        download_data_btn.clicked.connect(self.download_processed_data)
        scroll_layout.addWidget(download_data_btn)
        
        # 确认处理完成并加载到分析模块按钮
        confirm_processing_btn = QPushButton("确认处理完成并加载到分析")
        confirm_processing_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #2196f3, stop:1 #1976d2);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #64b5f6, stop:1 #42a5f5);
            }
        """)
        confirm_processing_btn.clicked.connect(self.confirm_processing_completed)
        scroll_layout.addWidget(confirm_processing_btn)
        
        scroll_layout.addStretch()
        
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_widget)
        
        # 将滚动区域添加到数据处理布局
        self.data_processing_layout.addWidget(scroll_area)
    
    def init_data_analysis_tab(self):
        """
        初始化数据分析选项卡
        """
        # 创建滚动区域的容器
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # 数据分析标题
        analysis_title = QLabel("📈 数据分析功能")
        analysis_title.setFont(QFont("Arial", 14, QFont.Bold))
        analysis_title.setStyleSheet("color: #1976d2;")
        scroll_layout.addWidget(analysis_title)
        
        scroll_layout.addSpacing(10)
        
        # 回归分析组
        regression_group = QGroupBox("回归分析")
        regression_layout = QVBoxLayout(regression_group)
        
        # 方法选择
        regression_method_label = QLabel("选择回归方法:")
        regression_layout.addWidget(regression_method_label)
        
        self.regression_method = QComboBox()
        self.regression_method.addItems(["线性回归", "Logistic回归", "Cox回归"])
        self.regression_method.setStyleSheet("""
            QComboBox {
                border: 1px solid rgba(25, 118, 210, 0.3);
                border-radius: 8px;
                padding: 8px;
                background-color: rgba(255, 255, 255, 0.9);
            }
        """)
        regression_layout.addWidget(self.regression_method)
        
        # 自变量选择（支持多选）
        x_label = QLabel("选择自变量 (X):")
        regression_layout.addWidget(x_label)
        
        self.regression_x = QListWidget()
        self.regression_x.setSelectionMode(QListWidget.MultiSelection)
        self.regression_x.setStyleSheet("""
            QListWidget {
                border: 1px solid rgba(25, 118, 210, 0.3);
                border-radius: 8px;
                padding: 8px;
                background-color: rgba(255, 255, 255, 0.9);
            }
            QListWidget::item:selected {
                background-color: rgba(25, 118, 210, 0.2);
                color: #1976d2;
            }
        """)
        regression_layout.addWidget(self.regression_x)
        
        # 因变量选择
        y_label = QLabel("选择因变量 (Y):")
        regression_layout.addWidget(y_label)
        
        self.regression_y = QComboBox()
        self.regression_y.setStyleSheet("""
            QComboBox {
                border: 1px solid rgba(25, 118, 210, 0.3);
                border-radius: 8px;
                padding: 8px;
                background-color: rgba(255, 255, 255, 0.9);
            }
        """)
        regression_layout.addWidget(self.regression_y)
        
        # 分析按钮
        regression_btn = QPushButton("执行回归分析")
        regression_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #1976d2, stop:1 #1e88e5);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #2196f3, stop:1 #42a5f5);
            }
        """)
        regression_btn.clicked.connect(self.perform_regression_analysis)
        regression_layout.addWidget(regression_btn)
        
        scroll_layout.addWidget(regression_group)
        
        scroll_layout.addSpacing(10)
        
        # 时序性分析组
        temporal_group = QGroupBox("时序性分析")
        temporal_layout = QVBoxLayout(temporal_group)
        
        # 结局类型选择
        outcome_type_label = QLabel("选择结局类型:")
        temporal_layout.addWidget(outcome_type_label)
        
        self.outcome_type_combo = QComboBox()
        self.outcome_type_combo.addItems(["连续性结局", "分类结局"])
        self.outcome_type_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid rgba(25, 118, 210, 0.3);
                border-radius: 8px;
                padding: 8px;
                background-color: rgba(255, 255, 255, 0.9);
            }
        """)
        self.outcome_type_combo.currentIndexChanged.connect(self.on_outcome_type_changed)
        temporal_layout.addWidget(self.outcome_type_combo)
        
        # 分析方法选择
        method_label = QLabel("选择分析方法:")
        temporal_layout.addWidget(method_label)
        
        self.temporal_method_combo = QComboBox()
        # 初始显示连续性结局的分析方法
        self.temporal_method_combo.addItems(["重复测量方差分析", "GEE", "协方差分析"])
        self.temporal_method_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid rgba(25, 118, 210, 0.3);
                border-radius: 8px;
                padding: 8px;
                background-color: rgba(255, 255, 255, 0.9);
            }
        """)
        temporal_layout.addWidget(self.temporal_method_combo)
        
        # 因变量选择
        y_label = QLabel("选择因变量 (Y):")
        temporal_layout.addWidget(y_label)
        
        self.temporal_y = QComboBox()
        self.temporal_y.setStyleSheet("""
            QComboBox {
                border: 1px solid rgba(25, 118, 210, 0.3);
                border-radius: 8px;
                padding: 8px;
                background-color: rgba(255, 255, 255, 0.9);
            }
        """)
        temporal_layout.addWidget(self.temporal_y)
        
        # 时间变量选择
        time_label = QLabel("选择时间变量:")
        temporal_layout.addWidget(time_label)
        
        self.time_var = QComboBox()
        self.time_var.setStyleSheet("""
            QComboBox {
                border: 1px solid rgba(25, 118, 210, 0.3);
                border-radius: 8px;
                padding: 8px;
                background-color: rgba(255, 255, 255, 0.9);
            }
        """)
        temporal_layout.addWidget(self.time_var)
        
        # 受试者ID选择
        subject_label = QLabel("选择受试者ID:")
        temporal_layout.addWidget(subject_label)
        
        self.subject_id_var = QComboBox()
        self.subject_id_var.setStyleSheet("""
            QComboBox {
                border: 1px solid rgba(25, 118, 210, 0.3);
                border-radius: 8px;
                padding: 8px;
                background-color: rgba(255, 255, 255, 0.9);
            }
        """)
        temporal_layout.addWidget(self.subject_id_var)
        
        # 分组变量选择
        group_label = QLabel("选择分组变量 (可选):")
        temporal_layout.addWidget(group_label)
        
        self.group_var = QComboBox()
        self.group_var.setStyleSheet("""
            QComboBox {
                border: 1px solid rgba(25, 118, 210, 0.3);
                border-radius: 8px;
                padding: 8px;
                background-color: rgba(255, 255, 255, 0.9);
            }
        """)
        temporal_layout.addWidget(self.group_var)
        
        # 协变量选择（支持多选）
        covariates_label = QLabel("选择协变量 (可选):")
        temporal_layout.addWidget(covariates_label)
        
        self.temporal_covariates_var = QListWidget()
        self.temporal_covariates_var.setSelectionMode(QListWidget.MultiSelection)
        self.temporal_covariates_var.setStyleSheet("""
            QListWidget {
                border: 1px solid rgba(25, 118, 210, 0.3);
                border-radius: 8px;
                padding: 8px;
                background-color: rgba(255, 255, 255, 0.9);
            }
            QListWidget::item:selected {
                background-color: rgba(25, 118, 210, 0.2);
                color: #1976d2;
            }
        """)
        temporal_layout.addWidget(self.temporal_covariates_var)
        
        # 分析按钮
        temporal_btn = QPushButton("执行时序性分析")
        temporal_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #1976d2, stop:1 #1e88e5);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #2196f3, stop:1 #42a5f5);
            }
        """)
        temporal_btn.clicked.connect(self.perform_temporal_analysis)
        temporal_layout.addWidget(temporal_btn)
        
        scroll_layout.addWidget(temporal_group)
        
        scroll_layout.addStretch()
        
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_widget)
        
        # 将滚动区域添加到数据分析布局
        self.data_analysis_layout.addWidget(scroll_area)
    
    def init_propensity_score_tab(self):
        """
        初始化倾向性评分分析选项卡
        """
        # 创建滚动区域的容器
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # 倾向性评分标题
        ps_title = QLabel("🎯 倾向性评分分析")
        ps_title.setFont(QFont("Arial", 14, QFont.Bold))
        ps_title.setStyleSheet("color: #1976d2;")
        scroll_layout.addWidget(ps_title)
        
        scroll_layout.addSpacing(10)
        
        # 1. PS计算参数组
        calculation_group = QGroupBox("1. PS计算参数")
        calculation_layout = QVBoxLayout(calculation_group)
        
        # 处理组选择
        treatment_label = QLabel("选择处理组变量 (必须是二分类):")
        calculation_layout.addWidget(treatment_label)
        
        self.treatment_var = QComboBox()
        self.treatment_var.setStyleSheet("""
            QComboBox {
                border: 1px solid rgba(25, 118, 210, 0.3);
                border-radius: 8px;
                padding: 8px;
                background-color: rgba(255, 255, 255, 0.9);
            }
        """)
        calculation_layout.addWidget(self.treatment_var)
        
        # 协变量选择（支持多选）
        covariates_label = QLabel("选择协变量 (用于计算倾向性评分):")
        calculation_layout.addWidget(covariates_label)
        
        self.covariates_var = QListWidget()
        self.covariates_var.setSelectionMode(QListWidget.MultiSelection)
        self.covariates_var.setStyleSheet("""
            QListWidget {
                border: 1px solid rgba(25, 118, 210, 0.3);
                border-radius: 8px;
                padding: 8px;
                background-color: rgba(255, 255, 255, 0.9);
                max-height: 100px;
            }
            QListWidget::item:selected {
                background-color: rgba(25, 118, 210, 0.2);
                color: #1976d2;
            }
        """)
        calculation_layout.addWidget(self.covariates_var)
        
        # 计算PS按钮
        calculate_ps_btn = QPushButton("计算倾向性评分")
        calculate_ps_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #1976d2, stop:1 #1e88e5);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #2196f3, stop:1 #42a5f5);
            }
        """)
        calculate_ps_btn.clicked.connect(self.calculate_propensity_score)
        calculation_layout.addWidget(calculate_ps_btn)
        
        scroll_layout.addWidget(calculation_group)
        
        scroll_layout.addSpacing(10)
        
        # 2. SMD评估结果组
        smd_group = QGroupBox("2. 标准化均数差(SMD)评估")
        smd_layout = QVBoxLayout(smd_group)
        
        # SMD结果展示选项
        smd_display_label = QLabel("选择SMD展示方式:")
        smd_layout.addWidget(smd_display_label)
        
        self.smd_display_combo = QComboBox()
        self.smd_display_combo.addItems(["柱状图", "森林图"])
        self.smd_display_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid rgba(25, 118, 210, 0.3);
                border-radius: 8px;
                padding: 8px;
                background-color: rgba(255, 255, 255, 0.9);
            }
        """)
        smd_layout.addWidget(self.smd_display_combo)
        
        # 展示SMD按钮
        show_smd_btn = QPushButton("展示SMD结果")
        show_smd_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #1976d2, stop:1 #1e88e5);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #2196f3, stop:1 #42a5f5);
            }
        """)
        show_smd_btn.clicked.connect(self.show_smd_results)
        smd_layout.addWidget(show_smd_btn)
        
        scroll_layout.addWidget(smd_group)
        
        scroll_layout.addSpacing(10)
        
        # 3. PS应用方法组
        application_group = QGroupBox("3. PS应用方法")
        application_layout = QVBoxLayout(application_group)
        
        # PS分层
        stratification_label = QLabel("PS分层:")
        application_layout.addWidget(stratification_label)
        
        self.strata_num_combo = QComboBox()
        self.strata_num_combo.addItems(["3层", "4层", "5层"])
        self.strata_num_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid rgba(25, 118, 210, 0.3);
                border-radius: 8px;
                padding: 8px;
                background-color: rgba(255, 255, 255, 0.9);
            }
        """)
        application_layout.addWidget(self.strata_num_combo)
        
        stratify_btn = QPushButton("执行PS分层")
        stratify_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #1976d2, stop:1 #1e88e5);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #2196f3, stop:1 #42a5f5);
            }
        """)
        stratify_btn.clicked.connect(self.perform_ps_stratification)
        application_layout.addWidget(stratify_btn)
        
        application_layout.addSpacing(10)
        
        # PS匹配
        matching_label = QLabel("PS匹配:")
        application_layout.addWidget(matching_label)
        
        self.matching_method_combo = QComboBox()
        self.matching_method_combo.addItems(["最近邻匹配", "半径匹配", "核匹配"])
        self.matching_method_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid rgba(25, 118, 210, 0.3);
                border-radius: 8px;
                padding: 8px;
                background-color: rgba(255, 255, 255, 0.9);
            }
        """)
        application_layout.addWidget(self.matching_method_combo)
        
        match_btn = QPushButton("执行PS匹配")
        match_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #1976d2, stop:1 #1e88e5);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #2196f3, stop:1 #42a5f5);
            }
        """)
        match_btn.clicked.connect(self.perform_ps_matching)
        application_layout.addWidget(match_btn)
        
        application_layout.addSpacing(10)
        
        # 将PS加入协变量调整
        ps_in_covariates_btn = QPushButton("将PS加入协变量调整")
        ps_in_covariates_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #1976d2, stop:1 #1e88e5);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #2196f3, stop:1 #42a5f5);
            }
        """)
        ps_in_covariates_btn.clicked.connect(self.add_ps_to_covariates)
        application_layout.addWidget(ps_in_covariates_btn)
        
        scroll_layout.addWidget(application_group)
        
        scroll_layout.addSpacing(10)
        
        # 4. 数据加载组
        loading_group = QGroupBox("4. 数据加载")
        loading_layout = QVBoxLayout(loading_group)
        
        # 加载到数据分析模块
        load_to_analysis_btn = QPushButton("加载处理后数据到数据分析模块")
        load_to_analysis_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #2196f3, stop:1 #1976d2);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #64b5f6, stop:1 #42a5f5);
            }
        """)
        load_to_analysis_btn.clicked.connect(self.confirm_processing_completed)
        loading_layout.addWidget(load_to_analysis_btn)
        
        scroll_layout.addWidget(loading_group)
        
        scroll_layout.addStretch()
        
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_widget)
        
        # 将滚动区域添加到倾向性评分布局
        self.propensity_score_layout.addWidget(scroll_area)
    
    def upload_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择数据文件", "", "数据文件 (*.csv *.xlsx *.xls)"
        )
        
        if file_path:
            try:
                self.df = self.loader.load_data(file_path)
                if self.df is not None:
                    self.original_df = self.df.copy()  # 保存原始数据副本
                    self.data_types = self.analyzer.determine_data_types(self.df)
                    self.upload_status.setText(f"已上传: {os.path.basename(file_path)}")
                    self.upload_status.setStyleSheet("color: #2e7d32;")
                    self.update_data_preview()
                    self.update_column_comboboxes()  # 更新所有组合框
                else:
                    self.upload_status.setText("数据加载失败")
                    self.upload_status.setStyleSheet("color: #d32f2f;")
            except Exception as e:
                self.upload_status.setText(f"数据加载错误: {str(e)}")
                self.upload_status.setStyleSheet("color: #d32f2f;")
                QMessageBox.critical(self, "错误", f"数据加载时发生错误: {str(e)}")
    
    def update_data_preview(self):
        # 清空当前布局
        self.clear_layout(self.data_preview_layout)
        
        # 数据预览标题
        preview_title = QLabel("📊 数据预览")
        preview_title.setFont(QFont("Arial", 14, QFont.Bold))
        preview_title.setStyleSheet("color: #1976d2;")
        self.data_preview_layout.addWidget(preview_title)
        
        self.data_preview_layout.addSpacing(10)
        
        # 创建垂直分割器，允许用户调整各部分大小
        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: rgba(25, 118, 210, 0.1);
                height: 5px;
            }
            QSplitter::handle:hover {
                background-color: rgba(25, 118, 210, 0.3);
            }
        """)
        
        # ========== 上方：治理前数据 ==========
        original_widget = QWidget()
        original_layout = QVBoxLayout(original_widget)
        
        original_group = QGroupBox("🔄 治理前数据")
        original_group_layout = QVBoxLayout(original_group)
        
        # 治理前数据预览表格
        original_table = QTableWidget()
        if hasattr(self, 'original_df'):
            # 显示更多行，最多1000行，让用户可以滚动查看
            original_table.setRowCount(min(1000, len(self.original_df)))
            original_table.setColumnCount(len(self.original_df.columns))
            original_table.setHorizontalHeaderLabels(self.original_df.columns)
            
            for row in range(min(1000, len(self.original_df))):
                for col in range(len(self.original_df.columns)):
                    item = QTableWidgetItem(str(self.original_df.iloc[row, col]))
                    original_table.setItem(row, col, item)
        
        original_table.resizeColumnsToContents()
        original_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid rgba(25, 118, 210, 0.2);
                border-radius: 8px;
                background-color: rgba(255, 255, 255, 0.9);
            }
            QHeaderView::section {
                background-color: rgba(25, 118, 210, 0.1);
                color: #1976d2;
                font-weight: bold;
            }
        """)
        # 移除固定高度，让表格可以自适应大小
        original_group_layout.addWidget(original_table)
        original_layout.addWidget(original_group)
        
        main_splitter.addWidget(original_widget)
        
        # ========== 中间：治理后数据 ==========
        processed_widget = QWidget()
        processed_layout = QVBoxLayout(processed_widget)
        
        processed_group = QGroupBox("✅ 治理后数据")
        processed_group_layout = QVBoxLayout(processed_group)
        
        # 治理后数据预览表格
        processed_table = QTableWidget()
        # 显示更多行，最多1000行，让用户可以滚动查看
        processed_table.setRowCount(min(1000, len(self.df)))
        processed_table.setColumnCount(len(self.df.columns))
        processed_table.setHorizontalHeaderLabels(self.df.columns)
        
        for row in range(min(1000, len(self.df))):
            for col in range(len(self.df.columns)):
                item = QTableWidgetItem(str(self.df.iloc[row, col]))
                processed_table.setItem(row, col, item)
        
        processed_table.resizeColumnsToContents()
        processed_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid rgba(46, 125, 50, 0.2);
                border-radius: 8px;
                background-color: rgba(255, 255, 255, 0.9);
            }
            QHeaderView::section {
                background-color: rgba(46, 125, 50, 0.1);
                color: #2e7d32;
                font-weight: bold;
            }
        """)
        # 移除固定高度，让表格可以自适应大小
        processed_group_layout.addWidget(processed_table)
        processed_layout.addWidget(processed_group)
        
        main_splitter.addWidget(processed_widget)
        
        # ========== 下方：数据特征分页 ==========
        features_widget = QWidget()
        features_layout = QVBoxLayout(features_widget)
        
        features_label = QLabel("📋 数据特征")
        features_label.setFont(QFont("Arial", 12, QFont.Bold))
        features_layout.addWidget(features_label)
        
        # 数据特征分页标签
        features_tab = QTabWidget()
        features_tab.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid rgba(25, 118, 210, 0.2);
                border-radius: 8px;
                background-color: rgba(255, 255, 255, 0.9);
                padding: 10px;
            }
            QTabBar::tab {
                background-color: rgba(255, 255, 255, 0.9);
                border: 1px solid rgba(25, 118, 210, 0.2);
                border-radius: 8px 8px 0 0;
                padding: 6px 12px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: rgba(25, 118, 210, 0.1);
                color: #1976d2;
                font-weight: bold;
            }
        """)
        
        # 治理前数据特征
        original_features_widget = QWidget()
        original_features_layout = QVBoxLayout(original_features_widget)
        
        if hasattr(self, 'original_df'):
            original_features_summary = self.analyzer.summarize_features(self.original_df)
            original_features_text = QTextEdit()
            original_features_text.setPlainText(str(original_features_summary))
            original_features_text.setReadOnly(True)
            original_features_text.setStyleSheet("""
                QTextEdit {
                    border: none;
                    background-color: transparent;
                    font-size: 10px;
                }
            """)
            # 移除固定高度，让特征信息可以自适应大小
            original_features_layout.addWidget(original_features_text)
        
        features_tab.addTab(original_features_widget, "治理前特征")
        
        # 治理后数据特征
        processed_features_widget = QWidget()
        processed_features_layout = QVBoxLayout(processed_features_widget)
        
        processed_features_summary = self.analyzer.summarize_features(self.df)
        processed_features_text = QTextEdit()
        processed_features_text.setPlainText(str(processed_features_summary))
        processed_features_text.setReadOnly(True)
        processed_features_text.setStyleSheet("""
            QTextEdit {
                border: none;
                background-color: transparent;
                font-size: 10px;
            }
        """)
        # 移除固定高度，让特征信息可以自适应大小
        processed_features_layout.addWidget(processed_features_text)
        
        features_tab.addTab(processed_features_widget, "治理后特征")
        
        features_layout.addWidget(features_tab)
        
        main_splitter.addWidget(features_widget)
        
        # 设置初始大小比例
        main_splitter.setSizes([200, 200, 150])
        
        # 添加分割器到布局
        self.data_preview_layout.addWidget(main_splitter)
        
        # 添加实时更新提示
        update_label = QLabel("🔄 数据已实时更新")
        update_label.setFont(QFont("Arial", 9))
        update_label.setStyleSheet("color: #2e7d32;")
        update_label.setAlignment(Qt.AlignRight)
        self.data_preview_layout.addWidget(update_label)
        
        # 更新数据处理和分析选项卡中的列下拉框
        self.update_column_comboboxes()
    
    def update_column_comboboxes(self):
        """更新数据处理和分析选项卡中的列下拉框"""
        # 数据处理选项卡
        # 缺失值填补
        self.fill_missing_column_combo.clear()
        self.fill_missing_column_combo.addItems(self.df.columns.tolist())
        
        # 数据转换
        self.transform_column_combo.clear()
        self.transform_column_combo.addItems(self.df.columns.tolist())
        
        # 数据标准化
        self.standardize_column_combo.clear()
        self.standardize_column_combo.addItems(self.df.columns.tolist())
        
        # 数据分析选项卡
        # 回归分析
        self.regression_y.clear()
        self.regression_y.addItems(self.df.columns.tolist())
        
        self.regression_x.clear()
        for column in self.df.columns.tolist():
            self.regression_x.addItem(column)
        
        # 倾向性评分
        self.treatment_var.clear()
        self.treatment_var.addItems(self.df.columns.tolist())
        
        self.covariates_var.clear()
        for column in self.df.columns.tolist():
            self.covariates_var.addItem(column)
        
        # 时序性分析
        if hasattr(self, 'temporal_y'):
            # 因变量选择
            self.temporal_y.clear()
            self.temporal_y.addItems(self.df.columns.tolist())
            
            # 时间变量选择
            self.time_var.clear()
            self.time_var.addItems(self.df.columns.tolist())
            
            # 受试者ID选择
            self.subject_id_var.clear()
            self.subject_id_var.addItems(self.df.columns.tolist())
            
            # 分组变量选择
            self.group_var.clear()
            self.group_var.addItem("")  # 添加空选项
            self.group_var.addItems(self.df.columns.tolist())
            
            # 协变量选择
            self.temporal_covariates_var.clear()
            for column in self.df.columns.tolist():
                self.temporal_covariates_var.addItem(column)
    
    def fill_missing_values(self):
        """处理缺失值"""
        try:
            column = self.fill_missing_column_combo.currentText()
            method = self.fill_missing_method_combo.currentText()
            
            if method == "均值填充":
                self.df[column] = self.df[column].fillna(self.df[column].mean())
            elif method == "中位数填充":
                self.df[column] = self.df[column].fillna(self.df[column].median())
            elif method == "众数填充":
                self.df[column] = self.df[column].fillna(self.df[column].mode()[0])
            elif method == "插值填充":
                self.df[column] = self.df[column].interpolate()
            elif method == "删除缺失值":
                self.df = self.df.dropna(subset=[column])
            
            # 更新数据预览
            self.update_data_preview()
            
            QMessageBox.information(self, "成功", f"已使用{method}方法处理{column}列的缺失值")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理缺失值时发生错误: {str(e)}")
    
    def transform_data(self):
        """数据转换"""
        try:
            column = self.transform_column_combo.currentText()
            method = self.transformation_method.currentText()
            
            # 确保列是数值类型
            self.df[column] = pd.to_numeric(self.df[column], errors='coerce')
            
            if method == "对数转换":
                # 确保所有值为正
                if (self.df[column] <= 0).any():
                    QMessageBox.warning(self, "警告", "对数转换要求所有值为正")
                    return
                self.df[column] = np.log(self.df[column])
            elif method == "平方根转换":
                # 确保所有值非负
                if (self.df[column] < 0).any():
                    QMessageBox.warning(self, "警告", "平方根转换要求所有值非负")
                    return
                self.df[column] = np.sqrt(self.df[column])
            elif method == "平方转换":
                self.df[column] = self.df[column] ** 2
            elif method == "指数转换":
                self.df[column] = np.exp(self.df[column])
            
            # 更新数据预览
            self.update_data_preview()
            
            QMessageBox.information(self, "成功", f"已使用{method}方法转换{column}列")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"数据转换时发生错误: {str(e)}")
    
    def standardize_data(self):
        """数据标准化"""
        try:
            column = self.standardize_column_combo.currentText()
            method = self.normalization_method.currentText()
            
            # 确保列是数值类型
            self.df[column] = pd.to_numeric(self.df[column], errors='coerce')
            
            if method == "Z-score标准化":
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                self.df[column] = scaler.fit_transform(self.df[[column]])
            elif method == "Min-Max标准化":
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                self.df[column] = scaler.fit_transform(self.df[[column]])
            elif method == "Robust标准化":
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
                self.df[column] = scaler.fit_transform(self.df[[column]])
            
            # 更新数据预览
            self.update_data_preview()
            
            QMessageBox.information(self, "成功", f"已使用{method}方法标准化{column}列")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"数据标准化时发生错误: {str(e)}")
    
    def run_regression(self):
        """运行回归分析"""
        try:
            method = self.regression_method_combo.currentText()
            dep_var = self.regression_dep_var_combo.currentText()
            indep_var = self.regression_indep_vars_combo.currentText()
            
            # 确保因变量是数值类型
            self.df[dep_var] = pd.to_numeric(self.df[dep_var], errors='coerce')
            # 确保自变量是数值类型
            self.df[indep_var] = pd.to_numeric(self.df[indep_var], errors='coerce')
            
            # 移除包含NaN的行
            df_clean = self.df[[dep_var, indep_var]].dropna()
            
            import statsmodels.api as sm
            
            # 添加常数项
            X = sm.add_constant(df_clean[indep_var])
            y = df_clean[dep_var]
            
            if method == "线性回归":
                model = sm.OLS(y, X).fit()
            elif method == "Logistic回归":
                # 确保因变量是二分类变量
                if len(df_clean[dep_var].unique()) != 2:
                    QMessageBox.warning(self, "警告", "Logistic回归要求因变量是二分类变量")
                    return
                model = sm.Logit(y, X).fit()
            elif method == "Cox回归":
                # Cox回归需要生存分析包
                try:
                    from lifelines import CoxPHFitter
                    # 确保数据包含生存时间和事件指示器
                    # 这里假设indep_var是生存时间，dep_var是事件指示器
                    if not (y.isin([0, 1]).all()):
                        QMessageBox.warning(self, "警告", "Cox回归要求因变量是事件指示器(0/1)")
                        return
                    cph = CoxPHFitter()
                    cph.fit(df_clean[[indep_var, dep_var]], duration_col=indep_var, event_col=dep_var)
                    result_text = str(cph.summary)
                    
                    # 显示结果
                    self.show_analysis_result(f"{method}结果", result_text)
                    return
                except ImportError:
                    QMessageBox.critical(self, "错误", "Cox回归需要lifelines包，请先安装")
                    return
            
            # 显示结果
            result_text = f"{method}结果:\n\n"
            result_text += str(model.summary())
            
            self.show_analysis_result(f"{method}结果", result_text)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"运行回归分析时发生错误: {str(e)}")
    
    def calculate_propensity_score(self):
        """计算倾向性评分"""
        try:
            treatment_var = self.propensity_treatment_combo.currentText()
            covariate_var = self.propensity_covariates_combo.currentText()
            
            # 确保处理变量是二分类变量
            if len(self.df[treatment_var].unique()) != 2:
                QMessageBox.warning(self, "警告", "倾向性评分要求处理变量是二分类变量")
                return
            
            # 确保协变量是数值类型
            self.df[covariate_var] = pd.to_numeric(self.df[covariate_var], errors='coerce')
            
            # 移除包含NaN的行
            df_clean = self.df[[treatment_var, covariate_var]].dropna()
            
            from sklearn.linear_model import LogisticRegression
            
            # 准备数据
            X = df_clean[[covariate_var]]
            y = df_clean[treatment_var]
            
            # 拟合Logistic回归模型
            model = LogisticRegression()
            model.fit(X, y)
            
            # 计算倾向性评分
            df_clean['propensity_score'] = model.predict_proba(X)[:, 1]
            
            # 将评分添加回原始数据框
            self.df['propensity_score'] = df_clean['propensity_score']
            
            # 显示结果
            result_text = f"倾向性评分计算结果:\n\n"
            result_text += f"处理变量: {treatment_var}\n"
            result_text += f"协变量: {covariate_var}\n\n"
            result_text += "模型系数:\n"
            result_text += f"截距: {model.intercept_[0]:.4f}\n"
            result_text += f"协变量系数: {model.coef_[0][0]:.4f}\n\n"
            result_text += "前10个样本的倾向性评分:\n"
            result_text += str(df_clean[['propensity_score']].head(10))
            
            self.show_analysis_result("倾向性评分结果", result_text)
            
            # 更新数据预览
            self.update_data_preview()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"计算倾向性评分时发生错误: {str(e)}")
    
    def show_analysis_result(self, *args):
        """显示分析结果
        
        支持两种调用方式：
        1. show_analysis_result(title, result_text)
        2. show_analysis_result({"title": "标题", "result": "结果"})
        """
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton
        
        # 处理不同的调用方式
        if len(args) == 1 and isinstance(args[0], dict):
            # 字典参数调用方式
            title = args[0].get("title", "分析结果")
            result_text = args[0].get("result", "")
        else:
            # 位置参数调用方式
            if len(args) < 2:
                title = "分析结果"
                result_text = args[0] if args else ""
            else:
                title, result_text = args
        
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(800, 600)
        
        layout = QVBoxLayout(dialog)
        
        text_edit = QTextEdit()
        text_edit.setPlainText(result_text)
        text_edit.setReadOnly(True)
        layout.addWidget(text_edit)
        
        close_button = QPushButton("关闭")
        close_button.clicked.connect(dialog.close)
        layout.addWidget(close_button, 0, Qt.AlignRight)
        
        dialog.exec_()
    
    def send_request(self):
        user_input = self.user_input.toPlainText()
        
        if user_input:
            # 添加用户消息到对话历史
            self.chat_history.append({"role": "user", "content": user_input})
            self.update_chat_history()
            
            # 清空输入框
            self.user_input.clear()
            
            if self.df is not None:
                # 开始分析
                self.start_analysis(user_input)
            else:
                error_msg = "请先上传数据文件"
                self.chat_history.append({"role": "assistant", "content": error_msg})
                self.update_chat_history()
    
    def reset_data(self):
        # 重置数据为原始状态
        if self.original_df is not None:
            self.df = self.original_df.copy()
            self.update_data_preview()
            self.update_column_comboboxes()
            QMessageBox.information(self, "数据重置", "数据已成功重置为原始状态！")
        else:
            QMessageBox.warning(self, "数据重置", "请先上传数据文件！")
    
    def download_processed_data(self):
        # 下载处理后的数据
        if not hasattr(self, 'df') or self.df.empty:
            QMessageBox.warning(self, "下载失败", "没有数据可供下载")
            return
        
        # 打开文件保存对话框
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存处理后的数据", "", "CSV Files (*.csv);;Excel Files (*.xlsx)"
        )
        
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    self.df.to_csv(file_path, index=False, encoding='utf-8-sig')
                elif file_path.endswith('.xlsx'):
                    self.df.to_excel(file_path, index=False, engine='openpyxl')
                QMessageBox.information(self, "下载成功", f"处理后的数据已保存到 {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "下载失败", f"保存数据时出错: {str(e)}")
    
    def confirm_processing_completed(self):
        # 确认数据处理完成并加载到分析模块
        if not hasattr(self, 'df') or self.df.empty:
            QMessageBox.warning(self, "加载失败", "没有数据可供加载到分析模块")
            return
        
        # 确保数据已准备好用于分析
        self.df_analysis = self.df.copy()
        
        # 更新数据分析模块的列选择
        self.update_column_comboboxes()
        
        QMessageBox.information(self, "加载成功", "处理后的数据已加载到分析模块")
    
    def perform_regression_analysis(self):
        # 执行回归分析
        try:
            method = self.regression_method.currentText()
            dep_var = self.regression_y.currentText()
            
            # 获取所有选中的自变量
            selected_items = self.regression_x.selectedItems()
            if not selected_items:
                QMessageBox.warning(self, "警告", "请至少选择一个自变量！")
                return
            
            indep_vars = [item.text() for item in selected_items]
            
            if not dep_var:
                QMessageBox.warning(self, "警告", "请选择因变量！")
                return
            
            # 使用处理后的数据（如果存在），否则使用原始数据
            analysis_df = self.df_analysis if hasattr(self, 'df_analysis') and not self.df_analysis.empty else self.df
            
            # 确保因变量是数值类型
            analysis_df[dep_var] = pd.to_numeric(analysis_df[dep_var], errors='coerce')
            
            # 确保所有自变量是数值类型
            for var in indep_vars:
                analysis_df[var] = pd.to_numeric(analysis_df[var], errors='coerce')
            
            # 移除包含NaN的行
            df_clean = analysis_df[[dep_var] + indep_vars].dropna()
            
            if df_clean.empty:
                # 检查原始数据中有多少行是完整的
                total_rows = len(self.df)
                QMessageBox.warning(self, "警告", f"没有足够的数据进行分析！\n原始数据共有{total_rows}行，但选择的变量组合包含缺失值，导致没有完整数据行可用。\n建议：1. 选择其他变量组合；2. 使用数据预处理功能填充缺失值后再分析。")
                return
            elif len(df_clean) < 5:
                # 如果有效数据行太少（少于5行），也提示用户
                QMessageBox.warning(self, "警告", f"有效数据行太少（仅{len(df_clean)}行），可能影响分析结果的可靠性。\n建议：1. 选择其他变量组合；2. 使用数据预处理功能填充缺失值后再分析。")
                # 这里我们仍然允许分析继续，但给出警告
            
            import statsmodels.api as sm
            
            # 添加常数项
            X = sm.add_constant(df_clean[indep_vars])
            y = df_clean[dep_var]
            
            if method == "线性回归":
                model = sm.OLS(y, X).fit()
            elif method == "Logistic回归":
                # 确保因变量是二分类变量
                if len(df_clean[dep_var].unique()) != 2:
                    QMessageBox.warning(self, "警告", "Logistic回归要求因变量是二分类变量")
                    return
                model = sm.Logit(y, X).fit()
            elif method == "Cox回归":
                QMessageBox.warning(self, "警告", "Cox回归需要生存时间数据，暂不支持")
                return
            else:
                QMessageBox.warning(self, "警告", "不支持的回归方法")
                return
            
            # 生成结果
            result_text = f"{method}结果：\n"
            result_text += "\n模型摘要：\n"
            result_text += str(model.summary())
            
            # 保存结果到self.analysis_result
            self.analysis_result = {
                "code": f"# {method}分析代码\nimport pandas as pd\nimport statsmodels.api as sm\n\n# 数据准备\nX = sm.add_constant(df_clean[{indep_vars}])\ny = df_clean[{dep_var}]\n\n# 模型拟合\nmodel = sm.{model.__class__.__name__}(y, X).fit()\n\n# 结果输出\nprint(model.summary())",
                "result": result_text,
                "summary": f"## {method}分析结果\n\n模型摘要：\n\n```\n{model.summary()}\n```",
                "plt": None  # 回归分析没有直接的可视化结果
            }
            
            # 更新分析结果标签页
            self.update_result_tab()
            self.tab_widget.setCurrentIndex(self.tab_widget.indexOf(self.result_tab))
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"回归分析失败：{str(e)}")
    
    def perform_propensity_score(self):
        # 执行倾向性评分分析（旧方法，保留兼容）
        self.calculate_propensity_score()
    
    def calculate_propensity_score(self):
        # 计算倾向性评分
        try:
            treatment_var = self.treatment_var.currentText()
            
            # 获取所有选中的协变量
            selected_covariates = self.covariates_var.selectedItems()
            if not selected_covariates:
                QMessageBox.warning(self, "警告", "请至少选择一个协变量！")
                return
            
            covariates_vars = [item.text() for item in selected_covariates]
            
            if not treatment_var:
                QMessageBox.warning(self, "警告", "请选择处理组变量！")
                return
            
            # 使用处理后的数据（如果存在），否则使用原始数据
            analysis_df = self.df_analysis if hasattr(self, 'df_analysis') and not self.df_analysis.empty else self.df
            
            # 确保变量是数值类型
            analysis_df[treatment_var] = pd.to_numeric(analysis_df[treatment_var], errors='coerce')
            for var in covariates_vars:
                analysis_df[var] = pd.to_numeric(analysis_df[var], errors='coerce')
            
            # 移除包含NaN的行
            df_clean = analysis_df[[treatment_var] + covariates_vars].dropna()
            
            if df_clean.empty:
                # 检查原始数据中有多少行是完整的
                total_rows = len(self.df)
                QMessageBox.warning(self, "警告", f"没有足够的数据进行分析！\n原始数据共有{total_rows}行，但选择的变量组合包含缺失值，导致没有完整数据行可用。\n建议：1. 选择其他变量组合；2. 使用数据预处理功能填充缺失值后再分析。")
                return
            elif len(df_clean) < 5:
                # 如果有效数据行太少（少于5行），也提示用户
                QMessageBox.warning(self, "警告", f"有效数据行太少（仅{len(df_clean)}行），可能影响分析结果的可靠性。\n建议：1. 选择其他变量组合；2. 使用数据预处理功能填充缺失值后再分析。")
            
            # 执行倾向性评分计算
            from sklearn.linear_model import LogisticRegression
            
            X = df_clean[covariates_vars]
            y = df_clean[treatment_var]
            
            # 训练模型
            lr = LogisticRegression()
            lr.fit(X, y)
            
            # 计算倾向性得分
            propensity_scores = lr.predict_proba(X)[:, 1]
            
            # 将倾向性评分添加到原始数据中
            analysis_df['propensity_score'] = pd.NA  # 初始化
            analysis_df.loc[df_clean.index, 'propensity_score'] = propensity_scores
            
            # 保存更新后的数据
            if hasattr(self, 'df_analysis'):
                self.df_analysis = analysis_df
            else:
                self.df_analysis = analysis_df.copy()
            
            # 生成结果
            result_text = "倾向性评分计算结果：\n"
            result_text += f"\n处理组变量：{treatment_var}"
            result_text += f"\n协变量：{covariates_vars}"
            result_text += f"\n\n模型系数：{lr.coef_}"
            result_text += f"\n截距：{lr.intercept_}"
            result_text += f"\n\n倾向性得分示例：{propensity_scores[:5]}"
            result_text += f"\n\n倾向性评分已添加到数据中，列名为 'propensity_score'"
            
            # 保存结果到self.analysis_result
            self.analysis_result = {
                "code": f"# 倾向性评分计算代码\nimport pandas as pd\nfrom sklearn.linear_model import LogisticRegression\n\n# 数据准备\nX = df_clean[{covariates_vars}]\ny = df_clean[{treatment_var}]\n\n# 模型拟合\nlr = LogisticRegression()\nlr.fit(X, y)\n\n# 计算倾向性评分\npropensity_scores = lr.predict_proba(X)[:, 1]\n\n# 将倾向性评分添加到数据中\ndf['propensity_score'] = pd.NA\ndf.loc[df_clean.index, 'propensity_score'] = propensity_scores\n\n# 结果输出\nprint('处理组变量：', {treatment_var})\nprint('协变量：', {covariates_vars})\nprint('模型系数：', lr.coef_)\nprint('截距：', lr.intercept_)\nprint('倾向性得分示例：', propensity_scores[:5])",
                "result": result_text,
                "summary": f"## 倾向性评分计算结果\n\n- 处理组变量：{treatment_var}\n- 协变量：{covariates_vars}\n- 模型系数：{lr.coef_}\n- 截距：{lr.intercept_}\n- 倾向性得分示例：{propensity_scores[:5]}\n- 倾向性评分已添加到数据中，列名为 'propensity_score'",
                "plt": None
            }
            
            # 更新分析结果标签页
            self.update_result_tab()
            self.tab_widget.setCurrentIndex(self.tab_widget.indexOf(self.result_tab))
            
            QMessageBox.information(self, "成功", "倾向性评分计算完成！倾向性评分已添加到数据中。")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"倾向性评分计算失败：{str(e)}")
    
    def show_smd_results(self):
        # 展示标准化均数差(SMD)结果
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            treatment_var = self.treatment_var.currentText()
            
            # 获取所有选中的协变量
            selected_covariates = self.covariates_var.selectedItems()
            if not selected_covariates:
                QMessageBox.warning(self, "警告", "请至少选择一个协变量！")
                return
            
            covariates_vars = [item.text() for item in selected_covariates]
            
            if not treatment_var:
                QMessageBox.warning(self, "警告", "请选择处理组变量！")
                return
            
            # 使用处理后的数据（如果存在），否则使用原始数据
            analysis_df = self.df_analysis if hasattr(self, 'df_analysis') and not self.df_analysis.empty else self.df
            
            # 确保数据包含倾向性评分
            if 'propensity_score' not in analysis_df.columns:
                QMessageBox.warning(self, "警告", "请先计算倾向性评分！")
                return
            
            # 计算SMD
            def calculate_smd(df, treatment, variables):
                """计算标准化均数差"""
                smd_values = {}
                for var in variables:
                    treated = df[df[treatment] == 1][var]
                    control = df[df[treatment] == 0][var]
                    mean_diff = treated.mean() - control.mean()
                    pooled_std = np.sqrt((treated.var() + control.var()) / 2)
                    smd = mean_diff / pooled_std if pooled_std != 0 else 0
                    smd_values[var] = abs(smd)
                return smd_values
            
            # 计算匹配前后的SMD
            smd_before = calculate_smd(analysis_df, treatment_var, covariates_vars)
            
            # 选择SMD展示方式
            display_type = self.smd_display_combo.currentText()
            
            # 创建SMD图表
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if display_type == "柱状图":
                # 柱状图展示
                vars = list(smd_before.keys())
                values = list(smd_before.values())
                ax.bar(vars, values, color='#1976d2', alpha=0.7)
                ax.axhline(y=0.1, color='red', linestyle='--', label='SMD阈值 (0.1)')
                ax.set_xlabel('协变量')
                ax.set_ylabel('标准化均数差 (SMD)')
                ax.set_title('倾向性评分匹配前的协变量平衡情况')
                plt.xticks(rotation=45, ha='right')
            else:
                # 森林图展示
                vars = list(smd_before.keys())
                values = list(smd_before.values())
                y_pos = np.arange(len(vars))
                ax.errorbar(values, y_pos, xerr=0, fmt='o', color='#1976d2', capsize=5)
                ax.axvline(x=0.1, color='red', linestyle='--', label='SMD阈值 (0.1)')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(vars)
                ax.set_xlabel('标准化均数差 (SMD)')
                ax.set_title('倾向性评分匹配前的协变量平衡情况')
            
            ax.legend()
            plt.tight_layout()
            
            # 保存结果到self.analysis_result
            self.analysis_result = {
                "code": f"# SMD计算和可视化代码\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n# 计算SMD函数\ndef calculate_smd(df, treatment, variables):\n    smd_values = {{}}\n    for var in variables:\n        treated = df[df[treatment] == 1][var]\n        control = df[df[treatment] == 0][var]\n        mean_diff = treated.mean() - control.mean()\n        pooled_std = np.sqrt((treated.var() + control.var()) / 2)\n        smd = mean_diff / pooled_std if pooled_std != 0 else 0\n        smd_values[var] = abs(smd)\n    return smd_values\n\n# 计算SMD\nsmd_before = calculate_smd(df, '{treatment_var}', {covariates_vars})\n\n# 可视化SMD\nfig, ax = plt.subplots(figsize=(10, 6))\n# {'柱状图' if display_type == '柱状图' else '森林图'} 代码...\nplt.show()",
                "result": f"SMD计算结果：\n{smd_before}",
                "summary": f"## 标准化均数差(SMD)评估结果\n\n- 展示方式：{display_type}\n- 协变量数量：{len(covariates_vars)}\n- SMD阈值：0.1（<0.1表示平衡良好）\n- 详细结果：{smd_before}",
                "figure": fig
            }
            
            # 更新分析结果标签页
            self.update_result_tab()
            self.tab_widget.setCurrentIndex(self.tab_widget.indexOf(self.result_tab))
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"SMD结果展示失败：{str(e)}")
    
    def perform_ps_stratification(self):
        # 执行PS分层
        try:
            treatment_var = self.treatment_var.currentText()
            
            # 获取分层数量
            strata_num = int(self.strata_num_combo.currentText().split('层')[0])
            
            # 使用处理后的数据（如果存在），否则使用原始数据
            analysis_df = self.df_analysis if hasattr(self, 'df_analysis') and not self.df_analysis.empty else self.df
            
            # 确保数据包含倾向性评分
            if 'propensity_score' not in analysis_df.columns:
                QMessageBox.warning(self, "警告", "请先计算倾向性评分！")
                return
            
            # 执行PS分层，处理NaN值
            # 首先移除NaN值，然后对有效数据进行分层
            valid_indices = analysis_df['propensity_score'].notna()
            # 初始化分层结果为NaN
            analysis_df['ps_stratum'] = pd.NA
            # 只对非NaN值进行分层
            analysis_df.loc[valid_indices, 'ps_stratum'] = pd.qcut(
                analysis_df.loc[valid_indices, 'propensity_score'], 
                q=strata_num, 
                labels=False, 
                duplicates='drop'
            )
            
            # 保存更新后的数据
            if hasattr(self, 'df_analysis'):
                self.df_analysis = analysis_df
            else:
                self.df_analysis = analysis_df.copy()
            
            # 生成结果
            result_text = f"PS分层结果：\n"
            result_text += f"\n分层数量：{strata_num}层"
            result_text += f"\n\n各层样本数：\n{analysis_df['ps_stratum'].value_counts().sort_index()}"
            result_text += f"\n\n分层信息已添加到数据中，列名为 'ps_stratum'"
            
            # 保存结果到self.analysis_result
            self.analysis_result = {
                "code": f"# PS分层代码\nimport pandas as pd\n\n# 执行PS分层\ndf['ps_stratum'] = pd.qcut(df['propensity_score'], q={strata_num}, labels=False, duplicates='drop')\n\n# 结果输出\nprint('分层数量：', {strata_num}层)\nprint('各层样本数：')\nprint(df['ps_stratum'].value_counts().sort_index())",
                "result": result_text,
                "summary": f"## PS分层结果\n\n- 分层数量：{strata_num}层\n- 分层信息已添加到数据中，列名为 'ps_stratum'\n- 各层样本数：{dict(analysis_df['ps_stratum'].value_counts().sort_index())}",
                "plt": None
            }
            
            # 更新分析结果标签页
            self.update_result_tab()
            self.tab_widget.setCurrentIndex(self.tab_widget.indexOf(self.result_tab))
            
            QMessageBox.information(self, "成功", f"PS分层完成！已将数据分为{strata_num}层，分层信息已添加到数据中。")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"PS分层失败：{str(e)}")
    
    def perform_ps_matching(self):
        # 执行PS匹配
        try:
            treatment_var = self.treatment_var.currentText()
            
            # 获取匹配方法
            matching_method = self.matching_method_combo.currentText()
            
            # 使用处理后的数据（如果存在），否则使用原始数据
            analysis_df = self.df_analysis if hasattr(self, 'df_analysis') and not self.df_analysis.empty else self.df
            
            # 确保数据包含倾向性评分
            if 'propensity_score' not in analysis_df.columns:
                QMessageBox.warning(self, "警告", "请先计算倾向性评分！")
                return
            
            # 执行PS匹配
            from sklearn.metrics import pairwise_distances
            
            # 分离处理组和对照组
            treated = analysis_df[analysis_df[treatment_var] == 1]
            control = analysis_df[analysis_df[treatment_var] == 0]
            
            # 过滤掉倾向性评分中的NaN值
            treated = treated[treated['propensity_score'].notna()]
            control = control[control['propensity_score'].notna()]
            
            # 检查过滤后是否还有数据
            if len(treated) == 0 or len(control) == 0:
                QMessageBox.warning(self, "警告", "过滤NaN值后没有足够的数据进行匹配！")
                return
            
            # 获取倾向性评分
            treated_ps = treated['propensity_score'].values.reshape(-1, 1)
            control_ps = control['propensity_score'].values.reshape(-1, 1)
            
            matched_indices = []
            
            if matching_method == "最近邻匹配":
                # 最近邻匹配
                distances = pairwise_distances(treated_ps, control_ps)
                for i in range(len(treated)):
                    nearest_idx = np.argmin(distances[i])
                    matched_indices.append(control.index[nearest_idx])
            elif matching_method == "半径匹配":
                # 半径匹配（卡尺0.05）
                caliper = 0.05
                for i in range(len(treated)):
                    distances = np.abs(treated_ps[i] - control_ps)
                    matches = np.where(distances <= caliper)[0]
                    if len(matches) > 0:
                        matched_indices.append(control.index[matches[0]])
            else:  # 核匹配
                # 核匹配（简单实现）
                QMessageBox.information(self, "提示", "核匹配功能正在开发中...")
                return
            
            # 合并匹配后的样本
            matched_data = pd.concat([treated, control.loc[matched_indices]])
            
            # 保存更新后的数据
            self.df_analysis = matched_data
            
            # 生成结果
            result_text = f"PS匹配结果：\n"
            result_text += f"\n匹配方法：{matching_method}"
            result_text += f"\n处理组样本数：{len(treated)}"
            result_text += f"\n对照组样本数：{len(control)}"
            result_text += f"\n匹配后样本数：{len(matched_data)}"
            result_text += f"\n匹配率：{len(matched_data) / (len(treated) + len(control)):.2f}"
            
            # 保存结果到self.analysis_result
            self.analysis_result = {
                "code": f"# PS{matched_data}代码\nimport pandas as pd\nfrom sklearn.metrics import pairwise_distances\n\n# 分离处理组和对照组\ntreated = df[df['{treatment_var}'] == 1]\ncontrol = df[df['{treatment_var}'] == 0]\n\n# 获取倾向性评分\ntreated_ps = treated['propensity_score'].values.reshape(-1, 1)\ncontrol_ps = control['propensity_score'].values.reshape(-1, 1)\n\n# {matching_method}代码...\n# 合并匹配后的样本\nmatched_data = pd.concat([treated, control.loc[matched_indices]])",
                "result": result_text,
                "summary": f"## PS匹配结果\n\n- 匹配方法：{matching_method}\n- 处理组样本数：{len(treated)}\n- 对照组样本数：{len(control)}\n- 匹配后样本数：{len(matched_data)}\n- 匹配率：{len(matched_data) / (len(treated) + len(control)):.2f}",
                "plt": None
            }
            
            # 更新分析结果标签页
            self.update_result_tab()
            self.tab_widget.setCurrentIndex(self.tab_widget.indexOf(self.result_tab))
            
            QMessageBox.information(self, "成功", "PS匹配完成！匹配后的数据已保存。")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"PS匹配失败：{str(e)}")
    
    def add_ps_to_covariates(self):
        # 将PS加入协变量调整
        try:
            # 使用处理后的数据（如果存在），否则使用原始数据
            analysis_df = self.df_analysis if hasattr(self, 'df_analysis') and not self.df_analysis.empty else self.df
            
            # 确保数据包含倾向性评分
            if 'propensity_score' not in analysis_df.columns:
                QMessageBox.warning(self, "警告", "请先计算倾向性评分！")
                return
            
            # 生成结果
            result_text = "将PS加入协变量调整：\n"
            result_text += "\n倾向性评分已添加到数据中，您可以在后续分析中直接使用 'propensity_score' 作为协变量。\n"
            result_text += "例如：在回归分析中，将 'propensity_score' 作为协变量加入模型，以控制混杂因素的影响。"
            
            # 保存结果到self.analysis_result
            self.analysis_result = {
                "code": f"# 将PS加入协变量调整示例代码\nimport statsmodels.api as sm\n\n# 示例：在回归模型中加入PS作为协变量\n# 假设y是结局变量，x1, x2是其他协变量\nX = sm.add_constant(df[['x1', 'x2', 'propensity_score']])\ny = df['y']\n\n# 拟合线性回归模型\nmodel = sm.OLS(y, X).fit()\nprint(model.summary())",
                "result": result_text,
                "summary": "## 将PS加入协变量调整\n\n倾向性评分已添加到数据中，列名为 'propensity_score'。\n\n**使用建议：**\n1. 在回归分析中，将 'propensity_score' 作为协变量加入模型\n2. 可以考虑加入PS的多项式项或交互项，以更好地控制混杂\n3. 对于Logistic回归，也可以使用PS加权方法",
                "plt": None
            }
            
            # 更新分析结果标签页
            self.update_result_tab()
            self.tab_widget.setCurrentIndex(self.tab_widget.indexOf(self.result_tab))
            
            QMessageBox.information(self, "提示", "倾向性评分已准备好作为协变量使用！")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"将PS加入协变量调整失败：{str(e)}")
    
    def on_outcome_type_changed(self, index):
        # 当结局类型改变时，更新分析方法选项
        self.temporal_method_combo.clear()
        if index == 0:
            # 连续性结局
            self.temporal_method_combo.addItems(["重复测量方差分析", "GEE", "协方差分析"])
        else:
            # 分类结局
            self.temporal_method_combo.addItems(["时依性COX分析", "多水平模型"])
    
    def perform_temporal_analysis(self):
        # 执行时序性分析
        try:
            outcome_type = self.outcome_type_combo.currentText()
            method = self.temporal_method_combo.currentText()
            outcome_var = self.temporal_y.currentText()
            time_var = self.time_var.currentText()
            subject_id = self.subject_id_var.currentText()
            group_var = self.group_var.currentText()
            
            # 获取选中的协变量
            selected_covariates = self.temporal_covariates_var.selectedItems()
            covariates_vars = [item.text() for item in selected_covariates] if selected_covariates else []
            
            # 验证必填字段
            if not outcome_var or not time_var or not subject_id:
                QMessageBox.warning(self, "警告", "请填写所有必填字段！")
                return
            
            # 使用处理后的数据（如果存在），否则使用原始数据
            analysis_df = self.df_analysis if hasattr(self, 'df_analysis') and not self.df_analysis.empty else self.df
            
            # 确保数据类型正确
            analysis_df[outcome_var] = pd.to_numeric(analysis_df[outcome_var], errors='coerce')
            analysis_df[time_var] = pd.to_numeric(analysis_df[time_var], errors='coerce')
            if group_var:
                analysis_df[group_var] = pd.to_numeric(analysis_df[group_var], errors='coerce')
            for var in covariates_vars:
                analysis_df[var] = pd.to_numeric(analysis_df[var], errors='coerce')
            
            # 移除包含NaN的行
            required_cols = [outcome_var, time_var, subject_id] + ([group_var] if group_var else []) + covariates_vars
            df_clean = analysis_df[required_cols].dropna()
            
            if df_clean.empty:
                total_rows = len(self.df)
                QMessageBox.warning(self, "警告", f"没有足够的数据进行分析！\n原始数据共有{total_rows}行，但选择的变量组合包含缺失值，导致没有完整数据行可用。\n建议：1. 选择其他变量组合；2. 使用数据预处理功能填充缺失值后再分析。")
                return
            elif len(df_clean) < 10:
                QMessageBox.warning(self, "警告", f"有效数据行太少（仅{len(df_clean)}行），可能影响分析结果的可靠性。\n建议：1. 选择其他变量组合；2. 使用数据预处理功能填充缺失值后再分析。")
            
            # 根据结局类型和方法执行不同的分析
            result_text, figure = None, None
            
            if outcome_type == "连续性结局":
                result_text, figure = self.analyze_continuous_temporal(df_clean, outcome_var, time_var, subject_id, group_var, covariates_vars, method)
            else:
                result_text, figure = self.analyze_categorical_temporal(df_clean, outcome_var, time_var, subject_id, group_var, covariates_vars, method)
            
            # 保存结果到self.analysis_result
            self.analysis_result = {
                "code": f"# 时序性分析代码\n# 结局类型：{outcome_type}\n# 分析方法：{method}\n# 因变量：{outcome_var}\n# 时间变量：{time_var}\n# 受试者ID：{subject_id}\n# 分组变量：{group_var}\n# 协变量：{covariates_vars}\n\n# 分析代码将根据实际情况自动生成\n",
                "result": result_text,
                "summary": f"## 时序性分析结果\n\n- 结局类型：{outcome_type}\n- 分析方法：{method}\n- 因变量：{outcome_var}\n- 时间变量：{time_var}\n- 受试者ID：{subject_id}\n- 分组变量：{group_var}\n- 协变量：{covariates_vars}\n\n详细结果：\n{result_text}",
                "figure": figure
            }
            
            # 更新分析结果标签页
            self.update_result_tab()
            self.tab_widget.setCurrentIndex(self.tab_widget.indexOf(self.result_tab))
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"时序性分析失败：{str(e)}")
    
    def analyze_continuous_temporal(self, df, outcome_var, time_var, subject_id, group_var, covariates_vars, method):
        # 分析连续性结局的时序数据
        try:
            if method == "重复测量方差分析":
                return self.repeated_measures_anova(df, outcome_var, subject_id, group_var, time_var)
            elif method == "GEE":
                return self.analyze_gee(df, outcome_var, time_var, subject_id, group_var, covariates_vars)
            elif method == "协方差分析":
                return self.analyze_ancova(df, outcome_var, time_var, subject_id, group_var, covariates_vars)
            else:
                raise ValueError(f"不支持的分析方法：{method}")
        except Exception as e:
            return f"分析失败：{str(e)}", None
    
    def analyze_categorical_temporal(self, df, outcome_var, time_var, subject_id, group_var, covariates_vars, method):
        # 分析分类结局的时序数据
        try:
            if method == "时依性COX分析":
                return self.analyze_time_dependent_cox(df, outcome_var, time_var, subject_id, group_var, covariates_vars)
            elif method == "多水平模型":
                return self.analyze_multi_level_model(df, outcome_var, time_var, subject_id, group_var, covariates_vars)
            else:
                raise ValueError(f"不支持的分析方法：{method}")
        except Exception as e:
            return f"分析失败：{str(e)}", None
    
    def repeated_measures_anova(self, df, outcome_var, subject_id, group_var, time_var):
        # 执行重复测量方差分析
        try:
            import pingouin as pg
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 设置中文显示
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 执行重复测量方差分析
            if group_var:
                # 混合设计ANOVA
                aov_results = pg.mixed_anova(
                    data=df,
                    dv=outcome_var,
                    within=time_var,
                    between=group_var,
                    subject=subject_id
                )
            else:
                # 单因素重复测量ANOVA
                aov_results = pg.rm_anova(
                    data=df,
                    dv=outcome_var,
                    within=time_var,
                    subject=subject_id
                )
            
            # 生成结果文本
            result_text = "重复测量方差分析结果：\n"
            result_text += f"\n{str(aov_results)}"
            
            # 可视化
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=df, x=time_var, y=outcome_var, hue=group_var if group_var else subject_id, err_style='bars', ax=ax)
            ax.set_title(f'\n{outcome_var}随{time_var}的变化趋势')
            ax.set_xlabel(time_var)
            ax.set_ylabel(outcome_var)
            plt.tight_layout()
            
            return result_text, fig
        except Exception as e:
            raise ValueError(f"重复测量方差分析失败：{str(e)}")
    
    def analyze_gee(self, df, outcome_var, time_var, subject_id, group_var, covariates_vars):
        # 执行广义估计方程（GEE）分析
        try:
            import statsmodels.api as sm
            from statsmodels.formula.api import gee
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 设置中文显示
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 构建公式
            formula_parts = [outcome_var, '~', time_var]
            if group_var:
                formula_parts.extend(['+', group_var])
                formula_parts.extend(['+', time_var, '*', group_var])
            if covariates_vars:
                formula_parts.extend(['+', '+'.join(covariates_vars)])
            formula = ''.join(formula_parts)
            
            # 执行GEE分析
            model = gee(
                formula=formula,
                groups=df[subject_id],
                data=df,
                cov_struct=sm.cov_struct.Exchangeable(),
                family=sm.families.Gaussian()
            )
            
            result = model.fit()
            
            # 生成结果文本
            result_text = "GEE分析结果：\n"
            result_text += f"\n{str(result.summary())}"
            
            # 可视化
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=df, x=time_var, y=outcome_var, hue=group_var if group_var else subject_id, ax=ax)
            ax.set_title(f'\n{outcome_var}随{time_var}的变化趋势')
            ax.set_xlabel(time_var)
            ax.set_ylabel(outcome_var)
            plt.tight_layout()
            
            return result_text, fig
        except Exception as e:
            raise ValueError(f"GEE分析失败：{str(e)}")
    
    def analyze_ancova(self, df, outcome_var, time_var, subject_id, group_var, covariates_vars):
        # 执行协方差分析
        try:
            import statsmodels.api as sm
            from statsmodels.formula.api import ols
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 设置中文显示
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 构建公式
            formula_parts = [outcome_var, '~', time_var]
            if group_var:
                formula_parts.extend(['+', group_var])
            if covariates_vars:
                formula_parts.extend(['+', '+'.join(covariates_vars)])
            formula = ''.join(formula_parts)
            
            # 执行协方差分析
            model = ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            # 生成结果文本
            result_text = "协方差分析结果：\n"
            result_text += f"\n{str(anova_table)}"
            result_text += f"\n\n回归系数：\n{str(model.params)}"
            
            # 可视化
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df, x=time_var, y=outcome_var, hue=group_var if group_var else None, ax=ax)
            sns.lineplot(data=df, x=time_var, y=model.fittedvalues, color='red', ax=ax)
            ax.set_title(f'\n{outcome_var}与{time_var}的关系（协方差分析）')
            ax.set_xlabel(time_var)
            ax.set_ylabel(outcome_var)
            plt.tight_layout()
            
            return result_text, fig
        except Exception as e:
            raise ValueError(f"协方差分析失败：{str(e)}")
    
    def analyze_time_dependent_cox(self, df, outcome_var, time_var, subject_id, group_var, covariates_vars):
        # 执行时依性COX分析
        try:
            from lifelines import CoxTimeVaryingFitter
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 设置中文显示
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 创建数据副本，避免修改原始数据
            df_copy = df.copy()
            
            # 准备数据 - 确保所有数值变量都是数值类型
            df_copy[time_var] = pd.to_numeric(df_copy[time_var], errors='coerce')  # 确保时间变量是数值类型
            df_copy[outcome_var] = df_copy[outcome_var].astype(int)  # 确保结局变量是整数类型
            
            # 确保分组变量是数值类型
            if group_var:
                df_copy[group_var] = pd.to_numeric(df_copy[group_var], errors='coerce').astype(int)
            
            # 确保所有协变量是数值类型
            for var in covariates_vars:
                df_copy[var] = pd.to_numeric(df_copy[var], errors='coerce')
            
            # 为时依性COX分析添加开始时间和结束时间
            # 对于每天的记录，开始时间是前一天，结束时间是当天
            df_copy['start_time'] = df_copy[time_var] - 1
            df_copy['stop_time'] = df_copy[time_var]
            
            # 选择分析变量
            analysis_vars = [subject_id, 'start_time', 'stop_time', outcome_var]
            if group_var:
                analysis_vars.append(group_var)
            analysis_vars.extend(covariates_vars)
            
            # 执行时依性COX分析
            ctv = CoxTimeVaryingFitter()
            ctv.fit(df_copy[analysis_vars], id_col=subject_id, event_col=outcome_var, start_col='start_time', stop_col='stop_time')
            
            # 生成结果文本
            result_text = "时依性COX分析结果：\n"
            result_text += f"\n{str(ctv.summary)}"
            
            # 可视化
            fig, ax = plt.subplots(figsize=(10, 6))
            ctv.plot_partial_effects_on_outcome(time_var, values=df_copy[time_var].quantile([0.25, 0.5, 0.75]), ax=ax)
            ax.set_title(f'\n时依性COX分析结果：{outcome_var}')
            plt.tight_layout()
            
            return result_text, fig
        except Exception as e:
            raise ValueError(f"时依性COX分析失败：{str(e)}")
    
    def analyze_multi_level_model(self, df, outcome_var, time_var, subject_id, group_var, covariates_vars):
        # 执行多水平模型分析
        try:
            import statsmodels.api as sm
            from statsmodels.regression.mixed_linear_model import MixedLM
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 设置中文显示
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 创建数据副本，避免修改原始数据
            df_copy = df.copy()
            
            # 确保所有数值变量都是数值类型
            df_copy[outcome_var] = pd.to_numeric(df_copy[outcome_var], errors='coerce').astype(float)
            df_copy[time_var] = pd.to_numeric(df_copy[time_var], errors='coerce').astype(float)
            
            # 确保分组变量是数值类型
            if group_var:
                df_copy[group_var] = pd.to_numeric(df_copy[group_var], errors='coerce').astype(float)
            
            # 确保所有协变量是数值类型
            for var in covariates_vars:
                if var in df_copy.columns:
                    df_copy[var] = pd.to_numeric(df_copy[var], errors='coerce').astype(float)
            
            # 构建固定效应公式
            fixed_effects_parts = [outcome_var, '~', time_var]
            if group_var:
                fixed_effects_parts.extend(['+', group_var])
            if covariates_vars:
                fixed_effects_parts.extend(['+', '+'.join(covariates_vars)])
            fixed_effects = ''.join(fixed_effects_parts)
            
            # 执行多水平模型分析（随机截距）
            model = MixedLM.from_formula(fixed_effects, groups=df_copy[subject_id], data=df_copy)
            result = model.fit()
            
            # 生成结果文本
            result_text = "多水平模型分析结果：\n"
            result_text += f"\n{str(result.summary())}"
            
            # 可视化
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=df_copy, x=time_var, y=outcome_var, hue=group_var if group_var else subject_id, err_style='bars', ax=ax)
            ax.set_title(f'\n{outcome_var}随{time_var}的变化趋势')
            ax.set_xlabel(time_var)
            ax.set_ylabel(outcome_var)
            plt.tight_layout()
            
            return result_text, fig
        except Exception as e:
            raise ValueError(f"多水平模型分析失败：{str(e)}")
    
    def start_analysis(self, user_input):
        # 显示加载消息
        loading_msg = "正在分析..."
        self.chat_history.append({"role": "assistant", "content": loading_msg})
        self.update_chat_history()
        
        # 创建分析线程
        self.analysis_thread = AnalysisThread(
            self.df, user_input, self.data_types, self.code_gen, self.result_summarizer
        )
        self.analysis_thread.result_signal.connect(self.on_analysis_complete)
        self.analysis_thread.error_signal.connect(self.on_analysis_error)
        self.analysis_thread.start()
    
    def on_analysis_complete(self, result_data):
        # 移除加载消息
        self.chat_history.pop()
        
        # 保存分析结果
        self.analysis_result = result_data
        
        # 添加结果到对话历史
        self.chat_history.append({"role": "assistant", "content": result_data["summary"]})
        self.update_chat_history()
        
        # 更新分析结果选项卡
        self.update_result_tab()
        
        # 切换到结果选项卡
        self.tab_widget.setCurrentIndex(1)
    
    def on_analysis_error(self, error_msg):
        # 移除加载消息
        self.chat_history.pop()
        
        # 添加错误消息到对话历史
        self.chat_history.append({"role": "assistant", "content": error_msg})
        self.update_chat_history()
    
    def update_chat_history(self):
        # 清空当前文本
        self.chat_history_text.clear()
        
        # 重新添加所有消息
        for msg in self.chat_history:
            if msg["role"] == "user":
                self.chat_history_text.append(f"**您**: {msg['content']}")
            else:
                self.chat_history_text.append(f"**系统**: {msg['content']}")
    
    def update_result_tab(self):
        # 清空当前布局
        self.clear_layout(self.result_layout)
        
        # 创建标签页控件来组织分析结果
        result_tabs = QTabWidget()
        result_tabs.setStyleSheet("""
            QTabWidget::tab-bar {
                alignment: left;
            }
            QTabBar::tab {
                background: rgba(25, 118, 210, 0.1);
                color: #1976d2;
                padding: 8px 16px;
                border: 1px solid rgba(25, 118, 210, 0.3);
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                min-width: 100px;
            }
            QTabBar::tab:selected {
                background: white;
                color: #1976d2;
                font-weight: bold;
            }
        """)
        
        # ========== 分析总结标签页 ==========
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        
        # 分析总结内容
        summary_text = QTextEdit()
        summary_text.setMarkdown(self.analysis_result["summary"])
        summary_text.setReadOnly(True)
        summary_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid rgba(25, 118, 210, 0.2);
                border-radius: 8px;
                padding: 10px;
                background-color: rgba(255, 255, 255, 0.9);
                min-height: 200px;
            }
        """)
        summary_layout.addWidget(summary_text)
        
        # 下载按钮布局
        download_summary_layout = QHBoxLayout()
        download_summary_layout.setAlignment(Qt.AlignRight)
        
        # 结果下载按钮
        download_summary_btn = QPushButton("📥 下载分析结果")
        download_summary_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #66bb6a, stop:1 #43a047);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #81c784, stop:1 #66bb6a);
            }
        """)
        download_summary_btn.clicked.connect(lambda: self.download_result("summary"))
        download_summary_layout.addWidget(download_summary_btn)
        
        summary_layout.addLayout(download_summary_layout)
        
        # 将标签页添加到标签页控件
        result_tabs.addTab(summary_tab, "分析总结")
        
        # ========== 生成的代码标签页 ==========
        code_tab = QWidget()
        code_layout = QVBoxLayout(code_tab)
        
        # 代码内容
        code_text = QPlainTextEdit()
        code_text.setPlainText(self.analysis_result["code"])
        code_text.setReadOnly(True)
        code_text.setStyleSheet("""
            QPlainTextEdit {
                border: 1px solid rgba(25, 118, 210, 0.2);
                border-radius: 8px;
                padding: 10px;
                background-color: rgba(255, 255, 255, 0.9);
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 10pt;
                min-height: 300px;
            }
        """)
        code_layout.addWidget(code_text)
        
        # 下载按钮布局
        download_code_layout = QHBoxLayout()
        download_code_layout.setAlignment(Qt.AlignRight)
        
        # 代码下载按钮
        download_code_btn = QPushButton("📥 下载代码")
        download_code_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #2196f3, stop:1 #1976d2);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient( x1:0 y1:0, x2:1 y2:0,
                    stop:0 #64b5f6, stop:1 #42a5f5);
            }
        """)
        download_code_btn.clicked.connect(lambda: self.download_result("code"))
        download_code_layout.addWidget(download_code_btn)
        
        code_layout.addLayout(download_code_layout)
        
        # 将标签页添加到标签页控件
        result_tabs.addTab(code_tab, "生成的代码")
        
        # ========== 可视化结果标签页 ==========
        viz_tab = QWidget()
        viz_layout = QVBoxLayout(viz_tab)
        
        # 检查是否有figure对象
        figure = self.analysis_result.get("figure", None)
        if figure is not None:
            try:
                # 直接使用figure对象创建FigureCanvas
                canvas = FigureCanvas(figure)
                canvas.setMinimumSize(600, 400)
                
                viz_layout.addWidget(canvas)
            except Exception as e:
                # 如果显示失败，显示错误信息
                error_label = QLabel(f"可视化结果显示失败: {e}")
                error_label.setStyleSheet("color: #f44336; font-style: italic;")
                viz_layout.addWidget(error_label)
                print(f"可视化结果显示错误: {e}")
        else:
            # 检查是否有旧的plt对象（兼容旧版本）
            plt = self.analysis_result.get("plt", None)
            if plt is not None:
                try:
                    # 从plt对象获取figure
                    figure = plt.gcf()
                    canvas = FigureCanvas(figure)
                    canvas.setMinimumSize(600, 400)
                    viz_layout.addWidget(canvas)
                except Exception as e:
                    error_label = QLabel(f"可视化结果显示失败: {e}")
                    error_label.setStyleSheet("color: #f44336; font-style: italic;")
                    viz_layout.addWidget(error_label)
                    print(f"可视化结果显示错误: {e}")
            else:
                # 没有可视化结果
                no_viz_label = QLabel("本次分析未生成可视化结果")
                no_viz_label.setStyleSheet("color: #757575; font-style: italic;")
                viz_layout.addWidget(no_viz_label)
        
        # 将标签页添加到标签页控件
        result_tabs.addTab(viz_tab, "可视化结果")
        
        # 将标签页控件添加到结果布局
        self.result_layout.addWidget(result_tabs)
    
    def download_result(self, result_type):
        """
        下载分析结果
        
        参数:
            result_type: str, 结果类型，可选值: "summary" 或 "code"
        """
        from PyQt5.QtWidgets import QFileDialog
        
        if result_type == "summary":
            content = self.analysis_result["summary"]
            default_file_name = "analysis_result.txt"
            file_filter = "文本文件 (*.txt)"
        elif result_type == "code":
            content = self.analysis_result["code"]
            default_file_name = "analysis_code.py"
            file_filter = "Python文件 (*.py)"
        else:
            return
        
        # 打开文件对话框
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存文件", default_file_name, file_filter
        )
        
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                QMessageBox.information(self, "成功", f"{result_type == 'summary' and '分析结果' or '代码'}已成功保存到: {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"保存文件失败: {e}")
    
    def clear_layout(self, layout):
        while layout.count() > 0:
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

# 主应用入口
class ClinicalAnalysisApp:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.users_db = {"wmq1009": "12345"}  # 简单的用户数据库
        self.logged_in = False
        self.current_user = None
        self.api_key = None
        self.selected_model = None
    
    def run(self):
        # 显示登录对话框
        self.login_dialog = LoginDialog(self.users_db)
        self.login_dialog.login_success.connect(self.on_login_success)
        self.login_dialog.show()
        
        sys.exit(self.app.exec_())
    
    def on_login_success(self, username, password):
        self.current_user = username
        
        # 显示API配置对话框
        self.api_config_dialog = APIConfigDialog()
        self.api_config_dialog.config_success.connect(self.on_config_success)
        self.api_config_dialog.show()
    
    def on_config_success(self, api_key, selected_model):
        self.api_key = api_key
        self.selected_model = selected_model
        
        # 设置环境变量
        os.environ["OPENAI_API_KEY"] = self.api_key
        os.environ["OPENAI_MODEL"] = self.selected_model
        
        # 显示主应用窗口
        self.main_window = MainApplication(self.current_user)
        self.main_window.show()

# 运行应用
if __name__ == "__main__":
    app = ClinicalAnalysisApp()
    app.run()