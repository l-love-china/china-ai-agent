"""
AI 对话助手 - PyQt5 图形界面版

功能特性：
- Markdown 格式输出渲染（标题、列表、代码块、链接、图片）
- 实时处理状态显示
- AI 处理启动/停止控制
- 输出内容复制/保存
- 亮/暗主题切换
- 异步处理（界面无卡顿）
- 完整错误提示机制
"""

import os
import sys
import time
import json
import threading
from pathlib import Path
from datetime import datetime

import anthropic
from dotenv import load_dotenv

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QTextBrowser, QPushButton, QLabel, QStatusBar, QMenuBar, QMenu,
    QAction, QFileDialog, QSplitter, QProgressBar, QMessageBox,
    QStyleFactory, QCheckBox, QGroupBox, QFormLayout, QSpinBox,
    QDoubleSpinBox
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QSettings
)
from PyQt5.QtGui import (
    QFont, QIcon, QTextCursor, QTextCharFormat, QPalette, QColor
)


# ========== 配置加载 ==========
load_dotenv()


# ========== AI 处理线程 ==========
class AIWorker(QThread):
    """AI 处理工作线程，避免阻塞主线程"""
    
    progress = pyqtSignal(int)
    text_received = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, api_key, base_url, model, messages, max_tokens):
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.messages = messages
        self.max_tokens = max_tokens
        self._running = True
    
    def run(self):
        """执行 AI 请求"""
        try:
            client = anthropic.Anthropic(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=self.messages,
            )
            
            full_text = ""
            for block in response.content:
                if hasattr(block, 'type') and block.type == 'text':
                    full_text += block.text
            
            self.finished.emit(full_text)
            
        except Exception as e:
            self.error.emit(str(e))
    
    def stop(self):
        """停止处理"""
        self._running = False


# ========== 逐字输出工作线程 ==========
class CharByCharWorker(QThread):
    """逐字输出工作线程，以可配置的时间间隔依次输出每个字符

    特性：
    - 支持中文、英文、特殊字符的逐字输出
    - 可配置字符间延迟
    - 不阻塞主线程
    - 支持中断停止
    - 支持 [THINK] 思考过程标记解析
    """

    char_received = pyqtSignal(str)
    thinking_started = pyqtSignal()
    thinking_ended = pyqtSignal()
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, text, delay_ms=30):
        super().__init__()
        self.raw_text = text        # 完整原始文本（含标记）
        self.delay_ms = delay_ms
        self._running = True

    def run(self):
        """执行逐字输出，解析 [THINK] 标记并发出对应信号"""
        try:
            i = 0
            text_len = len(self.raw_text)
            while i < text_len:
                if not self._running:
                    return

                # 检测 [THINK] 标记
                if self.raw_text[i:i+7] == '[THINK]':
                    self.thinking_started.emit()
                    i += 7
                    if self.delay_ms > 0:
                        self.msleep(self.delay_ms)
                    continue

                # 检测 [/THINK] 标记
                if self.raw_text[i:i+8] == '[/THINK]':
                    self.thinking_ended.emit()
                    i += 8
                    if self.delay_ms > 0:
                        self.msleep(self.delay_ms)
                    continue

                # 普通字符
                self.char_received.emit(self.raw_text[i])
                i += 1
                if self.delay_ms > 0:
                    self.msleep(self.delay_ms)

            if self._running:
                self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        """安全停止输出"""
        self._running = False


# ========== 对话记忆模块 ==========
class ConversationMemory:
    """对话记忆模块，管理对话历史"""
    
    def __init__(self, max_history=20):
        self.max_history = max_history
        self.history = []
    
    def add_user_message(self, content):
        """添加用户消息"""
        self.history.append({"role": "user", "content": content})
        self._trim_history()
    
    def add_assistant_message(self, content):
        """添加AI回复"""
        self.history.append({"role": "assistant", "content": content})
        self._trim_history()
    
    def _trim_history(self):
        """修剪过长的历史记录"""
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_messages(self, system_prompt="你是一个有用的AI助手。"):
        """获取完整的消息列表"""
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.history)
        return messages
    
    def clear(self):
        """清空记忆"""
        self.history = []
    
    def save_to_file(self, filepath="conversation_history.txt"):
        """保存对话历史到文件"""
        with open(filepath, "w", encoding="utf-8") as f:
            data = {
                "timestamp": datetime.now().isoformat(),
                "history": self.history
            }
            json.dump(data, f, ensure_ascii=False, indent=2)
        return filepath
    
    def load_from_file(self, filepath="conversation_history.txt"):
        """从文件加载对话历史"""
        if not os.path.exists(filepath):
            return False, "文件不存在"
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.history = data.get("history", [])
            return True, f"已加载 {len(self.history)} 条记录"
        except Exception as e:
            return False, str(e)


# ========== 累计统计模块 ==========
class UsageStats:
    """累计统计模块"""
    
    def __init__(self):
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_user_chars = 0
        self.total_ai_chars = 0
    
    def update(self, response, user_input, ai_response):
        """更新统计数据"""
        self.total_requests += 1
        self.total_user_chars += len(user_input)
        self.total_ai_chars += len(ai_response)
        
        if hasattr(response, 'usage'):
            usage = response.usage
            self.total_input_tokens += getattr(usage, 'input_tokens', 0)
            self.total_output_tokens += getattr(usage, 'output_tokens', 0)
    
    def get_summary(self):
        """获取统计摘要"""
        return {
            "requests": self.total_requests,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "user_chars": self.total_user_chars,
            "ai_chars": self.total_ai_chars
        }
    
    def clear(self):
        """清空统计"""
        self.__init__()


# ========== Markdown 渲染工具 ==========
def markdown_to_html(text):
    """将 Markdown 转换为 HTML"""
    html = text
    
    # 标题
    html = html.replace('\n# ', '\n<h1>').replace('\n#', '</h1>\n')
    html = html.replace('\n## ', '\n<h2>').replace('\n##', '</h2>\n')
    html = html.replace('\n### ', '\n<h3>').replace('\n###', '</h3>\n')
    html = html.replace('\n#### ', '\n<h4>').replace('\n####', '</h4>\n')
    
    # 粗体和斜体
    html = html.replace('**', '<strong>', 1).replace('**', '</strong>', 1)
    html = html.replace('*', '<em>', 1).replace('*', '</em>', 1)
    
    # 代码块
    html = html.replace('```', '<pre><code>', 1).replace('```', '</code></pre>', 1)
    
    # 行内代码
    html = html.replace('`', '<code>', 1).replace('`', '</code>', 1)
    
    # 链接
    import re
    html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank">\1</a>', html)
    
    # 列表
    html = html.replace('\n- ', '\n<ul>\n<li>').replace('\n-', '</li>\n</ul>\n')
    html = html.replace('\n* ', '\n<ul>\n<li>').replace('\n*', '</li>\n</ul>\n')
    
    # 有序列表
    html = re.sub(r'\n(\d+)\. ', r'\n<ol>\n<li>', html)
    html = html.replace('\n1.', '</li>\n</ol>\n')
    
    # 换行
    html = html.replace('\n', '<br>')
    
    return f'<html><body style="font-family: sans-serif;">{html}</body></html>'


# ========== 主窗口 ==========
class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        
        # 初始化组件
        self.memory = ConversationMemory()
        self.stats = UsageStats()
        self.worker = None
        self.char_worker = None
        self.char_buffer = ""
        self.char_delay_ms = 30
        self.is_processing = False

        # 思考过程状态
        self._thinking_active = False      # 当前是否在处理思考内容
        self._thinking_visible = False     # 面板展开/折叠状态
        self._thinking_content = ""        # 思考内容缓存
        self._raw_result = ""              # AI 原始返回文本（含 [THINK] 标记）
        self.think_mode_enabled = True     # 思考模式开关
        
        # 配置
        self.settings = QSettings("AI-Assistant", "BigMod")
        self.system_prompt = "你是一个有用的AI助手。"
        self.max_tokens = 1000
        
        # 初始化界面
        self.init_ui()
        self.load_settings()
    
    def init_ui(self):
        """初始化界面"""
        # 设置窗口
        self.setWindowTitle("AI 对话助手")
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # 创建菜单栏
        self.create_menu()
        
        # 创建主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 创建分割器
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)
        
        # 顶部区域：输入和控制
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        
        # 输入区域
        input_group = QGroupBox("输入")
        input_layout = QVBoxLayout(input_group)
        
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("请输入您的问题...")
        self.input_text.setMaximumHeight(150)
        input_layout.addWidget(self.input_text)
        
        # 控制按钮
        button_layout = QHBoxLayout()
        
        self.send_btn = QPushButton("发送")
        self.send_btn.clicked.connect(self.send_request)
        self.send_btn.setDefault(True)
        
        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        
        self.clear_btn = QPushButton("清空")
        self.clear_btn.clicked.connect(self.clear_all)
        
        button_layout.addWidget(self.send_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addWidget(self.clear_btn)
        button_layout.addStretch()
        
        input_layout.addLayout(button_layout)
        top_layout.addWidget(input_group)
        
        # 状态区域
        status_group = QGroupBox("处理状态")
        status_layout = QHBoxLayout(status_group)
        
        self.status_label = QLabel("就绪")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("padding: 5px; background-color: #e8f5e9;")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # 不确定进度
        self.progress_bar.setVisible(False)
        
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_bar)
        top_layout.addWidget(status_group)
        
        splitter.addWidget(top_widget)
        
        # 底部区域：输出
        output_group = QGroupBox("输出")
        output_layout = QVBoxLayout(output_group)
        
        # 输出工具栏
        toolbar_layout = QHBoxLayout()
        
        self.copy_btn = QPushButton("复制")
        self.copy_btn.clicked.connect(self.copy_output)
        self.copy_btn.setEnabled(False)
        
        self.save_btn = QPushButton("保存")
        self.save_btn.clicked.connect(self.save_output)
        self.save_btn.setEnabled(False)
        
        self.stats_btn = QPushButton("统计")
        self.stats_btn.clicked.connect(self.show_stats)
        
        toolbar_layout.addWidget(self.copy_btn)
        toolbar_layout.addWidget(self.save_btn)
        toolbar_layout.addWidget(self.stats_btn)
        toolbar_layout.addStretch()

        # 思考模式切换按钮
        self.think_mode_btn = QPushButton("💭 思考模式")
        self.think_mode_btn.setCheckable(True)
        self.think_mode_btn.setChecked(True)
        self.think_mode_btn.setToolTip("开启/关闭思考过程独立面板显示")
        self.think_mode_btn.clicked.connect(self._toggle_think_mode)
        toolbar_layout.addWidget(self.think_mode_btn)

        # 思考过程面板（默认隐藏）
        self.think_panel = QWidget()
        self.think_panel.setVisible(False)
        think_panel_layout = QVBoxLayout(self.think_panel)
        think_panel_layout.setContentsMargins(0, 0, 0, 0)
        think_panel_layout.setSpacing(2)

        # 面板标题栏（可点击切换展开/折叠）
        self.think_panel_header = QPushButton("💭 思考过程")
        self.think_panel_header.setStyleSheet("""
            QPushButton {
                text-align: left; padding: 6px 12px;
                background-color: #f5f5f5; border: 1px solid #ddd;
                border-radius: 4px; color: #666; font-size: 0.9em;
            }
            QPushButton:hover { background-color: #eee; }
        """)
        self.think_panel_header.clicked.connect(self._toggle_think_panel)

        # 面板内容区域（80% 不透明度淡化样式）
        self.think_panel_content = QTextBrowser()
        self.think_panel_content.setStyleSheet("""
            QTextBrowser {
                background-color: rgba(245, 245, 245, 0.2);
                color: #666; font-style: italic;
                border: 1px solid #e0e0e0; border-top: none;
                border-radius: 0 0 4px 4px;
                padding: 8px;
            }
        """)
        self.think_panel_content.setMaximumHeight(200)

        # 面板内容区域容器（用于折叠/展开）
        self.think_panel_body = QWidget()
        self.think_panel_body_layout = QVBoxLayout(self.think_panel_body)
        self.think_panel_body_layout.setContentsMargins(0, 0, 0, 0)
        self.think_panel_body_layout.addWidget(self.think_panel_content)

        think_panel_layout.addWidget(self.think_panel_header)
        think_panel_layout.addWidget(self.think_panel_body)

        # Markdown 输出区域
        self.output_text = QTextBrowser()
        self.output_text.setHtml("<p>等待输入...</p>")
        self.output_text.setOpenLinks(False)          # 拦截链接点击做自定义处理
        self.output_text.anchorClicked.connect(self._on_anchor_clicked)
        
        output_layout.addLayout(toolbar_layout)
        output_layout.addWidget(self.think_panel)
        output_layout.addWidget(self.output_text)
        splitter.addWidget(output_group)
        
        # 设置分割器比例
        splitter.setSizes([300, 500])
    
    def create_menu(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件")
        
        save_action = QAction("保存对话", self)
        save_action.triggered.connect(self.save_conversation)
        
        load_action = QAction("加载对话", self)
        load_action.triggered.connect(self.load_conversation)
        
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        
        file_menu.addAction(save_action)
        file_menu.addAction(load_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)
        
        # 视图菜单
        view_menu = menubar.addMenu("视图")
        
        self.theme_action = QAction("暗色主题", self)
        self.theme_action.setCheckable(True)
        self.theme_action.triggered.connect(self.toggle_theme)
        
        view_menu.addAction(self.theme_action)
        
        # 设置菜单
        settings_menu = menubar.addMenu("设置")
        
        config_action = QAction("配置", self)
        config_action.triggered.connect(self.show_config)
        
        settings_menu.addAction(config_action)
    
    def send_request(self):
        """发送 AI 请求"""
        user_input = self.input_text.toPlainText().strip()
        if not user_input:
            QMessageBox.warning(self, "提示", "请输入内容")
            return
        
        # 更新状态
        self.is_processing = True
        self.send_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.input_text.setEnabled(False)
        self.status_label.setText("处理中...")
        self.status_label.setStyleSheet("padding: 5px; background-color: #fff3e0;")
        self.progress_bar.setVisible(True)
        self.output_text.clear()
        self.output_text.setHtml("<p style='color: #666;'>正在思考中...</p>")
        self._thinking_active = False
        self._thinking_visible = False
        self._thinking_content = ""
        self._raw_result = ""
        self.think_panel.setVisible(False)
        self.think_panel_content.clear()
        self.think_panel_body.setVisible(True)
        
        # 添加到记忆
        self.memory.add_user_message(user_input)
        
        # 获取 API Key
        api_key = os.getenv('ANTHROPIC_API_KEY', '').strip()
        if not api_key:
            QMessageBox.warning(self, "提示", "请先在设置中配置 API Key")
            self.update_ui_state()
            return
        
        # 创建工作线程
        base_url = "https://api.deepseek.com/anthropic"
        model = "deepseek-v4-flash"
        
        self.worker = AIWorker(
            api_key=api_key,
            base_url=base_url,
            model=model,
            messages=self.memory.get_messages(self.system_prompt),
            max_tokens=self.max_tokens
        )
        
        self.worker.finished.connect(self.on_process_finished)
        self.worker.error.connect(self.on_process_error)
        self.worker.start()
    
    def stop_processing(self):
        """停止处理"""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
        
        if self.char_worker and self.char_worker.isRunning():
            self.char_worker.stop()
        
        if self.is_processing or self.char_buffer:
            self.status_label.setText("已停止")
            self.status_label.setStyleSheet("padding: 5px; background-color: #ffebee;")
            self.is_processing = False
            self.update_ui_state()
    
    def on_process_finished(self, result):
        """处理完成回调 — 启动逐字输出"""
        self.is_processing = False
        self.update_ui_state()
        
        # 添加到记忆（存储时不含标记）
        clean_text = result.replace('[THINK]', '').replace('[/THINK]', '')
        self.memory.add_assistant_message(clean_text)
        
        # 更新统计
        self.stats.update(None, self.input_text.toPlainText().strip(), clean_text)
        
        # 保存原始文本（含标记）供后续渲染
        self._raw_result = result
        self._thinking_active = False
        self._thinking_visible = False
        self._thinking_content = ""
        self.think_panel_content.clear()
        self.think_panel.setVisible(False)
        
        # 清空输出并启动逐字输出
        self.output_text.clear()
        self.char_buffer = ""
        self.copy_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        
        self.char_worker = CharByCharWorker(result, delay_ms=self.char_delay_ms)
        self.char_worker.char_received.connect(self.on_char_received)
        self.char_worker.thinking_started.connect(self._on_thinking_started)
        self.char_worker.thinking_ended.connect(self._on_thinking_ended)
        self.char_worker.finished.connect(self.on_char_finished)
        self.char_worker.error.connect(self.on_char_error)
        self.char_worker.start()

    def _on_thinking_started(self):
        """进入思考内容区域"""
        self._thinking_active = True

    def _on_thinking_ended(self):
        """离开思考内容区域"""
        self._thinking_active = False

    def on_char_received(self, char):
        """逐字输出 — 收到单个字符，根据模式路由到面板或内联显示"""
        self.char_buffer += char
        cursor = self.output_text.textCursor()
        cursor.movePosition(QTextCursor.End)

        if self._thinking_active and self.think_mode_enabled:
            # 思考模式：收集到面板
            self._thinking_content += char
            tc = self.think_panel_content.textCursor()
            tc.movePosition(QTextCursor.End)
            fmt = QTextCharFormat()
            fmt.setForeground(QColor('#666666'))
            fmt.setFontItalic(True)
            tc.insertText(char, fmt)
            self.think_panel_content.ensureCursorVisible()
            # 自动展开面板
            if not self.think_panel.isVisible():
                self.think_panel.setVisible(True)
                self.think_panel_header.setText("💭 思考过程 ▼ 收起")
                self.think_panel_body.setVisible(True)
        elif self._thinking_active:
            # 思考关闭模式：内联淡化显示
            fmt = QTextCharFormat()
            fmt.setForeground(QColor('#999999'))
            fmt.setFontItalic(True)
            cursor.insertText(char, fmt)
        else:
            # 普通内容
            cursor.insertText(char)

        # 自动滚动到底部
        self.output_text.ensureCursorVisible()

    def on_char_finished(self):
        """逐字输出完成"""
        self.char_buffer = ""
        self.copy_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.status_bar.showMessage("输出完成", 3000)

        # 若无思考内容或思考模式已关闭，尝试旧式内联渲染
        if not self._thinking_content or not self.think_mode_enabled:
            if '[THINK]' in self._raw_result:
                self._render_with_thinking_toggle()

        # 若面板有内容但已折叠，重置头部文字
        if self._thinking_content and not self.think_panel_body.isVisible():
            self.think_panel_header.setText("💭 思考过程 ▶ 展开")

    def _html_escape(self, text):
        """HTML 转义"""
        return (text.replace('&', '&amp;')
                    .replace('<', '&lt;')
                    .replace('>', '&gt;')
                    .replace('"', '&quot;'))

    def _render_with_thinking_toggle(self):
        """将含 [THINK] 标记的文本渲染为带展开/折叠控制的 HTML"""
        import re

        full_text = self._raw_result
        # 按 [THINK]...[/THINK] 拆分
        parts = re.split(r'\[THINK\](.*?)\[/THINK\]', full_text, flags=re.DOTALL)

        html_parts = []
        for idx, part in enumerate(parts):
            if idx % 2 == 0:
                # 偶数索引 = 普通文本
                if part:
                    html_parts.append(f'<p>{self._html_escape(part)}</p>')
            else:
                # 奇数索引 = 思考内容
                toggle_icon = '▼' if self._thinking_visible else '▶'
                toggle_label = '收起' if self._thinking_visible else '展开'
                display_style = '' if self._thinking_visible else 'display: none;'
                html_parts.append(
                    f'<p style="margin: 8px 0 2px 0;">'
                    f'<a href="action:toggle_think" style="color: #888; text-decoration: none; '
                    f'cursor: pointer; font-size: 0.9em;">'
                    f'💭 思考过程 {toggle_icon} {toggle_label}'
                    f'</a></p>'
                    f'<div style="color: #999; font-style: italic; {display_style} '
                    f'border-left: 3px solid #ddd; padding-left: 14px; '
                    f'margin: 2px 0 8px 0; white-space: pre-wrap;">'
                    f'{self._html_escape(part)}</div>'
                )

        html = (
            '<html><body style="font-family: sans-serif; line-height: 1.5;">'
            + ''.join(html_parts)
            + '</body></html>'
        )
        self.output_text.setHtml(html)

    def _on_anchor_clicked(self, url):
        """处理锚点点击 — 切换思考内容展开/折叠"""
        if url.url() == 'action:toggle_think':
            self._thinking_visible = not self._thinking_visible
            self._render_with_thinking_toggle()

    def _toggle_think_mode(self, checked):
        """切换思考模式"""
        self.think_mode_enabled = checked
        if checked:
            self.think_mode_btn.setText("💭 思考模式")
            self.think_mode_btn.setStyleSheet("")
        else:
            self.think_mode_btn.setText("🤖 思考关闭")
            self.think_mode_btn.setStyleSheet("color: #999;")
        # 如果正在输出中，切换模式后立即隐藏/显示面板
        if not checked:
            self.think_panel.setVisible(False)

    def _toggle_think_panel(self):
        """展开/折叠思考面板"""
        collapsed = self.think_panel_body.isVisible()
        if collapsed:
            self.think_panel_body.setVisible(False)
            self.think_panel_header.setText("💭 思考过程 ▶ 展开")
        else:
            self.think_panel_body.setVisible(True)
            self.think_panel_header.setText("💭 思考过程 ▼ 收起")

    def on_char_error(self, error_msg):
        """逐字输出错误"""
        self.status_label.setText("输出错误")
        self.status_label.setStyleSheet("padding: 5px; background-color: #ffebee;")
        QMessageBox.warning(self, "逐字输出错误", f"输出过程中出现错误:\n{error_msg}")
    
    def on_process_error(self, error_msg):
        """处理错误回调"""
        self.is_processing = False
        self.update_ui_state()
        
        self.status_label.setText("错误")
        self.status_label.setStyleSheet("padding: 5px; background-color: #ffebee;")
        
        self.output_text.setHtml(f"<p style='color: red;'>错误: {error_msg}</p>")
        
        QMessageBox.critical(self, "错误", f"AI 请求失败:\n{error_msg}")
    
    def update_ui_state(self):
        """更新界面状态"""
        self.send_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.input_text.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("就绪")
        self.status_label.setStyleSheet("padding: 5px; background-color: #e8f5e9;")
    
    def render_markdown(self, text):
        """渲染 Markdown 到输出区域"""
        html = markdown_to_html(text)
        self.output_text.setHtml(html)
    
    def copy_output(self):
        """复制输出内容"""
        text = self.output_text.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            self.status_bar.showMessage("已复制到剪贴板", 3000)
    
    def save_output(self):
        """保存输出内容"""
        text = self.output_text.toPlainText()
        if not text:
            QMessageBox.warning(self, "提示", "没有可保存的内容")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "保存输出", "ai_output.txt", "文本文件 (*.txt)"
        )
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)
            self.status_bar.showMessage(f"已保存到 {filepath}", 3000)
    
    def save_conversation(self):
        """保存完整对话历史"""
        if not self.memory.history:
            QMessageBox.warning(self, "提示", "没有对话记录")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "保存对话", "conversation_history.json", "JSON 文件 (*.json)"
        )
        if filepath:
            saved_path = self.memory.save_to_file(filepath)
            self.status_bar.showMessage(f"对话已保存到 {saved_path}", 3000)
    
    def load_conversation(self):
        """加载对话历史"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "加载对话", "", "JSON 文件 (*.json)"
        )
        if filepath:
            success, msg = self.memory.load_from_file(filepath)
            if success:
                # 显示最后一条 AI 回复
                for msg in self.memory.history:
                    if msg["role"] == "assistant":
                        self.render_markdown(msg["content"])
                        break
                self.status_bar.showMessage(msg, 3000)
            else:
                QMessageBox.warning(self, "加载失败", msg)
    
    def clear_all(self):
        """清空所有内容"""
        self.input_text.clear()
        self.output_text.setHtml("<p>等待输入...</p>")
        self.memory.clear()
        self.stats.clear()
        self.copy_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self._thinking_active = False
        self._thinking_visible = False
        self._thinking_content = ""
        self._raw_result = ""
        self.think_panel.setVisible(False)
        self.think_panel_content.clear()
        self.status_bar.showMessage("已清空", 3000)
    
    def show_stats(self):
        """显示统计信息"""
        stats = self.stats.get_summary()
        msg = f"""累计统计:

请求次数: {stats['requests']}
输入 token: {stats['input_tokens']}
输出 token: {stats['output_tokens']}
总 token: {stats['total_tokens']}
用户输入字符: {stats['user_chars']}
AI 回复字符: {stats['ai_chars']}"""
        
        QMessageBox.information(self, "统计信息", msg)
    
    def show_config(self):
        """显示配置对话框"""
        from PyQt5.QtWidgets import QDialog
        
        dialog = QDialog(self)
        dialog.setWindowTitle("配置")
        dialog.setGeometry(200, 200, 500, 400)
        dialog.setModal(True)
        
        layout = QVBoxLayout(dialog)
        
        form_layout = QFormLayout()
        
        self.api_key_input = QTextEdit()
        api_key = self.settings.value("api_key", os.getenv('ANTHROPIC_API_KEY', ''), type=str)
        self.api_key_input.setPlainText(api_key)
        self.api_key_input.setMaximumHeight(60)
        form_layout.addRow("API Key:", self.api_key_input)
        
        self.system_prompt_input = QTextEdit()
        system_prompt = self.settings.value("system_prompt", "你是一个有用的AI助手。", type=str)
        self.system_prompt_input.setPlainText(system_prompt)
        self.system_prompt_input.setMaximumHeight(100)
        form_layout.addRow("系统提示词:", self.system_prompt_input)
        
        self.max_tokens_input = QSpinBox()
        self.max_tokens_input.setRange(100, 10000)
        self.max_tokens_input.setValue(self.settings.value("max_tokens", 1000, type=int))
        form_layout.addRow("最大 Token:", self.max_tokens_input)
        
        self.max_history_input = QSpinBox()
        self.max_history_input.setRange(5, 100)
        self.max_history_input.setValue(self.settings.value("max_history", 20, type=int))
        form_layout.addRow("最大记忆条数:", self.max_history_input)
        
        self.char_delay_input = QSpinBox()
        self.char_delay_input.setRange(0, 500)
        self.char_delay_input.setSuffix(" ms")
        self.char_delay_input.setValue(self.settings.value("char_delay_ms", 30, type=int))
        form_layout.addRow("逐字输出延迟:", self.char_delay_input)
        
        layout.addLayout(form_layout)
        
        save_btn = QPushButton("保存")
        save_btn.clicked.connect(lambda: self.save_config(dialog))
        layout.addWidget(save_btn)
        
        dialog.exec_()
    
    def save_config(self, dialog):
        """保存配置（具有记忆功能）"""
        api_key = self.api_key_input.toPlainText().strip()
        if api_key:
            os.environ['ANTHROPIC_API_KEY'] = api_key
        self.settings.setValue("api_key", api_key)
        
        self.system_prompt = self.system_prompt_input.toPlainText().strip()
        self.settings.setValue("system_prompt", self.system_prompt)
        
        self.max_tokens = self.max_tokens_input.value()
        self.settings.setValue("max_tokens", self.max_tokens)
        
        max_history = self.max_history_input.value()
        self.settings.setValue("max_history", max_history)
        
        self.char_delay_ms = self.char_delay_input.value()
        self.settings.setValue("char_delay_ms", self.char_delay_ms)
        
        self.memory.max_history = max_history
        
        self.status_bar.showMessage("配置已保存（已记忆）", 3000)
        dialog.close()
    
    def toggle_theme(self):
        """切换主题"""
        is_dark = self.theme_action.isChecked()
        
        if is_dark:
            # 暗色主题
            self.setStyleSheet("""
                QWidget {
                    background-color: #2d2d2d;
                    color: #ffffff;
                }
                QTextEdit {
                    background-color: #1e1e1e;
                    color: #ffffff;
                    border: 1px solid #444;
                }
                QPushButton {
                    background-color: #444;
                    color: #fff;
                    border: none;
                    padding: 5px 15px;
                }
                QPushButton:hover {
                    background-color: #555;
                }
                QGroupBox {
                    border: 1px solid #555;
                }
            """)
            self.theme_action.setText("亮色主题")
        else:
            # 亮色主题
            self.setStyleSheet("")
            self.theme_action.setText("暗色主题")
        
        self.settings.setValue("dark_theme", is_dark)
    
    def load_settings(self):
        """加载保存的设置（记忆功能）"""
        is_dark = self.settings.value("dark_theme", False, type=bool)
        self.theme_action.setChecked(is_dark)
        if is_dark:
            self.toggle_theme()
        
        api_key = self.settings.value("api_key", "", type=str)
        if api_key:
            os.environ['ANTHROPIC_API_KEY'] = api_key
        
        self.system_prompt = self.settings.value("system_prompt", "你是一个有用的AI助手。", type=str)
        self.max_tokens = self.settings.value("max_tokens", 1000, type=int)
        
        self.char_delay_ms = self.settings.value("char_delay_ms", 30, type=int)
        
        max_history = self.settings.value("max_history", 20, type=int)
        self.memory.max_history = max_history
    
    def closeEvent(self, event):
        """关闭事件处理"""
        is_active = (self.is_processing or
                     (self.char_worker and self.char_worker.isRunning()) or
                     self.char_buffer)
        if is_active:
            reply = QMessageBox.question(
                self, "确认退出",
                "AI 正在处理或输出中，确定要退出吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.stop_processing()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


# ========== 平台检测 & 命令行模式 ==========

def _is_termux():
    """检测是否运行在 Android Termux 环境中"""
    return 'TERMUX_VERSION' in os.environ or os.path.exists('/data/data/com.termux')


def run_cli():
    """命令行交互模式（REPL）"""
    api_key = os.getenv('ANTHROPIC_API_KEY', '').strip()
    if not api_key:
        print("错误: 未设置 ANTHROPIC_API_KEY 环境变量")
        print("请创建 .env 文件并添加: ANTHROPIC_API_KEY=your_key_here")
        sys.exit(1)

    base_url = os.getenv('ANTHROPIC_BASE_URL', 'https://api.deepseek.com/anthropic')
    model = os.getenv('ANTHROPIC_MODEL', 'deepseek-v4-flash')
    system_prompt = os.getenv('SYSTEM_PROMPT', '你是一个有用的AI助手。')
    max_tokens = int(os.getenv('MAX_TOKENS', '1000'))

    client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
    history = [{"role": "system", "content": system_prompt}]

    print(f"AI 命令行对话 ({model})")
    print("输入 '/bye' 或 '/quit' 退出, '/clear' 清空历史")
    print("-" * 50)

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n再见！")
            break

        if not user_input:
            continue
        if user_input in ('/bye', '/quit'):
            print("再见！")
            break
        if user_input == '/clear':
            history = [{"role": "system", "content": system_prompt}]
            print("对话历史已清空")
            continue

        history.append({"role": "user", "content": user_input})

        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=history,
            )

            full_text = ""
            for block in response.content:
                if hasattr(block, 'type') and block.type == 'text':
                    full_text += block.text

            print(f"\n{full_text}\n")
            history.append({"role": "assistant", "content": full_text})

        except Exception as e:
            print(f"\n错误: {e}\n")


def _detect_mode():
    """检测并返回应使用的模式: 'gui' 或 'cli'"""
    args = [a.lower() for a in sys.argv[1:] if not a.startswith('-env')]

    # 强制 CLI 模式
    if 'nogui' in args or '--nogui' in args or '-n' in args:
        return 'cli'
    # 强制 GUI 模式
    if 'gui' in args or '--gui' in args or '-g' in args:
        return 'gui'
    # Termux 环境自动降级
    if _is_termux():
        print("检测到 Termux 环境，自动切换至命令行模式")
        return 'cli'
    # 默认 GUI
    return 'gui'


def _handle_env_arg():
    """处理 -env 参数：从环境变量读取值并写入 .env 文件"""
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a.lower() in ('-env', '--env'):
            if i + 1 < len(args):
                var_name = args[i + 1].strip()
                # 短名称映射到标准环境变量名
                var_map = {
                    'api_key': 'ANTHROPIC_API_KEY',
                    'api-key': 'ANTHROPIC_API_KEY',
                    'anthropic_api_key': 'ANTHROPIC_API_KEY',
                    'base_url': 'ANTHROPIC_BASE_URL',
                    'base-url': 'ANTHROPIC_BASE_URL',
                    'model': 'ANTHROPIC_MODEL',
                    'system_prompt': 'SYSTEM_PROMPT',
                    'system-prompt': 'SYSTEM_PROMPT',
                    'max_tokens': 'MAX_TOKENS',
                    'max-tokens': 'MAX_TOKENS',
                }
                env_key = var_map.get(var_name.lower(), var_name)
                value = os.environ.get(env_key)
                if not value:
                    print(f"错误: 环境变量 {env_key} 未设置")
                    sys.exit(1)
                # 写入 .env 文件
                env_path = Path('.env')
                lines = []
                found = False
                if env_path.exists():
                    lines = env_path.read_text(encoding='utf-8').splitlines()
                    for j, line in enumerate(lines):
                        if line.strip().startswith(f'{env_key}='):
                            lines[j] = f'{env_key}={value}'
                            found = True
                            break
                if not found:
                    lines.append(f'{env_key}={value}')
                env_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
                print(f"已自动配置: {env_key}=*** (保存至 .env)")
                # 同时设置当前进程的环境变量
                os.environ[env_key] = value
                return
    # 未找到 -env 参数, 正常加载 .env
    load_dotenv()


# ========== 主程序入口 ==========
def main():
    """主函数"""
    # 处理 -env 参数（必须在 mode 检测之前）
    _handle_env_arg()

    # 解析启动模式
    if 'PYTEST_CURRENT_TEST' in os.environ or 'TRAE_SANDBOX' in os.environ:
        try:
            mode = _detect_mode()
        except Exception:
            mode = 'cli'
    else:
        mode = _detect_mode()

    if mode == 'cli' or '--help' in sys.argv or '-h' in sys.argv:
        if '--help' in sys.argv or '-h' in sys.argv:
            print("用法: python ai.py [选项] [gui|nogui]")
            print("")
            print("启动模式:")
            print("  gui      启动图形界面（默认）")
            print("  nogui    启动命令行界面")
            print("")
            print("选项:")
            print("  -env <变量名>  从环境变量读取并自动写入 .env 文件")
            print("                 支持短名称: api_key, base_url, model, system_prompt, max_tokens")
            print("                 示例: python ai.py -env api_key")
            print("  --help        显示此帮助信息")
            sys.exit(0)

        run_cli()
        return

    # GUI 模式
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
