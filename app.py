from ultralytics import YOLO
import sys
import os
import cv2
import time
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QMessageBox, QGroupBox, QCheckBox,
                             QProgressBar, QTextEdit, QFrame, QSplitter,
                             QSizePolicy, QGridLayout, QScrollArea, QSlider,
                             QStackedWidget, QGraphicsDropShadowEffect)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread,  QSharedMemory
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QIcon
from PIL import Image



# --- MD3 风格配置 ---
class MD3Styles:
    # MD3 调色板 (基于 Teal/Blue 方案)
    PRIMARY = "#00668A"  # 主色
    ON_PRIMARY = "#FFFFFF"  # 主色上的文本
    PRIMARY_CONTAINER = "#C2E8FF"  # 主容器色
    ON_PRIMARY_CONTAINER = "#001E2C"

    SECONDARY = "#4D616C"
    SECONDARY_CONTAINER = "#D0E6F2"

    BACKGROUND = "#F8FDFF"  # 背景色
    SURFACE = "#FFFFFF"  # 卡片表面色
    SURFACE_VARIANT = "#DDE3EA"  # 边框/分割线/占位符背景
    OUTLINE = "#74777F"

    ERROR = "#BA1A1A"

    FONT_FAMILY = "Segoe UI, Microsoft YaHei, sans-serif"

    @staticmethod
    def get_stylesheet():
        return f"""
            QMainWindow {{
                background-color: {MD3Styles.BACKGROUND};
            }}
            QWidget {{
                font-family: "{MD3Styles.FONT_FAMILY}";
                font-size: 14px;
                color: #191C1E;
            }}
            /* 滚动区域 */
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}

            /* --- 自定义滚动条样式 (重点修改) --- */
            QScrollBar:vertical {{
                border: none;
                background: {MD3Styles.SURFACE_VARIANT};
                width: 10px;
                border-radius: 5px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {MD3Styles.SECONDARY};
                min-height: 20px;
                border-radius: 5px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {MD3Styles.PRIMARY};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: none;
            }}

            /* 标签 */
            QLabel {{
                color: #191C1E;
            }}
            QLabel#TitleLabel {{
                font-size: 24px;
                font-weight: bold;
                color: {MD3Styles.PRIMARY};
                margin-bottom: 10px;
            }}
            QLabel#HeaderLabel {{
                font-size: 18px;
                font-weight: 600;
                color: #191C1E;
            }}

            /* 组框 (作为卡片使用) */
            QGroupBox {{
                background-color: {MD3Styles.SURFACE};
                border: 1px solid {MD3Styles.SURFACE_VARIANT};
                border-radius: 16px;
                margin-top: 1em;
                padding-top: 10px;
                font-weight: bold;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
                color: {MD3Styles.PRIMARY};
            }}

            /* 进度条 */
            QProgressBar {{
                border: none;
                background-color: {MD3Styles.SURFACE_VARIANT};
                border-radius: 4px;
                height: 8px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {MD3Styles.PRIMARY};
                border-radius: 4px;
            }}

            /* 复选框 */
            QCheckBox {{
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border-radius: 2px;
                border: 2px solid {MD3Styles.SECONDARY};
            }}
            QCheckBox::indicator:checked {{
                background-color: {MD3Styles.PRIMARY};
                border-color: {MD3Styles.PRIMARY};
            }}

            /* 滑动条 */
            QSlider::groove:horizontal {{
                border: none;
                height: 6px;
                background: {MD3Styles.SURFACE_VARIANT};
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {MD3Styles.PRIMARY};
                border: 2px solid {MD3Styles.BACKGROUND};
                width: 18px;
                height: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }}

            /* 文本框 */
            QTextEdit {{
                background-color: {MD3Styles.SURFACE};
                border: 1px solid {MD3Styles.SURFACE_VARIANT};
                border-radius: 12px;
                padding: 8px;
                color: #444;
                selection-background-color: {MD3Styles.PRIMARY_CONTAINER};
            }}

            /* 卡片帧 */
            QFrame#ContentCard {{
                background-color: {MD3Styles.SURFACE};
                border: 1px solid {MD3Styles.SURFACE_VARIANT};
                border-radius: 24px;
            }}

            /* 视图帧 (用于显示图像的容器) */
            QFrame#ViewFrame {{
                background-color: #E8EDF2; /* 浅灰色背景代替纯黑 */
                border: 1px solid {MD3Styles.SURFACE_VARIANT};
                border-radius: 16px;
            }}

            QFrame#StatsCard {{
                background-color: {MD3Styles.SECONDARY_CONTAINER};
                border-radius: 12px;
            }}
        """


class VideoThread(QThread):
    """视频处理线程"""
    frame_processed = pyqtSignal(np.ndarray, int, int)
    finished = pyqtSignal()

    def __init__(self, video_path, model, save_video=False, conf_threshold=0.4, output_dir="output"):
        super().__init__()
        self.video_path = video_path
        self.model = model
        self.save_video = save_video
        self.conf_threshold = conf_threshold
        self.output_dir = output_dir
        self.running = True
        self.current_frame = None
        self._pause = False
        self.video_writer = None  # 视频写入器

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.finished.emit()
            return

        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        current_frame = 0

        # 如果启用了保存视频，初始化视频写入器
        if self.save_video:
            # 确保输出目录存在
            os.makedirs(self.output_dir, exist_ok=True)

            # 生成输出文件名（基于输入视频名和时间戳）
            input_name = os.path.splitext(os.path.basename(self.video_path))[0]
            timestamp = int(time.time())
            output_path = os.path.join(self.output_dir, f"{input_name}_detected_{timestamp}.mp4")

            # 初始化视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者使用 'avc1'
            self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            print(f"视频保存路径: {output_path}")

        while self.running and current_frame < frame_count:
            if self._pause:
                self.msleep(100)
                continue

            ret, frame = cap.read()
            if not ret:
                break

            try:
                results = self.model(frame, conf=self.conf_threshold)
                annotated_frame = results[0].plot()
                self.current_frame = annotated_frame.copy()

                # 如果启用了保存视频，将处理后的帧写入文件
                if self.save_video and self.video_writer is not None:
                    self.video_writer.write(annotated_frame)

                self.frame_processed.emit(annotated_frame, current_frame, frame_count)
                current_frame += 1

                delay = max(1, int(1000 / fps) - 10)
                self.msleep(delay)

            except Exception as e:
                print(f"视频处理错误: {e}")
                break

        # 释放资源
        cap.release()
        if self.video_writer is not None:
            self.video_writer.release()
            print("视频写入器已释放")

        self.finished.emit()

    def stop(self):
        self.running = False

    def pause(self):
        self._pause = True

    def resume(self):
        self._pause = False


class CameraThread(QThread):
    """摄像头线程"""
    frame_processed = pyqtSignal(np.ndarray)

    def __init__(self, camera_id, model, conf_threshold=0.4, save_video=False, output_dir="output"):
        super().__init__()
        self.camera_id = camera_id
        self.model = model
        self.conf_threshold = conf_threshold
        self.save_video = save_video
        self.output_dir = output_dir
        self.running = True
        self.video_writer = None
        self.recording_start_time = None

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # 获取摄像头帧率和尺寸
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 20.0  # 默认帧率

        # 初始化视频写入器（如果需要录制）
        if self.save_video:
            self.initialize_video_writer(cap, fps)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, conf=self.conf_threshold)
            annotated_frame = results[0].plot()

            # 如果启用了录制，写入帧
            if self.save_video and self.video_writer is not None:
                self.video_writer.write(annotated_frame)

            self.frame_processed.emit(annotated_frame)

        # 释放资源
        cap.release()
        if self.video_writer is not None:
            self.video_writer.release()

    def initialize_video_writer(self, cap, fps):
        """初始化视频写入器"""
        try:
            # 确保输出目录存在
            os.makedirs(self.output_dir, exist_ok=True)

            # 获取帧尺寸
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 生成输出文件名
            timestamp = int(time.time())
            output_path = os.path.join(self.output_dir, f"camera_recording_{timestamp}.mp4")

            # 初始化视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者 'XVID' for AVI
            self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            self.recording_start_time = time.time()
            print(f"开始录制摄像头视频: {output_path}")

        except Exception as e:
            print(f"初始化视频写入器失败: {e}")
            self.video_writer = None

    def stop(self):
        self.running = False
        if self.video_writer is not None:
            self.video_writer.release()
            recording_duration = time.time() - self.recording_start_time
            print(f"录制结束，时长: {recording_duration:.1f}秒")


class StyledButton(QPushButton):
    """MD3 风格按钮"""

    def __init__(self, text, icon_name=None, btn_type="Tonal", small=False):
        super().__init__(text)
        self.setCursor(Qt.PointingHandCursor)

        height = 32 if small else 40
        self.setMinimumHeight(height)

        # 字体设置
        font = QFont(MD3Styles.FONT_FAMILY)
        font.setWeight(QFont.DemiBold if btn_type in ["Filled", "Tonal"] else QFont.Normal)
        self.setFont(font)

        # 基础样式
        base_style = f"""
            QPushButton {{
                border-radius: {height // 2}px;
                padding: 0 20px;
                border: none;
            }}
        """

        if btn_type == "Filled" or (isinstance(btn_type, bool) and btn_type):
            style = base_style + f"""
                QPushButton {{
                    background-color: {MD3Styles.PRIMARY};
                    color: {MD3Styles.ON_PRIMARY};
                }}
                QPushButton:hover {{
                    background-color: #005472;
                }}
                QPushButton:pressed {{
                    background-color: #00425A;
                }}
                QPushButton:disabled {{
                    background-color: rgba(28, 27, 31, 0.12);
                    color: rgba(28, 27, 31, 0.38);
                }}
            """
        elif btn_type == "Outlined":
            style = base_style + f"""
                QPushButton {{
                    background-color: transparent;
                    border: 1px solid #79747E;
                    color: {MD3Styles.PRIMARY};
                }}
                QPushButton:hover {{
                    background-color: rgba(0, 102, 138, 0.08);
                }}
            """
        else:  # Tonal (Default)
            style = base_style + f"""
                QPushButton {{
                    background-color: {MD3Styles.SECONDARY_CONTAINER};
                    color: {MD3Styles.ON_PRIMARY_CONTAINER};
                }}
                QPushButton:hover {{
                    background-color: #C2E2F0; 
                }}
                QPushButton:pressed {{
                    background-color: #B0D4E4;
                }}
                QPushButton:disabled {{
                    background-color: rgba(28, 27, 31, 0.12);
                    color: rgba(28, 27, 31, 0.38);
                }}
            """

        self.setStyleSheet(style)
        if icon_name:
            self.setIcon(QIcon(f":/{icon_name}"))


class ClassDetailDialog(QMessageBox):
    def __init__(self, class_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("类别详情")
        self.setIcon(QMessageBox.Information)
        class_list = "\n".join([f"{i + 1}. {name}" for i, name in enumerate(class_names)])
        self.setText(f"模型可识别以下 {len(class_names)} 种类别：\n\n{class_list}")
        self.setStandardButtons(QMessageBox.Ok)


# --- 页面组件 ---

class MainMenuPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 40, 20, 20)
        layout.setAlignment(Qt.AlignTop)

        title = QLabel("海洋鱼类识别")
        title.setObjectName("TitleLabel")
        title.setAlignment(Qt.AlignCenter)

        subtitle = QLabel("智能视觉识别系统")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #555; font-size: 14px; margin-bottom: 30px;")

        layout.addWidget(title)
        layout.addWidget(subtitle)

        menu_container = QFrame()
        menu_container.setObjectName("ContentCard")
        menu_layout = QVBoxLayout(menu_container)
        menu_layout.setSpacing(15)
        menu_layout.setContentsMargins(30, 30, 30, 30)

        btn_image = StyledButton("🖼️  图片识别", btn_type="Filled")
        btn_image.setFixedHeight(50)
        btn_image.clicked.connect(lambda: self.parent.show_page("image"))

        btn_video = StyledButton("🎥  视频识别", btn_type="Filled")
        btn_video.setFixedHeight(50)
        btn_video.clicked.connect(lambda: self.parent.show_page("video"))

        btn_camera = StyledButton("📹  摄像头识别", btn_type="Filled")
        btn_camera.setFixedHeight(50)
        btn_camera.clicked.connect(lambda: self.parent.show_page("camera"))

        btn_settings = StyledButton("⚙️  系统设置", btn_type="Tonal")
        btn_settings.clicked.connect(lambda: self.parent.show_page("settings"))

        menu_layout.addWidget(btn_image)
        menu_layout.addWidget(btn_video)
        menu_layout.addWidget(btn_camera)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"color: {MD3Styles.SURFACE_VARIANT};")
        menu_layout.addWidget(line)

        menu_layout.addWidget(btn_settings)

        layout.addWidget(menu_container)
        self.setLayout(layout)


class ImageDetectionPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.image_files = []
        self.current_image_index = 0
        self.current_image_name = ""  # 新增：当前图片文件名
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        header = QHBoxLayout()
        title = QLabel("图片识别")
        title.setObjectName("HeaderLabel")
        back_btn = StyledButton("返回", btn_type="Text", small=True)
        back_btn.clicked.connect(lambda: self.parent.show_page("main"))
        header.addWidget(title)
        header.addStretch()
        header.addWidget(back_btn)
        layout.addLayout(header)

        controls = QGroupBox("操作面板")
        ctrl_layout = QVBoxLayout()

        self.image_btn = StyledButton("选择图片文件", btn_type="Tonal")
        self.image_btn.clicked.connect(self.select_images)

        self.detect_btn = StyledButton("开始检测", btn_type="Filled")
        self.detect_btn.clicked.connect(self.detect_images)
        self.detect_btn.setEnabled(False)

        action_row = QHBoxLayout()
        self.save_btn = StyledButton("保存结果", btn_type="Outlined", small=True)
        self.save_btn.clicked.connect(self.save_current_result)
        self.save_btn.setEnabled(False)
        action_row.addWidget(self.save_btn)

        ctrl_layout.addWidget(self.image_btn)
        ctrl_layout.addWidget(self.detect_btn)
        ctrl_layout.addLayout(action_row)
        controls.setLayout(ctrl_layout)
        layout.addWidget(controls)

        nav_layout = QHBoxLayout()
        self.prev_btn = StyledButton("◀", btn_type="Tonal", small=True)
        self.prev_btn.clicked.connect(self.previous_image)
        self.prev_btn.setEnabled(False)
        self.next_btn = StyledButton("▶", btn_type="Tonal", small=True)
        self.next_btn.clicked.connect(self.next_image)
        self.next_btn.setEnabled(False)

        nav_layout.addWidget(self.prev_btn)
        self.status_label = QLabel("未选择图片")
        self.status_label.setAlignment(Qt.AlignCenter)
        nav_layout.addWidget(self.status_label)
        nav_layout.addWidget(self.next_btn)

        layout.addLayout(nav_layout)

        # 新增：文件名显示标签
        self.filename_label = QLabel("")
        self.filename_label.setAlignment(Qt.AlignCenter)
        self.filename_label.setStyleSheet(f"color: {MD3Styles.SECONDARY}; font-size: 12px; margin-top: 5px;")
        layout.addWidget(self.filename_label)

        layout.addStretch()
        self.setLayout(layout)

    def select_images(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "选择图片", "", "Images (*.jpg *.png *.webp *.bmp)")
        if file_paths:
            self.image_files = file_paths
            self.current_image_index = 0
            self.load_current_image()
            self.detect_btn.setEnabled(True)
            self.prev_btn.setEnabled(len(file_paths) > 1)
            self.next_btn.setEnabled(len(file_paths) > 1)
            self.parent.log_message(f"📁 已选择 {len(file_paths)} 张图片")

    def load_current_image(self):
        if self.image_files:
            file_path = self.image_files[self.current_image_index]
            self.current_image = self.parent.read_image(file_path)
            if self.current_image is not None:
                self.current_image_name = os.path.basename(file_path)  # 保存文件名
                self.parent.display_image(self.current_image)
                self.update_display_info()  # 更新显示信息
            else:
                self.parent.log_message(f"❌ 读取失败: {os.path.basename(file_path)}")

    def update_display_info(self):
        """更新显示信息（序号和文件名）"""
        if self.image_files:
            base_name = os.path.basename(self.image_files[self.current_image_index])
            self.status_label.setText(f"{self.current_image_index + 1}/{len(self.image_files)}")
            self.filename_label.setText(f"文件: {base_name}")

            # 在主窗口显示区域也显示文件名
            if hasattr(self.parent, 'display_caption'):
                self.parent.display_caption.setText(f"图片检测: {base_name}")

    def detect_images(self):
        print(f"DEBUG: 开始检测，当前窗口ID: {id(self.parent)}")

        if not hasattr(self, 'parent') or self.parent is None:
            print("ERROR: 没有找到父窗口!")
            return

        if not self.image_files or self.parent.model is None:
            print("DEBUG: 条件不满足，直接返回")
            return

        try:
            self.detect_btn.setEnabled(False)
            self.detect_btn.setText("分析中...")

            # 更新显示状态
            base_name = os.path.basename(self.image_files[self.current_image_index])
            self.status_label.setText(f"分析中: {base_name}")

            QApplication.processEvents()

            print("DEBUG: 开始模型推理 - 准备调用 model()")
            results = self.parent.model(self.current_image, conf=self.parent.conf_threshold)
            print("DEBUG: 模型推理完成")

            annotated_frame = results[0].plot()
            print("DEBUG: 结果绘图完成")

            self.current_result = annotated_frame
            self.parent.display_image(annotated_frame)
            self.save_btn.setEnabled(True)

            boxes = results[0].boxes
            count = len(boxes) if boxes else 0
            self.parent.update_stats(count, annotated_frame.shape[:2])

            # 更新显示完成状态
            self.update_display_info()
            self.parent.log_message(f"🔍 检测完成: {base_name} - 发现 {count} 个目标")

        except Exception as e:
            print(f"DEBUG: 检测异常: {e}")
            import traceback
            traceback.print_exc()
            self.parent.log_message(f"❌ 检测失败: {e}")
        finally:
            self.detect_btn.setEnabled(True)
            self.detect_btn.setText("开始检测")

    def save_current_result(self):
        if hasattr(self, 'current_result'):
            # 使用原文件名加上结果标记
            base_name = os.path.splitext(self.current_image_name)[0]
            path = os.path.join(self.parent.output_dir, f"{base_name}_result_{int(time.time())}.jpg")
            cv2.imwrite(path, self.current_result)
            self.parent.log_message(f"💾 保存至: {os.path.basename(path)}")

    def previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()
            self.detect_images()

    def next_image(self):
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_current_image()
            self.detect_images()


class VideoDetectionPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.video_name = ""  # 新增：视频文件名
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        header = QHBoxLayout()
        title = QLabel("视频识别")
        title.setObjectName("HeaderLabel")
        back_btn = StyledButton("返回", btn_type="Text", small=True)
        back_btn.clicked.connect(lambda: self.parent.show_page("main"))
        header.addWidget(title)
        header.addStretch()
        header.addWidget(back_btn)
        layout.addLayout(header)

        controls = QGroupBox("视频控制")
        ctrl_layout = QVBoxLayout()

        self.video_btn = StyledButton("导入视频", btn_type="Tonal")
        self.video_btn.clicked.connect(self.select_video)

        self.detect_btn = StyledButton("开始分析", btn_type="Filled")
        self.detect_btn.clicked.connect(self.detect_video)
        self.detect_btn.setEnabled(False)

        play_ctrls = QHBoxLayout()
        self.pause_btn = StyledButton("暂停", btn_type="Outlined", small=True)
        self.pause_btn.clicked.connect(self.toggle_video_pause)
        self.pause_btn.setEnabled(False)

        self.save_frame_btn = StyledButton("抓拍", btn_type="Outlined", small=True)
        self.save_frame_btn.clicked.connect(self.save_video_frame)
        self.save_frame_btn.setEnabled(False)

        play_ctrls.addWidget(self.pause_btn)
        play_ctrls.addWidget(self.save_frame_btn)

        ctrl_layout.addWidget(self.video_btn)
        ctrl_layout.addWidget(self.detect_btn)
        ctrl_layout.addLayout(play_ctrls)
        controls.setLayout(ctrl_layout)
        layout.addWidget(controls)

        self.save_video_checkbox = QCheckBox("导出检测视频")
        layout.addWidget(self.save_video_checkbox)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("等待导入视频...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #666;")
        layout.addWidget(self.status_label)

        # 新增：视频信息显示
        self.video_info_label = QLabel("")
        self.video_info_label.setAlignment(Qt.AlignCenter)
        self.video_info_label.setStyleSheet(f"color: {MD3Styles.SECONDARY}; font-size: 12px;")
        layout.addWidget(self.video_info_label)

        layout.addStretch()
        self.setLayout(layout)

    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Video (*.mp4 *.avi *.mkv)")
        if path:
            self.video_path = path
            self.video_name = os.path.basename(path)  # 保存文件名
            self.detect_btn.setEnabled(True)
            self.status_label.setText(f"已选择: {self.video_name}")

            # 显示视频信息
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                minutes = int(duration // 60)
                seconds = int(duration % 60)
                self.video_info_label.setText(
                    f"时长: {minutes:02d}:{seconds:02d} | 帧数: {frame_count} | FPS: {fps:.1f}")
                cap.release()

            # 在主窗口显示区域显示视频名
            if hasattr(self.parent, 'display_caption'):
                self.parent.display_caption.setText(f"视频检测: {self.video_name}")

            self.parent.log_message(f"📁 载入视频: {self.video_name}")

    def detect_video(self):
        if not hasattr(self, 'video_path') or self.parent.model is None:
            self.parent.log_message("❌ 请先选择视频文件")
            return

        try:
            self.video_thread = VideoThread(
                self.video_path,
                self.parent.model,
                self.save_video_checkbox.isChecked(),
                self.parent.conf_threshold,
                self.parent.output_dir
            )
            self.video_thread.frame_processed.connect(self.update_frame)
            self.video_thread.finished.connect(self.video_finished)
            self.video_thread.start()

            # 设置按钮状态
            self.pause_btn.setEnabled(True)
            self.pause_btn.setText("暂停")
            self.save_frame_btn.setEnabled(True)
            self.detect_btn.setEnabled(False)

            # 更新显示状态
            self.status_label.setText(f"分析中: {self.video_name}")
            if self.save_video_checkbox.isChecked():
                self.parent.log_message(f"🎬 开始视频分析: {self.video_name} (将导出检测视频)")
            else:
                self.parent.log_message(f"🎬 开始视频分析: {self.video_name}")

        except Exception as e:
            self.parent.log_message(f"❌ 启动视频分析失败: {e}")

    def toggle_video_pause(self):
        if hasattr(self, 'video_thread') and self.video_thread.isRunning():
            if self.pause_btn.text() == "暂停":
                # 暂停视频
                self.video_thread.pause()
                self.pause_btn.setText("继续")
                self.parent.log_message(f"⏸️ 视频已暂停: {self.video_name}")
            else:
                # 继续视频
                self.video_thread.resume()
                self.pause_btn.setText("暂停")
                self.parent.log_message(f"▶️ 视频继续播放: {self.video_name}")
        else:
            self.parent.log_message("⚠️ 视频线程未运行")

    def update_frame(self, frame, current, total):
        try:
            self.current_video_frame = frame
            self.parent.display_image(frame)
            self.progress_bar.setValue(int((current / total) * 100))

            # 在主窗口显示当前视频信息和进度
            if hasattr(self.parent, 'display_caption'):
                progress = (current / total) * 100
                self.parent.display_caption.setText(f"视频检测: {self.video_name} - {progress:.1f}%")

        except Exception as e:
            print(f"更新帧错误: {e}")

    def video_finished(self):
        try:
            self.pause_btn.setEnabled(False)
            self.pause_btn.setText("暂停")
            self.save_frame_btn.setEnabled(False)
            self.detect_btn.setEnabled(True)
            self.status_label.setText(f"播放结束: {self.video_name}")
            self.progress_bar.setValue(100)

            # 更新主窗口显示
            if hasattr(self.parent, 'display_caption'):
                self.parent.display_caption.setText(f"视频检测完成: {self.video_name}")

            if hasattr(self.video_thread, 'save_video') and self.video_thread.save_video:
                self.parent.log_message(f"✅ 视频分析完成: {self.video_name}，检测视频已保存")
            else:
                self.parent.log_message(f"✅ 视频分析完成: {self.video_name}")

        except Exception as e:
            print(f"视频结束处理错误: {e}")

    def save_video_frame(self):
        if hasattr(self, 'current_video_frame'):
            try:
                # 使用视频名作为前缀
                base_name = os.path.splitext(self.video_name)[0]
                path = os.path.join(self.parent.output_dir, f"{base_name}_frame_{int(time.time())}.jpg")
                cv2.imwrite(path, self.current_video_frame)
                self.parent.log_message(f"📷 抓拍成功: {os.path.basename(path)}")
            except Exception as e:
                self.parent.log_message(f"❌ 保存帧失败: {e}")


class CameraDetectionPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        header = QHBoxLayout()
        title = QLabel("实时监控")
        title.setObjectName("HeaderLabel")
        back_btn = StyledButton("返回", btn_type="Text", small=True)
        back_btn.clicked.connect(lambda: self.parent.show_page("main"))
        header.addWidget(title)
        header.addStretch()
        header.addWidget(back_btn)
        layout.addLayout(header)

        controls = QGroupBox("设备控制")
        ctrl_layout = QVBoxLayout()

        self.start_btn = StyledButton("启动摄像头", btn_type="Filled")
        self.start_btn.clicked.connect(self.start_camera)

        self.stop_btn = StyledButton("停止", btn_type="Tonal")
        self.stop_btn.clicked.connect(self.stop_camera)
        self.stop_btn.setEnabled(False)

        self.save_btn = StyledButton("抓拍当前帧", btn_type="Outlined")
        self.save_btn.clicked.connect(self.save_camera_frame)
        self.save_btn.setEnabled(False)

        ctrl_layout.addWidget(self.start_btn)
        ctrl_layout.addWidget(self.stop_btn)
        ctrl_layout.addWidget(self.save_btn)
        controls.setLayout(ctrl_layout)
        layout.addWidget(controls)

        # 修改：添加录制状态标签
        self.save_check = QCheckBox("自动录制")
        self.recording_status = QLabel("")
        self.recording_status.setStyleSheet("color: #d32f2f; font-weight: bold;")

        ctrl_layout.addWidget(self.save_check)
        ctrl_layout.addWidget(self.recording_status)

        self.status_label = QLabel("设备就绪")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        layout.addStretch()
        self.setLayout(layout)

    def start_camera(self):

        try:
            # 获取录制状态
            save_video = self.save_check.isChecked()

            self.camera_thread = CameraThread(
                0,
                self.parent.model,
                self.parent.conf_threshold,
                save_video,
                self.parent.output_dir
            )
            self.camera_thread.frame_processed.connect(self.update_frame)
            self.camera_thread.start()

            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.status_label.setText("监控运行中...")

            # 更新录制状态显示
            if save_video:
                self.recording_start_time = time.time()
                self.recording_timer = QTimer()
                self.recording_timer.timeout.connect(self.update_recording_time)
                self.recording_timer.start(1000)  # 每秒更新一次
            else:
                self.recording_status.setText("")  # 清空状态
                self.parent.log_message("📹 摄像头已启动")

            self.parent.camera_start_time = time.time()
            self.parent.camera_frame_count = 0
            self.parent.fps_timer = QTimer()
            self.parent.fps_timer.timeout.connect(self.parent.update_fps)
            self.parent.fps_timer.start(1000)

        except Exception as e:
            self.parent.log_message(f"❌ 启动失败: {e}")

    def update_recording_time(self):
        if hasattr(self, 'recording_start_time'):
            elapsed = int(time.time() - self.recording_start_time)
            minutes = elapsed // 60
            seconds = elapsed % 60
            self.recording_status.setText(f"● 录制中... {minutes:02d}:{seconds:02d}")

    def update_frame(self, frame):
        self.current_frame = frame
        self.parent.display_image(frame)
        self.parent.camera_frame_count += 1

    def stop_camera(self):
        if hasattr(self, 'camera_thread') and self.camera_thread:
            # 检查是否正在录制
            was_recording = self.camera_thread.save_video

            self.camera_thread.stop()
            self.camera_thread.wait()

            if hasattr(self.parent, 'fps_timer'):
                self.recording_timer.stop()

            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            self.recording_status.setText("")  # 清空录制状态
            self.status_label.setText("已停止")

            # 记录录制完成信息
            if was_recording:
                self.parent.log_message("✅ 摄像头录制已完成并保存")
            else:
                self.parent.log_message("📹 摄像头已停止")

    def save_camera_frame(self):
        if hasattr(self, 'current_frame'):
            try:
                path = os.path.join(self.parent.output_dir, f"cam_{int(time.time())}.jpg")
                cv2.imwrite(path, self.current_frame)
                self.parent.log_message(f"📷 抓拍成功: {path}")
            except Exception as e:
                self.parent.log_message(f"❌ 保存帧失败: {e}")


class SettingsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        header = QHBoxLayout()
        title = QLabel("检测设置")
        title.setObjectName("HeaderLabel")
        back_btn = StyledButton("返回", btn_type="Text", small=True)
        back_btn.clicked.connect(lambda: self.parent.show_page("main"))
        header.addWidget(title)
        header.addStretch()
        header.addWidget(back_btn)
        layout.addLayout(header)

        conf_group = QGroupBox("识别灵敏度")
        conf_layout = QVBoxLayout()

        slider_layout = QHBoxLayout()
        self.conf_val = QLabel("0.40")
        self.conf_val.setStyleSheet(f"color: {MD3Styles.PRIMARY}; font-weight: bold;")

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(10, 90)
        self.slider.setValue(40)
        self.slider.valueChanged.connect(self.update_conf)

        slider_layout.addWidget(QLabel("低"))
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(QLabel("高"))
        slider_layout.addWidget(self.conf_val)

        conf_layout.addLayout(slider_layout)
        conf_layout.addWidget(QLabel("较高的阈值可以减少误报，但可能漏检目标。"))
        conf_group.setLayout(conf_layout)
        layout.addWidget(conf_group)

        info_group = QGroupBox("模型状态")
        info_layout = QVBoxLayout()
        self.model_status = QLabel("未加载")
        self.model_detail_btn = StyledButton("查看类别列表", btn_type="Outlined", small=True)
        self.model_detail_btn.clicked.connect(self.show_details)
        self.model_detail_btn.setEnabled(False)

        info_layout.addWidget(self.model_status)
        info_layout.addWidget(self.model_detail_btn)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        layout.addStretch()
        self.setLayout(layout)

    def update_conf(self, val):
        conf = val / 100.0
        self.parent.conf_threshold = conf
        self.conf_val.setText(f"{conf:.2f}")
        self.parent.conf_display.setText(f"阈值: {conf:.2f}")

    def show_details(self):
        if self.parent.class_names:
            ClassDetailDialog(self.parent.class_names, self.parent).exec_()


class FishDetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        # 检查是否已有实例运行
        self.shared_memory = QSharedMemory("FishDetectionApp")
        if self.shared_memory.attach():
            print("ERROR: 应用程序已在运行中!")
            sys.exit(1)

        if not self.shared_memory.create(1):
            print("ERROR: 无法创建共享内存!")
            sys.exit(1)

        print(f"DEBUG: 创建主窗口，ID: {id(self)}")


        self.model = None
        self.video_thread = None
        self.camera_thread = None
        self.output_dir = "output"
        self.class_names = []
        self.conf_threshold = 0.4
        os.makedirs(self.output_dir, exist_ok=True)

        self.setStyleSheet(MD3Styles.get_stylesheet())
        self.init_ui()
        self.load_model()

    def init_ui(self):
        self.setWindowTitle("海洋鱼类智能识别系统")
        self.resize(1100, 750)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # 1. 侧边栏
        sidebar = QFrame()
        sidebar.setObjectName("ContentCard")
        sidebar.setFixedWidth(320)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 20))
        shadow.setOffset(0, 4)
        sidebar.setGraphicsEffect(shadow)

        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)

        self.control_stack = QStackedWidget()
        self.main_menu = MainMenuPage(self)
        self.image_page = ImageDetectionPage(self)
        self.video_page = VideoDetectionPage(self)
        self.camera_page = CameraDetectionPage(self)
        self.settings_page = SettingsPage(self)

        self.control_stack.addWidget(self.main_menu)
        self.control_stack.addWidget(self.image_page)
        self.control_stack.addWidget(self.video_page)
        self.control_stack.addWidget(self.camera_page)
        self.control_stack.addWidget(self.settings_page)

        sidebar_layout.addWidget(self.control_stack)

        # 2. 主显示区
        display_area = QWidget()
        display_layout = QVBoxLayout(display_area)
        display_layout.setContentsMargins(0, 0, 0, 0)
        display_layout.setSpacing(15)

        # 视图窗口 (修改点：使用 ViewFrame 样式和浅色背景)
        view_frame = QFrame()
        view_frame.setObjectName("ViewFrame")  # 使用新定义的 ViewFrame ID

        view_shadow = QGraphicsDropShadowEffect()
        view_shadow.setBlurRadius(20)
        view_shadow.setColor(QColor(0, 0, 0, 15))  # 阴影稍微淡一点
        view_shadow.setOffset(0, 4)
        view_frame.setGraphicsEffect(view_shadow)

        view_layout = QVBoxLayout(view_frame)
        view_layout.setContentsMargins(2, 2, 2, 2)

        # 修改初始显示的文字和样式
        self.display_label = QLabel("暂无检测内容\n\n请从左侧菜单选择模式")
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet(f"color: {MD3Styles.SECONDARY}; font-size: 16px; font-weight: bold;")
        self.display_label.setMinimumSize(640, 480)

        view_layout.addWidget(self.display_label)
        display_layout.addWidget(view_frame, stretch=1)

        # 信息面板
        info_panel = QHBoxLayout()

        stats_frame = QFrame()
        stats_frame.setObjectName("StatsCard")
        stats_frame.setFixedWidth(250)
        stats_layout = QVBoxLayout(stats_frame)

        self.fps_label = QLabel("FPS: --")
        self.count_label = QLabel("目标数: 0")
        self.conf_display = QLabel(f"阈值: {self.conf_threshold:.2f}")

        for lbl in [self.fps_label, self.count_label, self.conf_display]:
            lbl.setStyleSheet(f"font-weight: bold; color: {MD3Styles.ON_PRIMARY_CONTAINER};")
            stats_layout.addWidget(lbl)

        info_panel.addWidget(stats_frame)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText("系统准备就绪...")
        self.log_text.setMaximumHeight(120)
        info_panel.addWidget(self.log_text)

        display_layout.addLayout(info_panel)

        main_layout.addWidget(sidebar)
        main_layout.addWidget(display_area)

    def show_page(self, name):
        widgets = {
            "main": self.main_menu, "image": self.image_page,
            "video": self.video_page, "camera": self.camera_page,
            "settings": self.settings_page
        }
        if name in widgets:
            self.control_stack.setCurrentWidget(widgets[name])
            if name == "settings" and self.model:
                self.settings_page.model_status.setText("✅ 模型已加载")
                self.settings_page.model_detail_btn.setEnabled(True)

    def load_model(self):
        try:
            # 判断是否是打包后的环境
            if getattr(sys, 'frozen', False):
                # 打包后，模型文件应该在exe同目录下
                base_path = os.path.dirname(sys.executable)
                model_path = os.path.join(base_path, "best.pt")
            else:
                # 开发环境中直接使用当前目录
                model_path = "best.pt"

            if not os.path.exists(model_path):
                self.log_message("⚠️ 模型文件未找到，请确保best.pt与本程序在同一目录下")
                return

            self.model = YOLO(model_path)
            self.class_names = list(self.model.names.values())
            self.log_message("🎉 系统初始化完成，模型加载成功")
        except Exception as e:
            self.log_message(f"❌ 模型错误: {e}")

    def log_message(self, msg):
        t = time.strftime("%H:%M:%S")
        self.log_text.append(f"<span style='color:#666'>[{t}]</span> {msg}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def read_image(self, path):
        if path.lower().endswith('.webp'):
            try:
                return cv2.cvtColor(np.array(Image.open(path).convert('RGB')), cv2.COLOR_RGB2BGR)
            except:
                return None
        return cv2.imread(path)

    def display_image(self, img):
        if img is None: return
        h, w, ch = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.display_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_stats(self, count, res):
        self.count_label.setText(f"目标数: {count}")

    def update_fps(self):
        if hasattr(self, 'camera_start_time'):
            elapsed = time.time() - self.camera_start_time
            fps = self.camera_frame_count / elapsed if elapsed > 0 else 0
            self.fps_label.setText(f"FPS: {fps:.1f}")
            self.camera_frame_count = 0
            self.camera_start_time = time.time()

    def closeEvent(self, event):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = FishDetectionGUI()
    window.show()
    sys.exit(app.exec_())