import sys
import os
import cv2
import numpy as np
import pandas as pd
import math
import tempfile
import shutil
import subprocess
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QListWidget, QTableWidget, QTableWidgetItem,
                             QGroupBox, QFileDialog, QMessageBox, QComboBox, QSpinBox,
                             QDoubleSpinBox, QProgressBar, QCheckBox, QTextEdit, QSlider,
                             QDialog, QDialogButtonBox, QFormLayout, QLineEdit)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QPoint, QEventLoop
from PyQt6.QtGui import QImage, QPixmap, QColor, QPainter

# optional: try import pyproj for utm<->lonlat conversion
try:
    from pyproj import CRS, Transformer
    _HAS_PYPROJ = True
except Exception:
    _HAS_PYPROJ = False

# try import yolo processor (your project may provide yolo_processor.py)
try:
    from yolo_processor import YOLOProcessor, YOLOProcessorThread
except Exception:
    # basic stubs for development if yolo_processor not available
    class YOLOProcessor(object):
        frame_processed = pyqtSignal(object, object) if hasattr(pyqtSignal, '__call__') else None
        progress_updated = pyqtSignal(int, int, int) if hasattr(pyqtSignal, '__call__') else None
        fps_updated = pyqtSignal(float) if hasattr(pyqtSignal, '__call__') else None
        error_occurred = pyqtSignal(str) if hasattr(pyqtSignal, '__call__') else None
        processing_finished = pyqtSignal()
        detection_info_updated = pyqtSignal(str)
        model_loaded = pyqtSignal(str)

        def __init__(self):
            pass

        def load_model(self, path):
            return False

        def set_tracking_enabled(self, b): pass
        def set_target_fps(self, f): pass
        def stop_processing(self): pass
        def pause_processing(self): pass
        def resume_processing(self): pass
        def set_export_options(self, *args, **kwargs): pass

    class YOLOProcessorThread(QThread):
        def __init__(self, proc):
            super().__init__()
            self.proc = proc
        def set_video_path(self, p): pass
        def run(self): pass

# ----------------- VideoLabel: 用于在 QLabel 显示可缩放/平移图像并映射到像素坐标 -------------
class VideoLabel(QLabel):
    clicked = pyqtSignal(int, int)  # 原始图像像素坐标（int,int）

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackgroundRole(self.backgroundRole())
        self.setSizePolicy(self.sizePolicy())
        self.setMinimumSize(200, 150)
        self._original_pixmap = None   # QPixmap 原始帧
        self._zoom = 1.0               # 缩放因子（相对于原始像素）
        self._offset = QPoint(0, 0)    # 平移偏移（像素）
        self._dragging = False
        self._last_mouse_pos = QPoint(0, 0)
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: black;")

    def set_frame(self, frame_bgr):
        """传入 OpenCV 的 BGR frame（numpy array），内部转换为 QPixmap 保存为原始图像"""
        if frame_bgr is None:
            self._original_pixmap = None
            self.update()
            return

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        self._original_pixmap = QPixmap.fromImage(qt_image)
        # 不自动 reset_view，保留用户当前缩放/偏移体验
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor('black'))
        if not self._original_pixmap:
            return

        # 计算用于绘制的缩放后的 pixmap
        target_w = int(self._original_pixmap.width() * self._zoom)
        target_h = int(self._original_pixmap.height() * self._zoom)
        if target_w <= 0 or target_h <= 0:
            return

        scaled_pix = self._original_pixmap.scaled(
            target_w, target_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        # 居中显示并加上偏移量
        x = (self.width() - scaled_pix.width()) // 2 + self._offset.x()
        y = (self.height() - scaled_pix.height()) // 2 + self._offset.y()
        painter.drawPixmap(x, y, scaled_pix)

    def wheelEvent(self, event):
        angle = event.angleDelta().y()
        factor = 1.1 if angle > 0 else (1.0 / 1.1)
        old_zoom = self._zoom
        new_zoom = max(0.1, min(4.0, old_zoom * factor))

        pos = event.position().toPoint()
        # 把鼠标在控件坐标映射到图像坐标（在旧缩放下）
        img_x_before, img_y_before = self._map_label_to_image_coords(pos)
        self._zoom = new_zoom
        img_x_after, img_y_after = self._map_label_to_image_coords(pos)

        # 调整偏移以尽可能保持鼠标位置不动（经验性调整）
        if img_x_before is not None and img_x_after is not None:
            self._offset -= QPoint(int((img_x_after - img_x_before) * self._zoom),
                                   int((img_y_after - img_y_before) * self._zoom))

        self._zoom = new_zoom
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            img_coords = self._map_label_to_image_coords(event.pos())
            if img_coords[0] is not None:
                self.clicked.emit(int(img_coords[0]), int(img_coords[1]))
        elif event.button() == Qt.MouseButton.MiddleButton:
            self._dragging = True
            self._last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self._dragging:
            delta = event.pos() - self._last_mouse_pos
            self._last_mouse_pos = event.pos()
            self._offset += delta
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._dragging = False

    def _map_label_to_image_coords(self, pos):
        """把 QLabel 内的坐标 pos(QPoint) 映射回原始图像坐标 (float x, float y)
           如果点击在图像外返回 (None, None)
        """
        if not self._original_pixmap:
            return (None, None)

        scaled_w = int(self._original_pixmap.width() * self._zoom)
        scaled_h = int(self._original_pixmap.height() * self._zoom)
        x0 = (self.width() - scaled_w) // 2 + self._offset.x()
        y0 = (self.height() - scaled_h) // 2 + self._offset.y()

        px = pos.x() - x0
        py = pos.y() - y0
        if 0 <= px < scaled_w and 0 <= py < scaled_h:
            img_x = px / self._zoom
            img_y = py / self._zoom
            return (img_x, img_y)
        else:
            return (None, None)

    def reset_view(self):
        self._zoom = 1.0
        self._offset = QPoint(0, 0)
        self.update()

    def set_zoom(self, zoom_value):
        self._zoom = max(0.1, min(4.0, zoom_value))
        self.update()

    def get_zoom(self):
        return self._zoom

# 设置原点坐标的对话框
class OriginDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置原点坐标")
        self.setModal(True)
        layout = QFormLayout(self)
        self.lon_input = QLineEdit()
        self.lat_input = QLineEdit()
        layout.addRow("经度:", self.lon_input)
        layout.addRow("纬度:", self.lat_input)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_coords(self):
        try:
            lon = float(self.lon_input.text())
            lat = float(self.lat_input.text())
            return lon, lat
        except ValueError:
            return None, None

# ----------------- 主窗口 -----------------
class VehicleTrajectoryAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("车辆轨迹与运动分析系统")
        self.setGeometry(100, 100, 1400, 800)

        # 初始化变量
        self.video_path = None
        self.cap = None
        self.current_frame = None
        self.homography_matrix = None
        self.origin_lonlat = None
        self.origin_pixel = None
        self.utm_zone = None
        self.is_northern = True
        self.vehicle_tracks = {}
        self.selected_vehicle_id = None
        self.pending_missing_ids = []

        # 视频 fps（由 load_video 读取）
        self.video_fps = None

        # 模型相关
        self.model_folder = None

        # YOLO处理器
        self.yolo_processor = YOLOProcessor()
        self.yolo_thread = None

        # 初始化UI
        self.init_ui()
        self.connect_signals()

        self.play_preview_btn.clicked.connect(self._on_preview_play)
        self.stop_preview_btn.clicked.connect(self._on_preview_stop)

        # 预览定时器
        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self._preview_next_frame)

        self.next_track_id = 100000


    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        self.video_label = VideoLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        left_layout.addWidget(self.video_label)

        zoom_layout = QHBoxLayout()
        self.zoom_out_btn = QPushButton("-")
        self.zoom_in_btn = QPushButton("+")
        self.fit_btn = QPushButton("适应窗口")
        zoom_layout.addWidget(self.zoom_out_btn)
        zoom_layout.addWidget(self.zoom_in_btn)
        zoom_layout.addWidget(self.fit_btn)

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(10, 400)  # 百分比
        self.zoom_slider.setValue(100)
        zoom_layout.addWidget(self.zoom_slider)

        left_layout.addLayout(zoom_layout)

        control_layout = QHBoxLayout()
        self.load_video_btn = QPushButton("加载视频")
        control_layout.addWidget(self.load_video_btn)

        self.load_model_btn = QPushButton("选择模型文件夹")
        control_layout.addWidget(self.load_model_btn)

        self.model_combo = QComboBox()
        self.model_combo.setEnabled(False)
        control_layout.addWidget(self.model_combo)

        self.refresh_models_btn = QPushButton("刷新模型列表")
        self.refresh_models_btn.setEnabled(False)
        control_layout.addWidget(self.refresh_models_btn)

        self.load_selected_model_btn = QPushButton("加载选中模型")
        self.load_selected_model_btn.setEnabled(False)
        control_layout.addWidget(self.load_selected_model_btn)

        self.start_btn = QPushButton("开始处理")
        self.start_btn.setEnabled(False)
        control_layout.addWidget(self.start_btn)

        self.play_preview_btn = QPushButton("播放预览")
        control_layout.addWidget(self.play_preview_btn)
        self.stop_preview_btn = QPushButton("停止预览")
        self.stop_preview_btn.setEnabled(False)
        control_layout.addWidget(self.stop_preview_btn)

        self.stop_btn = QPushButton("停止处理")
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)

        self.pause_btn = QPushButton("暂停")
        self.pause_btn.setEnabled(False)
        control_layout.addWidget(self.pause_btn)  

        left_layout.addLayout(control_layout)

        self.progress_bar = QProgressBar()
        left_layout.addWidget(self.progress_bar)

        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(100)
        left_layout.addWidget(self.status_text)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        homography_group = QGroupBox("单应矩阵设置")
        homography_layout = QVBoxLayout(homography_group)

        self.homography_input = QTextEdit()
        self.homography_input.setPlaceholderText("粘贴单应矩阵内容（9个数值，空格或换行分隔）")
        self.homography_input.setMaximumHeight(100)
        homography_layout.addWidget(self.homography_input)

        load_h_btn_layout = QHBoxLayout()
        self.load_h_file_btn = QPushButton("从文件加载")
        load_h_btn_layout.addWidget(self.load_h_file_btn)

        self.apply_h_btn = QPushButton("应用矩阵")
        load_h_btn_layout.addWidget(self.apply_h_btn)

        homography_layout.addLayout(load_h_btn_layout)

        origin_layout = QHBoxLayout()
        origin_layout.addWidget(QLabel("原点经度:"))
        self.origin_lon_input = QDoubleSpinBox()
        self.origin_lon_input.setRange(-180, 180)
        self.origin_lon_input.setDecimals(8)
        origin_layout.addWidget(self.origin_lon_input)

        origin_layout.addWidget(QLabel("原点纬度:"))
        self.origin_lat_input = QDoubleSpinBox()
        self.origin_lat_input.setRange(-90, 90)
        self.origin_lat_input.setDecimals(8)
        origin_layout.addWidget(self.origin_lat_input)

        homography_layout.addLayout(origin_layout)

        utm_layout = QHBoxLayout()
        utm_layout.addWidget(QLabel("UTM区域:"))
        self.utm_zone_input = QSpinBox()
        self.utm_zone_input.setRange(1, 60)
        utm_layout.addWidget(self.utm_zone_input)

        utm_layout.addWidget(QLabel("半球:"))
        self.hemisphere_combo = QComboBox()
        self.hemisphere_combo.addItems(["北半球", "南半球"])
        utm_layout.addWidget(self.hemisphere_combo)

        homography_layout.addLayout(utm_layout)
        right_layout.addWidget(homography_group)

        vehicle_group = QGroupBox("车辆选择")
        vehicle_layout = QVBoxLayout(vehicle_group)
        self.vehicle_list = QListWidget()
        vehicle_layout.addWidget(self.vehicle_list)
        right_layout.addWidget(vehicle_group)

        info_group = QGroupBox("车辆信息")
        info_layout = QVBoxLayout(info_group)
        self.info_table = QTableWidget()
        self.info_table.setColumnCount(6)
        self.info_table.setHorizontalHeaderLabels(["属性", "值", "属性", "值", "属性", "值"])
        info_layout.addWidget(self.info_table)
        right_layout.addWidget(info_group)

        # --- 添加：是否保存帧与标注的开关 ---
        self.save_frame_annotations_cb = QCheckBox("保存帧与标注到本地")
        self.save_frame_annotations_cb.setChecked(False)
        right_layout.addWidget(self.save_frame_annotations_cb)

        path_layout = QHBoxLayout()
        self.save_folder_le = QLineEdit()
        self.save_folder_le.setPlaceholderText("标注保存目录（默认空则使用临时目录）")
        path_layout.addWidget(self.save_folder_le)
        self.choose_save_folder_btn = QPushButton("选择保存目录")
        path_layout.addWidget(self.choose_save_folder_btn)
        right_layout.addLayout(path_layout)

        # 人工修正按钮（用户主动触发）
        manual_layout = QHBoxLayout()
        self.request_manual_btn = QPushButton("手动修正（打开 labelImg）")
        self.request_manual_btn.setEnabled(False)
        manual_layout.addWidget(self.request_manual_btn)
        right_layout.addLayout(manual_layout)

        export_layout = QHBoxLayout()
        self.export_btn = QPushButton("导出数据")
        self.export_btn.setEnabled(False)
        export_layout.addWidget(self.export_btn)

        self.clear_btn = QPushButton("清除数据")
        export_layout.addWidget(self.clear_btn)
        right_layout.addLayout(export_layout)

        main_layout.addWidget(left_panel, 70)
        main_layout.addWidget(right_panel, 30)

    def connect_signals(self):
        self.load_video_btn.clicked.connect(self.load_video)
        self.load_model_btn.clicked.connect(self.browse_model_folder)
        self.refresh_models_btn.clicked.connect(self.refresh_model_list)
        self.load_selected_model_btn.clicked.connect(self.load_selected_model)

        self.start_btn.clicked.connect(self.start_processing)
        self.stop_btn.clicked.connect(self.stop_processing)
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.load_h_file_btn.clicked.connect(self.load_homography_from_file)
        self.apply_h_btn.clicked.connect(self.apply_homography)
        self.export_btn.clicked.connect(self.export_data)
        self.clear_btn.clicked.connect(self.clear_data)

        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        self.fit_btn.clicked.connect(self.fit_to_window)
        self.zoom_slider.valueChanged.connect(self.on_zoom_slider_changed)

        self.vehicle_list.itemSelectionChanged.connect(self.on_vehicle_selected)

        # YOLOProcessor 信号连接（若存在）
        try:
            self.yolo_processor.frame_processed.connect(self.on_frame_processed)
            self.yolo_processor.progress_updated.connect(self.on_progress_updated)
            self.yolo_processor.fps_updated.connect(self.on_fps_updated)
            self.yolo_processor.error_occurred.connect(self.on_error_occurred)
            self.yolo_processor.processing_finished.connect(self.on_processing_finished)
            self.yolo_processor.detection_info_updated.connect(self.on_detection_info_updated)
            self.yolo_processor.model_loaded.connect(self.on_model_loaded)
        except Exception as e:
            self.status_text.append(f"信号连接失败: {str(e)}")

        self.video_label.clicked.connect(self.on_video_label_clicked)

        # 连接保存路径与手动修正按钮信号
        try:
            self.choose_save_folder_btn.clicked.connect(self.on_choose_save_folder)
            self.request_manual_btn.clicked.connect(self.on_request_manual_correction)
        except Exception:
            pass

        # 新增：当 model_combo 改变时，依据当前选项启用/禁用 load 按钮
        try:
            self.model_combo.currentIndexChanged.connect(self._on_model_combo_changed)
        except Exception:
            pass

    def _on_model_combo_changed(self, idx):
        try:
            txt = self.model_combo.currentText()
            ok = bool(txt) and not txt.startswith("（该文件夹中未发现")
            self.load_selected_model_btn.setEnabled(ok)
        except Exception:
            pass

    # ---------- 缩放相关 ----------
    def zoom_in(self):
        cur = self.video_label.get_zoom()
        new = min(4.0, cur * 1.2)
        self.video_label.set_zoom(new)
        self.zoom_slider.setValue(int(new * 100))

    def zoom_out(self):
        cur = self.video_label.get_zoom()
        new = max(0.1, cur / 1.2)
        self.video_label.set_zoom(new)
        self.zoom_slider.setValue(int(new * 100))

    def fit_to_window(self):
        self.video_label.reset_view()
        self.zoom_slider.setValue(100)

    def on_zoom_slider_changed(self, val):
        z = val / 100.0
        self.video_label.set_zoom(z)

    # ---------- 视频加载 ----------
    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)"
        )

        if not file_path:
            return

        # 关闭已有预览/资源
        try:
            if hasattr(self, 'preview_timer') and self.preview_timer.isActive():
                self.preview_timer.stop()
        except Exception:
            pass
        try:
            if getattr(self, 'cap', None) is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
        except Exception:
            pass

        # 打开视频并保留 cap 以便预览或后续处理使用
        cap = cv2.VideoCapture(file_path)
        if not cap or not cap.isOpened():
            self.status_text.append(f"无法打开视频文件: {file_path}")
            return

        self.video_path = file_path
        self.cap = cap
        # 读取 fps（若可获取）
        try:
            self.video_fps = cap.get(cv2.CAP_PROP_FPS) or None
        except Exception:
            self.video_fps = None

        # 读取第一帧并显示为静态预览
        ret, frame = cap.read()
        if not ret or frame is None:
            self.status_text.append("读取视频第一帧失败，可能是编码不兼容。")
            # 释放 cap（避免资源占用）
            try:
                cap.release()
                self.cap = None
            except Exception:
                pass
            return

        self.current_frame = frame
        # 如果你想把 frame_id 记录为 0:
        self.last_frame_id = 0
        self.last_frame_time = 0.0
        try:
            self.video_label.set_frame(frame)
        except Exception as e:
            self.status_text.append(f"显示帧失败: {e}")

        # 启用预览播放按钮和开始处理按钮（若模型已加载）
        self.play_preview_btn.setEnabled(True)
        self.stop_preview_btn.setEnabled(False)
        self.start_btn.setEnabled(True)

        self.status_text.append(f'已选择视频: {file_path} (fps={self.video_fps})：已显示第一帧。如需连续预览请点击"播放预览"。')

    # ---------- 模型加载 ----------
    def browse_model_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择模型文件夹", "")
        if folder:
            self.model_folder = folder
            self.refresh_models_btn.setEnabled(True)
            self.status_text.append(f"模型文件夹: {folder}")
            # 我们在选择文件夹后自动刷新一次，省得用户还要手动点"刷新"
            try:
                self.refresh_model_list()
            except Exception as e:
                # 如果自动刷新出现问题，仍然允许用户手动刷新
                self.status_text.append(f"自动刷新模型列表失败: {e}")

    def refresh_model_list(self):
        self.model_combo.clear()
        # disable load button by default
        try:
            self.load_selected_model_btn.setEnabled(False)
        except Exception:
            pass

        if not self.model_folder:
            self.status_text.append("未选择模型文件夹")
            return
        try:
            files = os.listdir(self.model_folder)
            candidates = [f for f in files if f.lower().endswith('.onnx') or f.lower().endswith('.pt') or f.lower().endswith('.pth')]
            if not candidates:
                self.model_combo.addItem("（该文件夹中未发现模型）")
                self.model_combo.setEnabled(False)
                # ensure load button disabled
                try:
                    self.load_selected_model_btn.setEnabled(False)
                except Exception:
                    pass
                self.status_text.append(f"在 {self.model_folder} 中未发现模型文件。")
            else:
                for c in candidates:
                    self.model_combo.addItem(c)
                self.model_combo.setEnabled(True)
                # **修正点：发现模型时启用"加载选中模型"按钮**
                try:
                    self.load_selected_model_btn.setEnabled(True)
                except Exception:
                    pass
                self.status_text.append(f"发现 {len(candidates)} 个模型文件")
        except Exception as e:
            self.status_text.append(f"读取模型目录失败: {e}")

    def load_selected_model(self):
        if not self.model_folder:
            self.status_text.append("未选择模型文件夹")
            return

        selected = self.model_combo.currentText()
        if not selected or selected.startswith("（该文件夹中未发现"):
            self.status_text.append("未选择有效模型文件")
            return

        model_path = os.path.join(self.model_folder, selected)
        if not os.path.exists(model_path):
            self.status_text.append(f"模型文件不存在: {model_path}")
            return

        try:
            success = self.yolo_processor.load_model(model_path)
            if success:
                self.status_text.append(f"模型加载成功: {model_path}")
                if self.video_path:
                    self.start_btn.setEnabled(True)
            else:
                self.status_text.append(f"模型加载失败: {model_path}")
        except Exception as e:
            self.status_text.append(f"加载模型时异常: {str(e)}")

    def start_processing(self):
        if not self.video_path:
            self.status_text.append("错误: 请先加载视频")
            return

        if self.homography_matrix is None:
            self.status_text.append("警告: 未设置单应矩阵，将无法计算地理坐标")

        try:
            self.yolo_processor.set_tracking_enabled(True)
            self.yolo_processor.set_target_fps(25)
        except Exception:
            pass

        self.yolo_thread = YOLOProcessorThread(self.yolo_processor)
        try:
            self.yolo_thread.set_video_path(self.video_path)
        except Exception:
            pass

        # === 新增：配置保存选项 ===
        try:
            if getattr(self, 'save_frame_annotations_cb', None) and self.save_frame_annotations_cb.isChecked():
                base = self.save_folder_le.text().strip() if self.save_folder_le.text() else None
                if not base:
                    from pathlib import Path
                    base = str(Path.home() / "traj_export")
                images_dir = os.path.join(base, "images")
                labels_dir = os.path.join(base, "labels")
                os.makedirs(images_dir, exist_ok=True)
                os.makedirs(labels_dir, exist_ok=True)

                self.yolo_processor.set_export_options(
                    save_txt=True, save_conf=True, output_dir=labels_dir,
                    save_images=True, image_dir=images_dir
                )
                self.status_text.append(f"已启用自动保存：{images_dir} / {labels_dir}")
        except Exception as e:
            self.status_text.append(f"设置自动保存失败：{e}")

        # === 再启动线程 ===
        try:
            self.yolo_thread.start()
            self.status_text.append("开始处理视频（线程已启动）.")
            self.start_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
        except Exception as e:
            self.status_text.append(f"启动处理失败: {str(e)}")


    def stop_processing(self):
        if self.yolo_thread and self.yolo_thread.isRunning():
            try:
                self.yolo_processor.stop_processing()
                self.yolo_thread.quit()
                self.yolo_thread.wait()
            except Exception:
                pass

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("暂停")
        self.status_text.append("处理已停止")

    def load_homography_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择单应矩阵文件", "", "NumPy文件 (*.npy);文本文件 (*.txt)"
        )

        if file_path:
            try:
                if file_path.endswith('.npy'):
                    H = np.load(file_path)
                else:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    H = self.parse_homography_text(content)

                if H is not None and H.shape == (3, 3):
                    self.homography_matrix = H
                    h_text = "\n".join([" ".join([f"{v:.12g}" for v in row]) for row in H.tolist()])
                    self.homography_input.setPlainText(h_text)
                    self.status_text.append(f"已加载单应矩阵 from {file_path}")
                else:
                    self.status_text.append("错误: 无效的单应矩阵文件")
            except Exception as e:
                self.status_text.append(f"加载单应矩阵失败: {str(e)}")

    def parse_homography_text(self, text):
        import re
        numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)
        if len(numbers) != 9:
            return None
        H = np.array([float(n) for n in numbers]).reshape(3, 3)
        if abs(H[2, 2]) > 1e-12:
            H = H / H[2, 2]
        return H

    def apply_homography(self):
        text = self.homography_input.toPlainText()
        H = self.parse_homography_text(text)

        if H is not None:
            self.homography_matrix = H
            self.status_text.append("单应矩阵已应用")

            self.origin_lonlat = (
                self.origin_lon_input.value(),
                self.origin_lat_input.value()
            )
            self.utm_zone = self.utm_zone_input.value()
            self.is_northern = self.hemisphere_combo.currentText() == "北半球"

            self.status_text.append(f"原点坐标: {self.origin_lonlat}")
            self.status_text.append(f"UTM区域: {self.utm_zone}, {'北半球' if self.is_northern else '南半球'}")
        else:
            self.status_text.append("输入的单应矩阵无效，请检查格式（9 个数字）。")

    def toggle_pause(self):
        """
        暂停 = stop 线程 + 保留当前位置
        继续 = 新建线程 + 从 last_frame_id+1 继续
        """
        paused = getattr(self, "_paused", False)

        if not paused:
            # --- 执行暂停 ---
            self._paused = True
            # 停止处理器线程，但不要清掉 last_frame_id
            try:
                if self.yolo_thread and self.yolo_thread.isRunning():
                    self.yolo_processor.stop_processing()
                    self.yolo_thread.quit()
                    self.yolo_thread.wait()
            except Exception:
                pass

            self.pause_btn.setText("继续")
            self.status_text.append(f"已暂停：位置停在帧 {getattr(self, 'last_frame_id', None)}。")
        else:
            # --- 执行继续 ---
            self._paused = False

            try:
                # 重新创建处理器线程
                self.yolo_thread = YOLOProcessorThread(self.yolo_processor)

                # 让处理器从 last_frame_id+1 继续
                start_frame = getattr(self, "last_frame_id", None)
                if start_frame is not None:
                    if hasattr(self.yolo_processor, "set_video_start_frame"):
                        self.yolo_processor.set_video_start_frame(int(start_frame) + 1)
                    else:
                        setattr(self.yolo_processor, "_requested_start_frame", int(start_frame) + 1)

                    # 如果 cap 存在，也 seek 一下
                    if getattr(self, "cap", None) is not None:
                        try:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame) + 1)
                        except Exception:
                            pass

                self.yolo_thread.set_video_path(self.video_path)
                self.yolo_thread.start()
                self.status_text.append(f"继续处理，从帧 {int(start_frame)+1 if start_frame is not None else 0} 开始。")

            except Exception as e:
                self.status_text.append(f"恢复继续失败: {e}")

            self.pause_btn.setText("暂停")



    # ---------- 处理 YOLO/跟踪器 输出 ----------
    def on_frame_processed(self, frame, detection_info):

        # 立即忽略来自处理器的帧（GUI 暂停状态）
        if getattr(self, "_paused", False):
            # 不更新显示、不处理 track（等待用户恢复）
            return
        """
        处理YOLO处理器返回的帧和检测信息
        """
        # 记录并注入系统时间（epoch float + ISO string）
        sys_now = datetime.now()
        system_timestamp = sys_now.timestamp()
        system_time_iso = sys_now.isoformat(timespec='milliseconds')

        # 确保 detection_info 是 dict
        if detection_info is None:
            detection_info = {}

        # Inject system time (如果外部已经提供则保留外部值)
        detection_info.setdefault('system_timestamp', system_timestamp)
        detection_info.setdefault('system_time', system_time_iso)

        # 显示当前帧
        self.video_label.set_frame(frame)
        self.current_frame = frame

        # 解析帧时间（视频时间 / 检测器时间） - 保持原有逻辑
        frame_time = None
        for key in ('frame_time', 'timestamp', 'time', 'frame_timestamp'):
            if key in detection_info and detection_info[key] is not None:
                try:
                    frame_time = float(detection_info[key])
                    break
                except Exception:
                    frame_time = None

        # 如果没有时间戳，使用帧号和FPS计算或系统时间作为备份
        if frame_time is None and 'frame_id' in detection_info:
            fid = int(detection_info.get('frame_id', 0))
            fps = getattr(self, "video_fps", None)
            if fps and fps > 0:
                frame_time = fid / float(fps)
            else:
                frame_time = system_timestamp
                self.status_text.append("警告：无法获取视频 FPS，用系统时间作为时间戳")

        detection_info['frame_time'] = frame_time

        # 更新车辆跟踪信息（现在 detection_info 中包含 system_time/system_timestamp）
        self.update_vehicle_tracks(detection_info)
        self.update_vehicle_list()
        self.last_frame_id = detection_info.get('frame_id')
        self.last_frame_time = detection_info.get('frame_time')
        self.last_detection_info = detection_info

        # 更新进度（保持原有）
        if 'frame_id' in detection_info and hasattr(self, 'cap'):
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current_frame = detection_info['frame_id']
            progress = int((current_frame / total_frames) * 100) if total_frames > 0 else 0
            self.progress_bar.setValue(progress)


    # ---------- 导出数据（CSV/Excel） ----------
    def export_data(self):
        if not self.vehicle_tracks:
            QMessageBox.information(self, "提示", "当前没有轨迹数据可导出。")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存数据", "", "CSV文件 (*.csv);;Excel文件 (*.xlsx)"
        )

        if not file_path:
            return

        try:
            all_data = []
            for vehicle_id, track in self.vehicle_tracks.items():
                n_positions = len(track['positions'])

                for i in range(n_positions):
                    data_row = {
                        'vehicle_id': vehicle_id,
                        'vehicle_type': track['class_name'],
                        'timestamp': track['frame_times'][i] if i < len(track['frame_times']) else None,
                        'system_time': track['system_time_strs'][i] if i < len(track.get('system_time_strs', [])) else None,
                        'pixel_x': track['positions'][i][0] if i < len(track['positions']) else None,
                        'pixel_y': track['positions'][i][1] if i < len(track['positions']) else None
                    }

                    # 地面坐标
                    if i < len(track['ground_positions']):
                        data_row['ground_x'] = track['ground_positions'][i][0]
                        data_row['ground_y'] = track['ground_positions'][i][1]

                    # 经纬度坐标
                    if i < len(track['lonlat_positions']):
                        data_row['longitude'] = track['lonlat_positions'][i][0]
                        data_row['latitude'] = track['lonlat_positions'][i][1]

                    # 速度信息
                    if i < len(track['velocities']) and track['velocities'][i]:
                        vel = track['velocities'][i]
                        data_row['speed_px_s'] = vel.get('speed_px')
                        data_row['speed_ms'] = vel.get('speed_ms')

                    # 方向信息
                    if i < len(track['directions']) and track['directions'][i]:
                        data_row['direction_deg'] = track['directions'][i].get('direction')

                    # 角点信息
                    if i < len(track['bbox_corners_px']):
                        corners = track['bbox_corners_px'][i]
                        for j, (x, y) in enumerate(corners):
                            if j < 4:
                                data_row[f'corner{j+1}_x'] = x
                                data_row[f'corner{j+1}_y'] = y

                    all_data.append(data_row)

            df = pd.DataFrame(all_data)
            if file_path.endswith('.csv'):
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
            else:
                df.to_excel(file_path, index=False)

            self.status_text.append(f"数据已导出到: {file_path}")
            QMessageBox.information(self, "导出成功", f"数据已成功导出到 {file_path}")
        except Exception as e:
            self.status_text.append(f"导出数据失败: {str(e)}")
            QMessageBox.critical(self, "导出失败", f"导出数据时发生错误: {str(e)}")


    def _on_preview_play(self):
        if getattr(self, 'cap', None) is None:
            self.status_text.append("请先加载视频再播放预览。")
            return
        # 若能获取真实 fps，则设置定时器间隔为 1000/fps ms
        if getattr(self, 'video_fps', None) and self.video_fps > 1:
            interval = max(10, int(1000.0 / float(self.video_fps)))
            try:
                self.preview_timer.setInterval(interval)
            except Exception:
                pass
        try:
            self.preview_timer.start()
            self.play_preview_btn.setEnabled(False)
            self.stop_preview_btn.setEnabled(True)
            self.status_text.append("开始视频预览")
        except Exception as e:
            self.status_text.append(f"启动预览失败: {e}")

    def _on_preview_stop(self):
        try:
            if getattr(self, 'preview_timer', None):
                self.preview_timer.stop()
            self.play_preview_btn.setEnabled(True)
            self.stop_preview_btn.setEnabled(False)
            self.status_text.append("停止视频预览")
        except Exception as e:
            self.status_text.append(f"停止预览失败: {e}")

    def _preview_next_frame(self):
        """
        QTimer 回调，从 self.cap 读取下一帧并显示。
        到达结尾时自动停止（或回到开头，取决于你想要的行为）。
        """
        cap = getattr(self, 'cap', None)
        if cap is None:
            self.preview_timer.stop()
            self.play_preview_btn.setEnabled(True)
            self.stop_preview_btn.setEnabled(False)
            return

        ret, frame = cap.read()
        if not ret or frame is None:
            # 到达视频末尾 - 停止预览并把 cap 回到起点（或直接停止）
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 回到开头（如果你希望循环）
                # 如果不想循环改为停止：
                # self.preview_timer.stop()
                # self.play_preview_btn.setEnabled(True)
                # self.stop_preview_btn.setEnabled(False)
                # self.status_text.append("已到达视频末尾，已停止预览。")
                # return
                # 这里选择循环播放以便测试
                ret2, frame = cap.read()
                if not ret2:
                    self.preview_timer.stop()
                    self.play_preview_btn.setEnabled(True)
                    self.stop_preview_btn.setEnabled(False)
                    self.status_text.append("读取视频帧失败，停止预览。")
                    return
            except Exception:
                self.preview_timer.stop()
                self.play_preview_btn.setEnabled(True)
                self.stop_preview_btn.setEnabled(False)
                self.status_text.append("读取视频帧异常，停止预览。")
                return

        # 显示帧
        try:
            self.current_frame = frame
            # 更新 frame id/time（自行按需求设置）
            if hasattr(self, 'last_frame_id') and self.last_frame_id is not None:
                try:
                    self.last_frame_id += 1
                except Exception:
                    self.last_frame_id = getattr(self, 'last_frame_id', 0)
            else:
                self.last_frame_id = 0
            self.last_frame_time = datetime.now().timestamp()
            self.video_label.set_frame(frame)
        except Exception as e:
            self.status_text.append(f"预览显示失败: {e}")


    def clear_data(self):
        self.vehicle_tracks.clear()
        self.update_vehicle_list()
        self.status_text.append("已清除轨迹数据")

    def on_progress_updated(self, cur, total, fps):
        try:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(cur)
        except Exception:
            pass

    def on_fps_updated(self, fps):
        self.status_text.append(f"处理器 FPS: {fps:.2f}")

    def on_error_occurred(self, errmsg):
        self.status_text.append("模型错误: " + str(errmsg))

    def on_processing_finished(self):
        self.status_text.append("处理完成")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def on_detection_info_updated(self, info):
        self.status_text.append(str(info))

    def on_model_loaded(self, model_path):
        self.status_text.append(f"模型已加载: {model_path}")

    def on_vehicle_selected(self):
        selected_items = self.vehicle_list.selectedItems()
        if not selected_items:
            self.selected_vehicle_id = None
            return
        
        item_text = selected_items[0].text()
        try:
            # 从格式化的文本中提取车辆ID
            parts = item_text.split(":")
            if len(parts) > 1:
                vehicle_id_str = parts[1].split("-")[0].strip()
                vehicle_id = int(vehicle_id_str)
                self.selected_vehicle_id = vehicle_id
                self.update_vehicle_info(vehicle_id)
        except (ValueError, IndexError):
            self.status_text.append("无法解析车辆ID")

    def update_vehicle_info(self, vehicle_id):
        if vehicle_id not in self.vehicle_tracks:
            return
        
        track = self.vehicle_tracks[vehicle_id]
        self.info_table.setRowCount(10)
        self.info_table.clearContents()
        
        # 基本信息
        self.info_table.setItem(0, 0, QTableWidgetItem("车辆ID"))
        self.info_table.setItem(0, 1, QTableWidgetItem(str(vehicle_id)))
        
        self.info_table.setItem(0, 2, QTableWidgetItem("类型"))
        self.info_table.setItem(0, 3, QTableWidgetItem(track['class_name']))
        
        self.info_table.setItem(0, 4, QTableWidgetItem("轨迹点数"))
        self.info_table.setItem(0, 5, QTableWidgetItem(str(len(track['positions']))))
        
        # 当前位置信息
        if track['positions']:
            x, y = track['positions'][-1]
            self.info_table.setItem(1, 0, QTableWidgetItem("当前位置(X)"))
            self.info_table.setItem(1, 1, QTableWidgetItem(f"{x:.2f}"))
            
            self.info_table.setItem(1, 2, QTableWidgetItem("当前位置(Y)"))
            self.info_table.setItem(1, 3, QTableWidgetItem(f"{y:.2f}"))
        
        # 经纬度信息
        if track['lonlat_positions']:
            lon, lat = track['lonlat_positions'][-1]
            self.info_table.setItem(2, 0, QTableWidgetItem("经度"))
            self.info_table.setItem(2, 1, QTableWidgetItem(f"{lon:.8f}"))
            
            self.info_table.setItem(2, 2, QTableWidgetItem("纬度"))
            self.info_table.setItem(2, 3, QTableWidgetItem(f"{lat:.8f}"))
        
        # 速度信息
        if track['velocities'] and track['velocities'][-1]:
            speed_info = track['velocities'][-1]
            self.info_table.setItem(3, 0, QTableWidgetItem("速度(像素/秒)"))
            self.info_table.setItem(3, 1, QTableWidgetItem(f"{speed_info.get('speed_px_s', 0):.2f}"))
            
            if speed_info.get('speed_ms') is not None:
                self.info_table.setItem(3, 2, QTableWidgetItem("速度(米/秒)"))
                self.info_table.setItem(3, 3, QTableWidgetItem(f"{speed_info['speed_ms']:.2f}"))
                self.info_table.setItem(3, 4, QTableWidgetItem("速度(km/h)"))
                self.info_table.setItem(3, 5, QTableWidgetItem(f"{speed_info['speed_ms']*3.6:.2f}"))
        
        # 方向信息
        if track['directions'] and track['directions'][-1]:
            direction_info = track['directions'][-1]
            self.info_table.setItem(4, 0, QTableWidgetItem("方向"))
            self.info_table.setItem(4, 1, QTableWidgetItem(f"{direction_info.get('direction', 0):.1f}°"))
        
        # 角点信息
        if track['bbox_corners_px']:
            corners = track['bbox_corners_px'][-1]
            for i, (x, y) in enumerate(corners):
                if i < 4:  # 只显示前4个角点
                    self.info_table.setItem(5+i, 0, QTableWidgetItem(f"角点{i+1}(X)"))
                    self.info_table.setItem(5+i, 1, QTableWidgetItem(f"{x:.2f}"))
                    self.info_table.setItem(5+i, 2, QTableWidgetItem(f"角点{i+1}(Y)"))
                    self.info_table.setItem(5+i, 3, QTableWidgetItem(f"{y:.2f}"))

    def update_vehicle_list(self):
        self.vehicle_list.clear()
        for vehicle_id, track in sorted(self.vehicle_tracks.items()):
            item_text = f"ID: {vehicle_id} - {track['class_name']} - 轨迹点: {len(track['positions'])}"
            self.vehicle_list.addItem(item_text)
        
        if self.vehicle_tracks:
            self.export_btn.setEnabled(True)

    def on_video_label_clicked(self, x, y):
        self.status_text.append(f"点击像素: ({x:.1f}, {y:.1f})")
        # 新增：调用设置原点的方法
        self.set_origin_point(int(x), int(y))

    def set_origin_point(self, img_x, img_y):
        dialog = OriginDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            lon, lat = dialog.get_coords()
            if lon is not None and lat is not None:
                self.origin_lonlat = (lon, lat)
                self.origin_pixel = (img_x, img_y)

                # 自动计算 UTM 区号与半球
                self.calculate_utm_zone()

                self.origin_lon_input.setValue(lon)
                self.origin_lat_input.setValue(lat)
                self.status_text.append(f"已设置原点: 像素坐标({img_x}, {img_y}), 经纬度({lon}, {lat})")

    def calculate_utm_zone(self):
        if not getattr(self, "origin_lonlat", None):
            self.status_text.append("无法计算 UTM 区域：尚未设置原点经纬度。")
            return None

        lon, lat = self.origin_lonlat
        zone = int((float(lon) + 180.0) / 6.0) + 1
        zone = max(1, min(60, zone))
        self.utm_zone = zone
        try:
            self.utm_zone_input.setValue(zone)
        except Exception:
            pass

        is_north = float(lat) >= 0.0
        self.is_northern = is_north
        try:
            self.hemisphere_combo.setCurrentText("北半球" if is_north else "南半球")
        except Exception:
            pass

        self.status_text.append(f"已自动计算 UTM 区号: {zone}，半球: {'北半球' if is_north else '南半球'}")
        return zone

        # ---------- 关键函数：update_vehicle_tracks（记录 bbox 四角点 + 中心点） ----------
    def update_vehicle_tracks(self, detection_info):
        """
        detection_info 结构：
        {
            'frame_id': int,
            'frame_time': float_seconds,   # 视频时间（可选）
            'system_timestamp': float,     # 新增：系统 epoch 秒（float）
            'system_time': str,            # 新增：系统 ISO 字符串（毫秒精度）
            'objects': [ { 'track_id': id, 'bbox': [.], 'bbox_type': 'obb'/'rect', 'class_name': str }, . ]
        }
        """
        frame_id = detection_info.get('frame_id', None)
        frame_time = detection_info.get('frame_time', None)

        # 使用传入的 system_timestamp（若有），否则用当前系统时间
        system_ts = detection_info.get('system_timestamp', None)
        if system_ts is None:
            system_ts = datetime.now().timestamp()
        # system iso 字符串
        system_iso = detection_info.get('system_time', datetime.fromtimestamp(system_ts).isoformat(timespec='milliseconds'))

        current_ids = set()
        for obj in detection_info.get('objects', []):
            vehicle_id = obj.get('track_id', None)
            if vehicle_id is None:
                vehicle_id = obj.get('track_uuid', None)
            if vehicle_id is None:
                continue
            current_ids.add(vehicle_id)

            # 解析 bbox -> corners_px & center_px
            if obj.get('bbox_type') == 'obb':
                pts = np.array(obj.get('bbox', [])).reshape(-1, 2)
                corners_px = [(float(pts[i,0]), float(pts[i,1])) for i in range(pts.shape[0])]
                if len(corners_px) > 4:
                    corners_px = corners_px[:4]
                elif len(corners_px) < 4:
                    xs = pts[:,0]; ys = pts[:,1]
                    x_min, x_max = float(np.min(xs)), float(np.max(xs))
                    y_min, y_max = float(np.min(ys)), float(np.max(ys))
                    corners_px = [(x_min,y_min),(x_max,y_min),(x_max,y_max),(x_min,y_max)]
                cx = float(np.mean([p[0] for p in corners_px]))
                cy = float(np.mean([p[1] for p in corners_px]))
                center_px = (cx, cy)
            else:
                x1, y1, x2, y2 = obj.get('bbox', [0,0,0,0])
                corners_px = [(float(x1), float(y1)), (float(x2), float(y1)),
                            (float(x2), float(y2)), (float(x1), float(y2))]
                center_px = ((float(x1)+float(x2))/2.0, (float(y1)+float(y2))/2.0)

            # 计算经纬（若 H 可用）
            corners_lonlat = []
            center_lonlat = (None, None)
            if self.homography_matrix is not None and self.origin_lonlat is not None:
                try:
                    for (px, py) in corners_px:
                        gx, gy = self.image_to_ground((px, py), self.homography_matrix)
                        lon, lat = self.utm_to_lonlat(gx, gy)
                        corners_lonlat.append((lon, lat))
                    gx, gy = self.image_to_ground(center_px, self.homography_matrix)
                    clon, clat = self.utm_to_lonlat(gx, gy)
                    center_lonlat = (clon, clat)
                except Exception as e:
                    self.status_text.append(f"坐标转换错误(角点/中心): {e}")
                    corners_lonlat = [(None, None)] * len(corners_px)
                    center_lonlat = (None, None)
            else:
                corners_lonlat = [(None, None)] * len(corners_px)
                center_lonlat = (None, None)

            # 新建或追加 track（新增 system_times / system_time_strs / bbox_centers_px 保持）
            if vehicle_id not in self.vehicle_tracks:
                self.vehicle_tracks[vehicle_id] = {
                    'class_name': obj.get('class_name', 'unknown'),
                    'positions': [],
                    'frame_times': [],
                    'system_times': [],           # 新增：epoch float
                    'system_time_strs': [],       # 新增：ISO 字符串
                    'ground_positions': [],
                    'lonlat_positions': [],
                    'velocities': [],
                    'directions': [],
                    'bbox_corners_px': [],
                    'bbox_centers_px': [],
                    'bbox_corners_lonlat': [],
                    'bbox_centers_lonlat': [],
                    'last_frame': None
                }

            track = self.vehicle_tracks[vehicle_id]
            track['positions'].append(center_px)
            # 兼容原有 frame_time（视频时间）
            track['frame_times'].append(float(frame_time) if frame_time is not None else None)
            # 存系统时间
            try:
                track['system_times'].append(float(system_ts))
                track['system_time_strs'].append(str(system_iso))
            except Exception:
                track['system_times'].append(float(datetime.now().timestamp()))
                track['system_time_strs'].append(datetime.now().isoformat(timespec='milliseconds'))

            # ground/ lonlat center (compat)
            if self.homography_matrix is not None and self.origin_lonlat is not None:
                try:
                    gx, gy = self.image_to_ground(center_px, self.homography_matrix)
                    track['ground_positions'].append((float(gx), float(gy)))
                    lon, lat = self.utm_to_lonlat(gx, gy)
                    track['lonlat_positions'].append((lon, lat))
                except Exception:
                    track['ground_positions'].append((None, None))
                    track['lonlat_positions'].append((None, None))
            else:
                track['ground_positions'].append((None, None))
                track['lonlat_positions'].append((None, None))

            track['bbox_corners_px'].append(corners_px)
            track['bbox_centers_px'].append(center_px)
            track['bbox_corners_lonlat'].append(corners_lonlat)
            track['bbox_centers_lonlat'].append(center_lonlat)
            track['last_frame'] = frame_id

            # 计算速度/方向（calculate_kinematics 内部会优先使用 system_times）
            if len(track['positions']) > 1:
                self.calculate_kinematics(vehicle_id)

        # 检测上一帧有但本帧缺失的 track -> 要求人工修正（保持原有逻辑）
        missing_ids = []
        for vid, tr in list(self.vehicle_tracks.items()):
            last = tr.get('last_frame', None)
            if last is not None and frame_id is not None:
                if last == frame_id - 1 and vid not in current_ids:
                    missing_ids.append(vid)

        if missing_ids:
            self.pending_missing_ids = missing_ids
            self.status_text.append(f'检测到漏检车辆 id: {missing_ids}。请在需要时点击"手动修正"按钮打开标注工具进行回填（不会自动弹出）。')
            try:
                self.request_manual_btn.setEnabled(True)
            except Exception:
                pass


    def calculate_kinematics(self, vehicle_id):
        track = self.vehicle_tracks.get(vehicle_id)
        if not track:
            return

        n = len(track['positions'])
        if n < 2:
            return

        x1, y1 = track['positions'][-2]
        x2, y2 = track['positions'][-1]
        t1 = float(track['frame_times'][-2])
        t2 = float(track['frame_times'][-1])
        dt = t2 - t1

        if dt <= 0:
            # 避免除零或负时间差
            return

        dx = x2 - x1
        dy = y2 - y1
        dist_px = np.hypot(dx, dy)
        speed_px = dist_px / dt

        speed_ms = None
        if len(track.get('ground_positions', [])) >= 2:
            gx1, gy1 = track['ground_positions'][-2]
            gx2, gy2 = track['ground_positions'][-1]
            dist_m = np.hypot(gx2 - gx1, gy2 - gy1)
            try:
                speed_ms = dist_m / dt
            except Exception:
                speed_ms = None

        direction = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0

        track['velocities'].append({
            'speed_px': float(speed_px),
            'speed_ms': float(speed_ms) if speed_ms is not None else None,
            'timestamp': t2
        })

        track['directions'].append({
            'direction': float(direction),
            'timestamp': t2
        })

    def haversine_m(self, lon1, lat1, lon2, lat2):
        # approximate distance (meters) between two lon/lat points
        try:
            R = 6371000.0
            phi1 = math.radians(lat1); phi2 = math.radians(lat2)
            dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
            a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            return R * c
        except Exception:
            return None

    def image_to_ground(self, pt_xy, H):
        arr = np.array([[[float(pt_xy[0]), float(pt_xy[1])]]], dtype=np.float64)
        proj = cv2.perspectiveTransform(arr, H).reshape(2)
        return float(proj[0]), float(proj[1])

    def utm_to_lonlat(self, easting, northing):
        try:
            from pyproj import CRS, Transformer
            if self.origin_lonlat is None:
                return None, None

            origin_lon, origin_lat = self.origin_lonlat
            zone = int((origin_lon + 180) / 6) + 1
            is_north = origin_lat >= 0
            epsg = 32600 + zone if is_north else 32700 + zone

            utm_crs = CRS.from_epsg(epsg)
            transformer = Transformer.from_crs(utm_crs, 'EPSG:4326', always_xy=True)

            origin_transformer = Transformer.from_crs('EPSG:4326', utm_crs, always_xy=True)
            ox, oy = origin_transformer.transform(origin_lon, origin_lat)
            world_x = easting + ox
            world_y = northing + oy

            lon, lat = transformer.transform(world_x, world_y)
            return lon, lat
        except Exception as e:
            self.status_text.append(f"UTM转经纬度错误: {str(e)}")
            return None, None

    def zoom_fit(self):
        self.video_label.reset_view()

    def on_choose_save_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择保存目录", "")
        if folder:
            self.save_folder_le.setText(folder)
            self.status_text.append(f"保存目录设置为: {folder}")

    def on_request_manual_correction(self):
        """
        用户点击“手动修正”时调用：
        - 确保使用视频原始帧（seek + read）
        - 打开 labelImg 编辑并返回 new_objs
        - 把 new_objs 应用回 vehicle_tracks（替换该帧数据 / 新增 id / 删除框）
        """
        frame_id = getattr(self, 'last_frame_id', None)
        if frame_id is None:
            self.status_text.append("没有可修正的帧（last_frame_id 未设置）。")
            return

        # 如果当前是运行状态（即没有暂停），先把处理彻底停下（stop + wait），以免产生新的预测
        was_running = not getattr(self, "_paused", False)
        if was_running:
            try:
                # stop completely
                if self.yolo_thread and self.yolo_thread.isRunning():
                    try:
                        self.yolo_processor.stop_processing()
                    except Exception:
                        pass
                    try:
                        self.yolo_thread.quit()
                        self.yolo_thread.wait()
                    except Exception:
                        pass
            except Exception:
                pass
            self._paused = True
            self.pause_btn.setText("继续")
            self.status_text.append(f"已暂停处理（用于人工修正），停在帧 {frame_id}。")

        # 获取原始视频帧（seek + read）
        frame = None
        cap = getattr(self, 'cap', None)
        opened_here = False
        try:
            if cap is None or not getattr(cap, "isOpened", lambda: True)():
                if getattr(self, 'video_path', None):
                    cap = cv2.VideoCapture(self.video_path)
                    opened_here = True
        except Exception:
            cap = None

        if cap is not None:
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
                ret, frm = cap.read()
                if ret and frm is not None:
                    frame = frm.copy()
            except Exception:
                frame = None

        if opened_here and cap is not None:
            try:
                cap.release()
            except Exception:
                pass

        if frame is None:
            # fallback to last shown frame if cannot read original frame
            frame = getattr(self, 'current_frame', None)
            if frame is None:
                self.status_text.append("无法读取指定帧的原始图像，无法进入人工修正。")
                # 若之前是运行状态，恢复它
                if was_running:
                    try:
                        # resume by restarting thread from last_frame_id+1
                        self._paused = False
                        self.toggle_pause()
                    except Exception:
                        pass
                return

        detection_info = getattr(self, 'last_detection_info', {}) or {}
        frame_time = detection_info.get('frame_time', None)

        try:
            new_objs = self.prompt_manual_correction(frame, int(frame_id), getattr(self, 'pending_missing_ids', []), detection_info)
        except Exception as e:
            self.status_text.append(f"调用人工修正失败: {e}")
            new_objs = []

        if not new_objs:
            self.status_text.append("人工修正结束：没有检测到新的标注或修改。")
            self.pending_missing_ids = []
            return

        try:
            self.apply_manual_annotations(int(frame_id), new_objs, frame_time=frame_time)
            self.status_text.append(f"人工修正已应用到帧 {frame_id}。")
        except Exception as e:
            self.status_text.append(f"应用人工修正失败: {e}")

        # 清理 pending
        self.pending_missing_ids = []



    def prompt_manual_correction(self, frame_bgr, frame_id, pending_ids, detection_info):
        """
        打开 labelImg 对当前帧进行人工修正：
        - 统一使用 base/images 与 base/labels 作为图片/标签的“标准位置”
        - 若该帧已有标签，则直接复用（不预写）
        - 编辑完成后，把 images/{name}.txt 覆盖回 labels/{name}.txt
        - 返回解析后的 new_objs（与 update_vehicle_tracks 兼容）
        """
        import os, shutil, cv2, numpy as _np, subprocess
        from pathlib import Path

        if frame_bgr is None:
            raise ValueError("frame_bgr is None")

        # 1) 基准目录：优先用UI里设置的目录；若未设置，固定到 ~/traj_export
        base = (self.save_folder_le.text().strip()
                if getattr(self, 'save_folder_le', None) and self.save_folder_le.text().strip()
                else str(Path.home() / "traj_export"))
        images_dir = os.path.join(base, "images")
        labels_dir = os.path.join(base, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        # 2) 规范化文件名：既兼容 fid，也兼容 06d；若 labels 里已有 06d 命名，就用 06d，否则用 fid
        fid = int(frame_id)
        name_6 = f"{fid:06d}"
        name = name_6 if (os.path.exists(os.path.join(labels_dir, name_6 + ".txt"))
                        or os.path.exists(os.path.join(images_dir, name_6 + ".jpg"))) else str(fid)

        canonical_img = os.path.join(images_dir, name + ".jpg")   # 标准图片位置
        canonical_lbl = os.path.join(labels_dir,  name + ".txt")  # 标准标签位置
        working_img   = canonical_img
        working_lbl   = os.path.join(images_dir, name + ".txt")   # 让 labelImg 在图片旁边读到标签

        # 3) 保存当前帧图片（覆盖写，保证就是当前帧）
        cv2.imwrite(canonical_img, frame_bgr)

        h, w = frame_bgr.shape[:2]

        # 4) 若已有标签：直接复用；否则再按当前检测结果“首次预写”
        def _prefill_from_detection():
            det_objs = detection_info.get('objects', []) if isinstance(detection_info, dict) else []
            objs_to_write = []
            if det_objs:
                # 用检测器给的对象
                objs_to_write = det_objs
            else:
                # 退化：从 vehicle_tracks 最后一帧拼回
                for vid, tr in self.vehicle_tracks.items():
                    if tr.get('last_frame', None) == fid:
                        if tr.get('bbox_corners_px'):
                            corners = tr['bbox_corners_px'][-1]
                            flat_norm = []
                            for (px, py) in corners:
                                flat_norm += [float(px)/w, float(py)/h]
                            objs_to_write.append(dict(
                                track_id=vid, class_id=0, class_name=tr.get('class_name'),
                                confidence=1.0, bbox_type='obb', bbox=flat_norm))
            if not objs_to_write:
                return
            with open(working_lbl, 'w', encoding='utf-8') as f:
                for o in objs_to_write:
                    cid  = int(o.get('class_id', 0) if o.get('class_id') is not None else 0)
                    conf = float(o.get('confidence', 1.0))
                    tid  = int(o['track_id']) if o.get('track_id') is not None else -1
                    if o.get('bbox_type') == 'obb':
                        flat = o.get('bbox', [])
                        # 展平+归一化保护
                        if any(isinstance(v, (list, tuple)) for v in flat):
                            tmp = []
                            for item in flat:
                                tmp += [float(v) for v in item] if isinstance(item, (list,tuple)) else [float(item)]
                            flat = tmp
                        else:
                            flat = [float(x) for x in flat]
                        if len(flat) == 8 and max(flat) > 1.0:
                            # 像素 → 归一化
                            flat = [ (flat[i]/w if i%2==0 else flat[i]/h) for i in range(8) ]
                        line = " ".join([str(cid)] + [f"{v:.6f}" for v in flat] + [f"{conf:.6f}", str(tid)])
                        f.write(line + "\n")

        if os.path.exists(canonical_lbl):
            # 有历史标签：复制到图片旁边，labelImg 会自动加载
            shutil.copy2(canonical_lbl, working_lbl)
        else:
            # 首次预写
            _prefill_from_detection()

        # 5) 启动 labelImg（优先内嵌，否则子进程）
        launched = False
        try:
            import labelImg
            mw = labelImg.MainWindow(defaultFilename=working_img)
            mw.show(); mw.raise_(); mw.activateWindow()
            from PyQt6.QtCore import QEventLoop
            loop = QEventLoop()
            mw.destroyed.connect(loop.quit)
            loop.exec()
            launched = True
        except Exception:
            pass

        if not launched:
            candidate = None
            base_dir = os.path.abspath(os.path.dirname(__file__))
            cur = base_dir
            for _ in range(6):
                p = os.path.join(cur, "projects", "labelimg_OBB", "labelImg.py")
                if os.path.exists(p):
                    candidate = p; break
                cur = os.path.dirname(cur)
            if candidate is None:
                alt = os.path.join(os.path.dirname(__file__), "labelImg.py")
                if os.path.exists(alt):
                    candidate = alt
            if candidate:
                subprocess.call([sys.executable, candidate, working_img])
            else:
                self.status_text.append(f"未找到 labelImg，可手工编辑 {working_lbl} 后返回。")
                return []

        # 6) 用户保存并退出后：把 images/{name}.txt 覆盖回 labels/{name}.txt（你要的“原地覆盖”）
        if os.path.exists(working_lbl):
            os.makedirs(labels_dir, exist_ok=True)
            shutil.copy2(working_lbl, canonical_lbl)

        # 7) 解析标签文件 → new_objs
        if not os.path.exists(working_lbl):
            self.status_text.append("未检测到保存的标注文件。")
            return []

        new_objs = []
        with open(working_lbl, 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        for ln in lines:
            parts = ln.split()
            if len(parts) < 5:
                continue
            try:
                cid = int(float(parts[0]))
            except Exception:
                cid = 0

            if len(parts) >= 11:  # OBB: class + 8 + conf + tid
                nums = list(map(float, parts[1:9]))
                pts = _np.array(nums, dtype=float).reshape(-1, 2)
                if _np.max(_np.abs(pts)) <= 1.0:  # 归一化 → 像素
                    pts[:, 0] *= w; pts[:, 1] *= h
                conf = float(parts[9]) if parts[9] != "" else 1.0
                try:
                    tid = int(float(parts[10])) if len(parts) >= 11 and parts[10] != "" else -1
                except Exception:
                    tid = -1
                new_objs.append({
                    'track_id': tid if tid >= 0 else None,
                    'class_id': cid,
                    'class_name': None,
                    'confidence': conf,
                    'bbox_type': 'obb',
                    'bbox': list(pts.reshape(-1))
                })
            elif len(parts) >= 6:  # 兼容 HBB: class cx cy w h [conf] [tid]
                nums = list(map(float, parts[1:5]))
                cx, cy, bw, bh = nums[:4]
                conf = float(parts[5]) if len(parts) >= 6 else 1.0
                tid = int(float(parts[6])) if len(parts) >= 7 else -1
                # 转为像素矩形
                x1 = (cx - bw/2) * w; y1 = (cy - bh/2) * h
                x2 = (cx + bw/2) * w; y2 = (cy + bh/2) * h
                new_objs.append({
                    'track_id': tid if tid >= 0 else None,
                    'class_id': cid,
                    'class_name': None,
                    'confidence': conf,
                    'bbox_type': 'rect',
                    'bbox': [x1, y1, x2, y2]
                })

        return new_objs



    def apply_manual_annotations(self, frame_id, new_objs, frame_time=None):
        """
        把 new_objs 应用回 vehicle_tracks：
        - 使用 track_id 替换/插入对应帧条目
        - 新框(无 track_id)分配新 ID
        - 删除旧被用户删掉的框
        - 重新计算被修改轨迹的速度/方向
        """
        if frame_time is None:
            frame_time = datetime.now().timestamp()

        # 记录原本在该帧存在的 track ids（以 last_frame==frame_id 为依据）
        old_ids_at_frame = set()
        for vid, tr in self.vehicle_tracks.items():
            if tr.get('last_frame', None) == frame_id:
                old_ids_at_frame.add(vid)

        new_ids = set()
        # 处理所有 new_objs：插入或替换对应 track 的该帧条目
        for o in new_objs:
            tid = o.get('track_id', None)
            if tid is None:
                tid = self.allocate_new_id()
            new_ids.add(tid)

            # 计算 corners_px 和 center_px
            if o.get('bbox_type') == 'obb':
                flat = o.get('bbox', [])
                # flatten nested if needed
                if any(isinstance(v, (list, tuple)) for v in flat):
                    flat2 = []
                    for item in flat:
                        if isinstance(item, (list, tuple)):
                            for vv in item:
                                flat2.append(float(vv))
                        else:
                            flat2.append(float(item))
                    flat = flat2
                pts = np.array(flat, dtype=float).reshape(-1, 2)
                corners_px = [(float(pts[i,0]), float(pts[i,1])) for i in range(pts.shape[0])]
                cx = float(np.mean([p[0] for p in corners_px])); cy = float(np.mean([p[1] for p in corners_px]))
                center_px = (cx, cy)
            else:
                x1, y1, x2, y2 = map(float, o.get('bbox', [0,0,0,0]))
                corners_px = [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
                center_px = ((x1+x2)/2.0, (y1+y2)/2.0)

            # ensure track exists
            if tid not in self.vehicle_tracks:
                self.vehicle_tracks[tid] = {
                    'class_name': o.get('class_name', f'cls{int(o.get("class_id",0))}'),
                    'positions': [],
                    'frame_times': [],
                    'ground_positions': [],
                    'lonlat_positions': [],
                    'velocities': [],
                    'directions': [],
                    'bbox_corners_px': [],
                    'bbox_centers_px': [],
                    'bbox_corners_lonlat': [],
                    'bbox_centers_lonlat': [],
                    'last_frame': None
                }
            tr = self.vehicle_tracks[tid]

            # find an existing index for this frame (by frame_times approximate match)
            idx = None
            for i, ft in enumerate(tr.get('frame_times', [])):
                try:
                    if abs(float(ft) - float(frame_time)) < 1e-3:
                        idx = i
                        break
                except Exception:
                    continue
            # fallback: if last_frame == frame_id, assume last index corresponds
            if idx is None and tr.get('last_frame', None) == frame_id and len(tr.get('frame_times', []))>0:
                idx = len(tr['frame_times']) - 1

            if idx is None:
                # append
                tr['positions'].append(center_px)
                tr['frame_times'].append(float(frame_time))
                tr['bbox_corners_px'].append(corners_px)
                tr['bbox_centers_px'].append(center_px)
                tr['last_frame'] = frame_id
                # compute ground & lonlat if possible
                if self.homography_matrix is not None and self.origin_lonlat is not None:
                    try:
                        gx, gy = self.image_to_ground(center_px, self.homography_matrix)
                        tr['ground_positions'].append((float(gx), float(gy)))
                        lon, lat = self.utm_to_lonlat(gx, gy)
                        tr['lonlat_positions'].append((lon, lat))
                    except Exception:
                        tr['ground_positions'].append((None, None))
                        tr['lonlat_positions'].append((None, None))
            else:
                # replace at idx
                tr['positions'][idx] = center_px
                tr['frame_times'][idx] = float(frame_time)
                tr['bbox_corners_px'][idx] = corners_px
                tr['bbox_centers_px'][idx] = center_px
                tr['last_frame'] = frame_id
                if self.homography_matrix is not None and self.origin_lonlat is not None:
                    try:
                        gx, gy = self.image_to_ground(center_px, self.homography_matrix)
                        tr['ground_positions'][idx] = (float(gx), float(gy))
                        lon, lat = self.utm_to_lonlat(gx, gy)
                        tr['lonlat_positions'][idx] = (lon, lat)
                    except Exception:
                        tr['ground_positions'][idx] = (None, None)
                        tr['lonlat_positions'][idx] = (None, None)

        # 移除被用户删掉的框（old_ids_at_frame 中，但不在 new_ids 中）
        removed = old_ids_at_frame - new_ids
        for vid in removed:
            tr = self.vehicle_tracks.get(vid)
            if not tr:
                continue
            # find index to remove (match frame_time if possible)
            idx_rm = None
            for i, ft in enumerate(tr.get('frame_times', [])):
                try:
                    if abs(float(ft) - float(frame_time)) < 1e-3:
                        idx_rm = i
                        break
                except Exception:
                    continue
            if idx_rm is None and tr.get('last_frame', None) == frame_id:
                idx_rm = len(tr.get('frame_times', [])) - 1
            if idx_rm is not None:
                for key in ['positions','frame_times','ground_positions','lonlat_positions','bbox_corners_px','bbox_centers_px','bbox_corners_lonlat','bbox_centers_lonlat','velocities','directions']:
                    if key in tr and len(tr[key]) > idx_rm:
                        try:
                            tr[key].pop(idx_rm)
                        except Exception:
                            pass
                # if track becomes empty, optionally remove it entirely
                if not tr.get('frame_times'):
                    try:
                        del self.vehicle_tracks[vid]
                    except Exception:
                        pass

        # 重新计算 velocities/directions 对被修改的 tracks（对简单实现：对所有 track 重新计算）
        for vid, tr in list(self.vehicle_tracks.items()):
            if len(tr.get('positions', [])) < 2:
                tr['velocities'] = []
                tr['directions'] = []
                continue
            tr['velocities'] = []
            tr['directions'] = []
            for i in range(1, len(tr['positions'])):
                x1,y1 = tr['positions'][i-1]
                x2,y2 = tr['positions'][i]
                t1 = float(tr['frame_times'][i-1])
                t2 = float(tr['frame_times'][i])
                dt = t2 - t1
                if dt <= 0:
                    tr['velocities'].append(None)
                    tr['directions'].append(None)
                    continue
                dx = x2 - x1; dy = y2 - y1
                dist_px = np.hypot(dx, dy)
                speed_px = dist_px / dt
                speed_ms = None
                if tr.get('ground_positions') and len(tr['ground_positions']) > i:
                    gx1, gy1 = tr['ground_positions'][i-1]
                    gx2, gy2 = tr['ground_positions'][i]
                    try:
                        speed_ms = np.hypot(gx2 - gx1, gy2 - gy1) / dt
                    except Exception:
                        speed_ms = None
                direction = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
                tr['velocities'].append({'speed_px': float(speed_px), 'speed_ms': float(speed_ms) if speed_ms is not None else None, 'timestamp': t2})
                tr['directions'].append({'direction': float(direction), 'timestamp': t2})




# ------------------------- 主程序启动 -------------------------
def main():
    app = QApplication(sys.argv)
    window = VehicleTrajectoryAnalyzer()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
