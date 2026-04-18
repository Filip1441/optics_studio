import sys
import os
import uuid
from PyQt6.QtWidgets import (QApplication, QMainWindow, QGraphicsView, 
                             QGraphicsScene, QGraphicsItem, QToolBar, 
                             QVBoxLayout, QWidget, QDoubleSpinBox, 
                             QLabel, QFormLayout, QPushButton, QFileDialog,
                             QGraphicsPathItem, QMenu, QHBoxLayout, QFrame,
                             QScrollArea, QDockWidget, QCheckBox, QGroupBox, QSpinBox)
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, QEvent, QObject, QRunnable, QThreadPool
from PyQt6.QtGui import QPen, QBrush, QColor, QAction, QPainterPath, QTransform, QPainter, QFont, QPixmap, QRadialGradient, QImage

import numpy as np

# Local imports
from optics_engine import Ray, OpticalSystem
from components import *
from persistence import SceneManager
from wave_engine import WaveEngine
from analysis_engine import calculate_analysis
from scipy.fft import fft2, fftshift

import logging
logger = logging.getLogger("SimulatorApp")
logger.setLevel(logging.WARNING)

class SaveWorker(QRunnable):
    def __init__(self, pixmap, path):
        super().__init__()
        self.pixmap = pixmap
        self.path = path
    def run(self):
        self.pixmap.save(self.path)
        logger.info(f"Worker: Saved to {self.path}")

class AnalysisSignals(QObject):
    finished = pyqtSignal(object)

class AnalysisWorker(QRunnable):
    def __init__(self, system, comp):
        super().__init__()
        self.system = system
        self.comp = comp
        self.signals = AnalysisSignals()
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            # Ta funkcja wykonuje się w osobnym wątku
            res = calculate_analysis(self.system, self.comp, cancel_check=lambda: self._is_cancelled)
            if res and not self._is_cancelled:
                image_arr, report_text = res
                self.signals.finished.emit((image_arr, report_text))
        except Exception as e:
            if not self._is_cancelled:
                print(f"Background Analysis Error: {e}")

DARK_THEME = """
QMainWindow, QWidget {
    background-color: #1e1e1e;
    color: #e0e0e0;
}
QToolBar {
    background: #2d2d2d;
    border: none;
    padding: 5px;
}
QDockWidget {
    color: #e0e0e0;
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}
QDockWidget::title {
    background: #333333;
    padding: 10px;
    border-radius: 4px;
}
QGroupBox {
    border: 1px solid #444;
    border-radius: 8px;
    margin-top: 15px;
    padding: 15px;
    font-weight: bold;
}
QLabel {
    color: #888;
}
QDoubleSpinBox {
    background: #252525;
    border: 1px solid #444;
    padding: 5px;
    color: #fff;
    border-radius: 4px;
}
QPushButton {
    background: #3d5afe;
    border-radius: 6px;
    padding: 10px;
    font-weight: bold;
    color: white;
}
QPushButton:hover { background: #536dfe; }
QPushButton#tool_btn {
    background: #333;
    border: 1px solid #444;
}
"""

class RotationHandle(QGraphicsItem):
    """Small circle handle to rotate the parent component."""
    def __init__(self, parent):
        super().__init__(parent)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setPos(0, -40) # 40px above center
        self.setCursor(Qt.CursorShape.SizeAllCursor)

    def boundingRect(self):
        return QRectF(-6, -6, 12, 12)

    def paint(self, painter, option, widget):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(255, 255, 255, 150), 2))
        painter.setBrush(QBrush(QColor(61, 90, 254)))
        painter.drawEllipse(-5, -5, 10, 10)
        # Line to center
        painter.setPen(QPen(QColor(255, 255, 255, 80), 1, Qt.PenStyle.DashLine))
        painter.drawLine(0, 5, 0, 40)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.parentItem():
            if getattr(self.parentItem(), '_syncing_handle', False):
                return value
            
            p = value
            angle_rad = np.arctan2(p.y(), p.x()) + np.pi/2
            angle_deg = np.degrees(angle_rad)
            
            # Aggressive rotation snapping (45°) for Mirrors by default
            # For others, snapping is 45° if Ctrl is held or 15° if not.
            snap_step = 45 if (isinstance(self.parentItem().component, Mirror) or 
                               QApplication.keyboardModifiers() & Qt.KeyboardModifier.ControlModifier) else 15
            angle_deg = round(angle_deg / snap_step) * snap_step
            
            self.parentItem().update_angle(angle_deg)
            r = 40
            new_x = r * np.cos(np.radians(angle_deg - 90))
            new_y = r * np.sin(np.radians(angle_deg - 90))
            return QPointF(new_x, new_y)
        return super().itemChange(change, value)

class VisualComponent(QGraphicsItem):
    """Enhanced graphical representation of an optical component."""
    def __init__(self, component: OpticalComponent, app_ref):
        super().__init__()
        self.component = component
        self.app = app_ref
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        
        self._syncing_handle = True # Start silenced to prevent handle-init reset
        self.handle = RotationHandle(self)
        self.handle.setVisible(False)
        self.refresh_pos()
        self._syncing_handle = False # Enable interaction after setup

    def refresh_pos(self):
        self._syncing_handle = True
        self.setPos(self.component.x * 5.0, self.component.y * 5.0)
        self.setRotation(self.component.angle)
        
        # Position handle correctly on the circle
        r = 40
        angle_rad = np.radians(self.component.angle - 90)
        hx = r * np.cos(angle_rad)
        hy = r * np.sin(angle_rad)
        self.handle.setPos(hx, hy)
        self._syncing_handle = False

    def update_angle(self, deg):
        logger.info(f"UI: Component {self.component.uid} rotation changed to {deg:.1f}°")
        self.component.angle = deg
        self.setRotation(deg)
        self.app.update_rays()
        self.app.props_panel.sync_data()

    def hoverEnterEvent(self, event):
        self.handle.setVisible(True)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        if not self.isSelected():
            self.handle.setVisible(False)
        super().hoverLeaveEvent(event)

    def boundingRect(self):
        # Calculate vertical reach (ri) based on component type
        if isinstance(self.component, Detector):
            size_mm = self.component.params.get("size", 10.0)
            ri = (size_mm / 2.0) * 5.0
        else:
            r_mm = self.component.params.get("r", 12.5)
            ri = r_mm * 5.0
            
        # Return a much tighter box: 
        # width from -15 to +15 (covering lens thickness/detector body)
        # height covering the full radius + small margin
        return QRectF(-15, -ri - 5, 30, 2 * ri + 10)

    def paint(self, painter, option, widget):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor(100, 200, 255) if self.isSelected() else QColor(255, 255, 255)
        
        # Shadows/Glow
        if self.isSelected():
            painter.setPen(QPen(QColor(100, 200, 255, 50), 8))
            self.draw_shape(painter)

        painter.setPen(QPen(color, 2))
        self.draw_shape(painter)
        
        # Label
        painter.setPen(QPen(QColor(255, 255, 255, 100), 1))
        painter.setFont(QFont("Inter", 8))
        painter.drawText(-20, 35, self.component.name.split('(')[0].strip())

    def draw_shape(self, painter):
        r = self.component.params.get("r", 12.5) * 5.0 # Half-width in pixels
        
        if isinstance(self.component, Lens):
            f = self.component.params.get("f", 50.0)
            ri = int(round(r))
            painter.drawLine(0, -ri, 0, ri)
            if f > 0:
                # Converging: Outward arrows
                painter.drawLine(-3, -ri+5, 0, -ri)
                painter.drawLine(3, -ri+5, 0, -ri)
                painter.drawLine(-3, ri-5, 0, ri)
                painter.drawLine(3, ri-5, 0, ri)
            else:
                # Diverging: Inward arrows
                painter.drawLine(-3, -ri, 3, -ri) # Flat ends
                painter.drawLine(-3, ri, 3, ri)
                painter.drawLine(0, -ri, -3, -ri+5)
                painter.drawLine(0, -ri, 3, -ri+5)
                painter.drawLine(0, ri, -3, ri-5)
                painter.drawLine(0, ri, 3, ri-5)
                
        elif isinstance(self.component, Mirror):
            ri = int(round(r))
            painter.setPen(QPen(painter.pen().color(), 3))
            painter.drawLine(0, -ri, 0, ri)
            painter.setPen(QPen(painter.pen().color(), 1))
            for i in np.arange(-ri, ri+1, 10):
                ii = int(i)
                painter.drawLine(0, ii, -5, ii+5)
        elif isinstance(self.component, TestTarget):
            # Draw a visual letter 'F' in the scene
            painter.setPen(QPen(QColor(255, 255, 100), 2))
            s = self.component.params.get('size', 5.0) * 2.5 # adjust for scene scale
            painter.drawLine(-2, -s, -2, s) # Stem
            painter.drawLine(-2, -s, 3, -s) # Top
            painter.drawLine(-2, 0, 1, 0)   # Mid
        elif isinstance(self.component, PointSource):
            painter.setBrush(QBrush(QColor(255, 50, 50)))
            painter.drawEllipse(-6, -6, 12, 12)
            # Directional line
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            painter.drawLine(0, 0, 10, 0)
        elif isinstance(self.component, (Lens, Grating, HighPassFilter, Aperture)):
            # Draw shared logic if any, or skip to specific
            pass
        elif isinstance(self.component, Aperture):
            # Aperture: Two blocks with a gap of 2*r
            painter.setPen(QPen(QColor(150, 150, 150), 4))
            gap_i = int(round(r))
            painter.drawLine(0, -500, 0, -gap_i) # Outer part
            painter.drawLine(0, gap_i, 0, 500)
            # Opening markers
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawLine(-3, -gap_i, 3, -gap_i)
            painter.drawLine(-3, gap_i, 3, gap_i)
        elif isinstance(self.component, Grating):
            # Grating: Symbol depends on pattern
            painter.setPen(QPen(QColor(200, 200, 50), 2))
            ri = int(round(r))
            pattern = self.component.params.get("pattern", "Linear")
            
            if pattern == "Linear":
                painter.drawLine(0, -ri, 0, ri)
                for i in np.arange(-ri, ri+1, 5):
                    ii = int(i)
                    painter.drawLine(0, ii, 4, ii)
            elif pattern in ["Crossed", "Chessboard"]:
                # Grid or Chessboard symbol
                for i in np.arange(-ri, ri+1, 10):
                    ii = int(i)
                    painter.drawLine(-ri, ii, ri, ii) 
                    painter.drawLine(ii, -ri, ii, ri)
                if pattern == "Chessboard":
                    painter.setPen(QPen(QColor(200, 200, 50, 100), 1))
                    painter.setBrush(QBrush(QColor(200, 200, 50, 50)))
                    for i in np.arange(-ri, ri, 10):
                        for j in np.arange(-ri, ri, 10):
                            if (i // 10 + j // 10) % 2 == 0:
                                painter.drawRect(i, j, 10, 10)
        elif self.component.__class__.__name__ == "HighPassFilter":
            # HighPassFilter: Central blocking dot on a thin line
            ri = int(round(r))
            # Support glass substrate line
            painter.setPen(QPen(QColor(255, 255, 255, 80), 1))
            painter.drawLine(0, -ri, 0, ri)
            # Opaque blocking center (the High-Pass part)
            block_r = self.component.params.get("r", 1.0) * 5.0
            painter.setBrush(QBrush(QColor(255, 100, 100)))
            painter.setPen(QPen(QColor(255, 100, 100), 1))
            painter.drawEllipse(-3, -int(block_r), 6, int(2*block_r))
        elif isinstance(self.component, Detector):
            # Facing LEFT by default to meet light from -X
            painter.setPen(QPen(QColor(148, 163, 184), 1))
            size_mm = self.component.params.get("size", 10.0)
            ri = int(round((size_mm / 2.0) * 5.0))
            
            # 1. Main Case (Back) - positioned on the Right
            painter.setBrush(QBrush(QColor(15, 23, 42)))
            painter.drawRect(2, -ri-4, 12, 2*ri+8)
            
            # 2. Sensor Mounting - positioned on the Left
            painter.setBrush(QBrush(QColor(30, 41, 59)))
            painter.drawRect(-10, -ri, 12, 2*ri)
            
            # 3. CCD/CMOS Active Surface (the green line is the actual sensor plane)
            painter.setPen(QPen(QColor(34, 197, 94), 3))
            painter.drawLine(-10, -ri+2, -10, ri-2)
            
            # 4. Small lens mount detail
            painter.setBrush(QBrush(QColor(71, 85, 105)))
            painter.drawRect(-14, int(-ri/2), 4, int(ri))

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.scene():
            if getattr(self, '_syncing_handle', False):
                return value
            
            new_pos = value
            
            # Dynamic Optical Axis Snapping
            if self.app.snapping_enabled:
                grid = 5.0 # 1.0mm = 5px
                new_pos.setX(round(new_pos.x() / grid) * grid)
                new_pos.setY(round(new_pos.y() / grid) * grid)
                
                axis_pts = self.app.system.get_axis_path()
                best_snap = None
                best_dist = 25 # 25px snapping radius
                
                # Check each segment of the optical axis path
                for i in range(len(axis_pts)-1):
                    p1 = np.array(axis_pts[i]) * 5.0
                    p2 = np.array(axis_pts[i+1]) * 5.0
                    pt = np.array([new_pos.x(), new_pos.y()])
                    
                    # Distance from point to line segment
                    v = p2 - p1
                    w = pt - p1
                    c1 = np.dot(w, v)
                    if c1 <= 0: d = np.linalg.norm(pt - p1)
                    else:
                        c2 = np.dot(v, v)
                        if c2 <= c1: d = np.linalg.norm(pt - p2)
                        else:
                            b = c1 / c2
                            pb = p1 + b * v
                            d = np.linalg.norm(pt - pb)
                    
                    if d < best_dist:
                        best_dist = d
                        # Project pt onto line (infinite line for better snap feel)
                        v_norm = v / np.linalg.norm(v)
                        pb = p1 + np.dot(pt - p1, v_norm) * v_norm
                        best_snap = (pb, v_norm)

                if best_snap:
                    pb, v_norm = best_snap
                    new_pos.setX(pb[0])
                    new_pos.setY(pb[1])
                    
                    # Auto-orient perpendicular to axis segment
                    if not isinstance(self.component, (Mirror, PointSource, BeamSource)):
                        angle_rad = np.arctan2(v_norm[1], v_norm[0])
                        angle_deg = np.degrees(angle_rad)
                        self.update_angle(angle_deg)

            self.component.x = new_pos.x() / 5.0
            self.component.y = new_pos.y() / 5.0
            logger.info(f"UI: Component {self.component.uid} snapped to Dynamic Axis. Angle: {self.component.angle}°")
            self.app.update_rays()
            self.app.props_panel.sync_data()
            return new_pos
        return super().itemChange(change, value)

class PropertyPanel(QWidget):
    """Side panel to edit selected component properties."""
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        self.label = QLabel("PROPERTIES")
        self.label.setStyleSheet("font-size: 14px; color: #3d5afe; font-weight: bold; margin-bottom: 20px;")
        self.layout.addWidget(self.label)
        
        self.form = QFormLayout()
        self.layout.addLayout(self.form)
        
        self.x_spin = QDoubleSpinBox()
        self.y_spin = QDoubleSpinBox()
        self.angle_spin = QDoubleSpinBox()
        
        for s in [self.x_spin, self.y_spin]:
            s.setRange(-2000, 2000)
            s.setSingleStep(1.0)
            s.setSuffix(" mm")
        
        self.angle_spin.setRange(-360, 360)
        self.angle_spin.setSuffix(" °")

        self.form.addRow("X Coordinate", self.x_spin)
        self.form.addRow("Y Coordinate", self.y_spin)
        self.angle_row = self.form.addRow("Rotation", self.angle_spin)

        self.specific_group = QGroupBox("Component Specific")
        self.specific_form = QFormLayout(self.specific_group)
        self.layout.addWidget(self.specific_group)

        # Detector Sensor Area
        self.sensor_label = QLabel("Click to see sensor")
        self.sensor_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sensor_label.setFixedSize(280, 280)
        self.sensor_label.setStyleSheet("border: 2px solid #3d5afe; border-radius: 4px; background: #000;")
        self.layout.addWidget(self.sensor_label)
        self.sensor_label.hide()
        
        self.x_spin.valueChanged.connect(self.apply_changes)
        self.y_spin.valueChanged.connect(self.apply_changes)
        self.angle_spin.valueChanged.connect(self.apply_changes)
        
        self._syncing = False

    def sync_data(self):
        items = self.app.scene.selectedItems()
        if not items or self._syncing: return
        
        self._syncing = True
        comp = items[0].component
        self.x_spin.setValue(comp.x)
        self.y_spin.setValue(comp.y)
        self.angle_spin.setValue(comp.angle)
        
        # Adjust label for sources
        label_text = "Emission Direction" if isinstance(comp, PointSource) else "Rotation"
        self.form.labelForField(self.angle_spin).setText(label_text)
        
        # Clear specific form
        while self.specific_form.count():
            child = self.specific_form.takeAt(0)
            if child.widget(): child.widget().deleteLater()
            
        # 1. Lens specifics
        if isinstance(comp, Lens):
            f_spin = QDoubleSpinBox()
            f_spin.setRange(-2000, 2000)
            f_spin.setValue(comp.params.get('f', 50.0))
            f_spin.setSuffix(" mm")
            f_spin.valueChanged.connect(lambda v: self.apply_param('f', v))
            self.specific_form.addRow("Focal Length", f_spin)
        
        # 2. Source specifics (Rays, Angle)
        if isinstance(comp, PointSource):
            nr_spin = QDoubleSpinBox()
            nr_spin.setRange(1, 100)
            nr_spin.setDecimals(0)
            nr_spin.setValue(comp.params.get('n_rays', 21))
            nr_spin.valueChanged.connect(lambda v: self.apply_param('n_rays', int(v)))
            self.specific_form.addRow("Num Rays", nr_spin)

            ar_spin = QDoubleSpinBox()
            ar_spin.setRange(1, 360)
            ar_spin.setSuffix(" °")
            current_deg = np.degrees(comp.params.get('angle_range', 0.1))
            ar_spin.setValue(current_deg)
            ar_spin.valueChanged.connect(lambda v: self.apply_param('angle_range', np.radians(v)))
            self.specific_form.addRow("Fan Angle", ar_spin)
            
            wl_spin = QDoubleSpinBox()
            wl_spin.setRange(380, 780)
            wl_spin.setSuffix(" nm")
            wl_spin.setValue(comp.params.get('wavelength', 532.0))
            wl_spin.valueChanged.connect(lambda v: self.apply_param('wavelength', v))
            self.specific_form.addRow("Wavelength", wl_spin)

        # 3. Diameter / Radius control (Only for valid targets)
        if isinstance(comp, (Lens, Mirror, Detector, Aperture, Grating, HighPassFilter, TestTarget)):
            if not isinstance(comp, Detector):
                r_spin = QDoubleSpinBox()
                r_spin.setRange(0.0, 500.0)
                r_spin.setDecimals(6)
                r_spin.setSuffix(" mm")
                label = "Opening (R)" if isinstance(comp, Aperture) else "Radius (R)"
                if isinstance(comp, TestTarget): label = "Size"
                self.specific_form.addRow(label, r_spin)
                r_spin.setValue(comp.params.get('r', 12.5) if not isinstance(comp, TestTarget) else comp.params.get('size', 5.0))
                r_spin.valueChanged.connect(lambda v: self.apply_param('r' if not isinstance(comp, TestTarget) else 'size', v))

        # 3b. Aperture Shape
        if isinstance(comp, Aperture):
            sh_combo = self.app.add_combo_to_form(self.specific_form, "Shape", 
                                                 ["Circular", "Square", "Gaussian"], 
                                                 comp.params.get('shape', 'Circular'),
                                                 lambda v: self.apply_param('shape', v))

        # 4. Grating Density & Preview Logic
        if isinstance(comp, Grating):
            p_combo = self.app.add_combo_to_form(self.specific_form, "Pattern", 
                                                 ["Linear Zebra", "Linear Cosine", "Crossed Zebra", "Crossed Cosine"], 
                                                 comp.params.get('pattern', 'Linear Cosine'),
                                                 lambda v: self.apply_param('pattern', v))

            d_spin = QDoubleSpinBox()
            d_spin.setRange(1.0, 5000.0)
            d_spin.setSuffix(" lines/mm")
            self.specific_form.addRow("Line Density", d_spin)
            d_spin.setValue(comp.params.get('line_density', 300))
            d_spin.valueChanged.connect(lambda v: self.apply_param('line_density', v))

            ord_spin = QSpinBox()
            ord_spin.setRange(0, 10)
            ord_spin.setValue(comp.params.get('n_orders', 2))
            ord_spin.valueChanged.connect(lambda v: self.apply_param('n_orders', v))
            self.specific_form.addRow("Visible Orders (m):", ord_spin)

            rpo_spin = QSpinBox()
            rpo_spin.setRange(1, 100)
            rpo_spin.setValue(comp.params.get('rays_per_order', 9))
            rpo_spin.valueChanged.connect(lambda v: self.apply_param('rays_per_order', v))
            self.specific_form.addRow("Rays Per Order:", rpo_spin)

        
        # 5. Detector & Target specifics (Size)
        if isinstance(comp, (Detector, TestTarget)):
            if isinstance(comp, Detector):
                s_spin = QDoubleSpinBox()
                s_spin.setRange(0.1, 500.0)
                s_spin.setSuffix(" mm")
                s_spin.setValue(comp.params.get('size', 10.0))
                s_spin.valueChanged.connect(lambda v: self.apply_param('size', v))
                self.specific_form.addRow("Detector Size", s_spin)

                log_check = QCheckBox("Logarithmic Intensity Scale")
                log_check.setStyleSheet("color: #3d5afe; font-weight: bold;")
                log_check.setChecked(comp.params.get('log_scale', False))
                log_check.toggled.connect(lambda v: self.apply_param('log_scale', v))
                self.specific_form.addRow(log_check)
                
                save_btn = QPushButton("Save Intensity Snapshot")
                save_btn.setStyleSheet("background: #2e7d32;")
                save_btn.clicked.connect(lambda: self.save_detector_snapshot(comp))
                self.specific_form.addRow(save_btn)

        del_btn = QPushButton("Delete Component")
        del_btn.setStyleSheet("background: #f44336;")
        del_btn.clicked.connect(self.app.delete_selected)
        self.specific_form.addRow(del_btn)
        
        # Update Sensor View if detector
        if isinstance(comp, Detector):
            self.update_detector_view(comp)
        else:
            self.sensor_label.hide()
            
        self._syncing = False

    def update_detector_view(self, comp):
        if not self.sensor_label: return
        self.sensor_label.show()
        self.sensor_label.setText("ANALYZING...")
        
        # Optimization: Use fixed sensor size for placeholder
        # Start background analysis
        if hasattr(self, '_current_worker') and self._current_worker:
            self._current_worker.cancel()
            
        self._current_worker = AnalysisWorker(self.app.system, comp)
        self._current_worker.signals.finished.connect(self.on_analysis_done)
        QThreadPool.globalInstance().start(self._current_worker)

    def on_analysis_done(self, result):
        self._current_worker = None
        noise_arr, report_text = result
        # Result received from background thread
        h, w = noise_arr.shape
        
        # 1. Get wavelength color (using pre-existing unified function)
        src = next((c for c in self.app.system.components if isinstance(c, PointSource)), None)
        wvl = src.params.get('wavelength', 532.0) if src else 532.0
        color = wavelength_to_color(wvl)
        r_m, g_m, b_m = color.red(), color.green(), color.blue()
        
        # 2. Colorize: Greyscale -> RGB
        rgb_data = np.zeros((h, w, 3), dtype=np.uint8)
        norm_noise = noise_arr.astype(float) / 255.0
        rgb_data[..., 0] = (norm_noise * r_m).astype(np.uint8)
        rgb_data[..., 1] = (norm_noise * g_m).astype(np.uint8)
        rgb_data[..., 2] = (norm_noise * b_m).astype(np.uint8)
        
        # 3. Create QImage (must copy because numpy array will be GC'd)
        qimg = QImage(rgb_data.data, w, h, w*3, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)
        
        # Scale to fit fixed sensor label size (280x280)
        scaled_pixmap = pixmap.scaled(280, 280, Qt.AspectRatioMode.KeepAspectRatio)
        self.sensor_label.setPixmap(scaled_pixmap)
        
        # Store latest result for saving (full res)
        self.last_img_pixmap = pixmap

    def save_detector_snapshot(self, comp):
        if not hasattr(self, 'last_img_pixmap') or self.last_img_pixmap.isNull():
            return
        
        size = comp.params.get('size', 10.0)
        import datetime
        ts = datetime.datetime.now().strftime("%H%M%S")
        filename = f"detector_{size}mm_{ts}.png"
        
        path, _ = QFileDialog.getSaveFileName(self, "Save Detector Snapshot", filename, "Images (*.png)")
        if path:
            # Use background worker for saving
            worker = SaveWorker(self.last_img_pixmap, path)
            QThreadPool.globalInstance().start(worker)
            logger.info(f"UI: Background save started for {path}")

    def apply_changes(self):
        items = self.app.scene.selectedItems()
        if not items or self._syncing: return
        
        self._syncing = True
        vcomp = items[0]
        vcomp.component.x = self.x_spin.value()
        vcomp.component.y = self.y_spin.value()
        vcomp.component.angle = self.angle_spin.value()
        vcomp.refresh_pos()
        self.app.update_rays()
        
        if isinstance(vcomp.component, Detector):
            self.update_detector_view(vcomp.component)
            
        self._syncing = False

    def apply_param(self, key, val):
        items = self.app.scene.selectedItems()
        if not items: return
        comp = items[0].component
        comp.params[key] = val
        logger.info(f"UI: Property '{key}' updated for {comp.uid} to {val}")
        
        # 1. Update rays and reflections
        self.app.update_rays()
        
        # 2. Trigger analysis refresh if we're dealing with a Detector
        # OR if any parameter change might affect the current detector's view
        if isinstance(comp, Detector):
            self.update_detector_view(comp)
            # If size changed, we also need to redraw its shape in the GUI
            items[0].update() # Force redraw of the visual component
        else:
            # Check if there's an active detector being viewed that needs refresh
            # because some other component (like a lens) changed
            current_selection = self.app.scene.selectedItems()
            if current_selection and isinstance(current_selection[0].component, Detector):
                self.update_detector_view(current_selection[0].component)

class ZoomableView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def drawBackground(self, painter, rect):
        # Draw Dark Grid
        painter.fillRect(rect, QColor(25, 25, 25))
        
        pen = QPen(QColor(40, 40, 40), 1)
        painter.setPen(pen)
        
        # Minor Grid (10mm = 50px)
        left = int(rect.left()) - (int(rect.left()) % 50)
        top = int(rect.top()) - (int(rect.top()) % 50)
        
        for x in range(left, int(rect.right()), 50):
            painter.drawLine(x, int(rect.top()), x, int(rect.bottom()))
        for y in range(top, int(rect.bottom()), 50):
            painter.drawLine(int(rect.left()), y, int(rect.right()), y)
            
        # Major Grid (100mm = 500px)
        pen.setColor(QColor(60, 60, 60))
        pen.setWidth(2)
        painter.setPen(pen)
        
        left_m = int(rect.left()) - (int(rect.left()) % 500)
        top_m = int(rect.top()) - (int(rect.top()) % 500)
        
        for x in range(left_m, int(rect.right()), 500):
            painter.drawLine(x, int(rect.top()), x, int(rect.bottom()))
        for y in range(top_m, int(rect.bottom()), 500):
            painter.drawLine(int(rect.left()), y, int(rect.right()), y)

        # Draw Dynamic Optical Axis
        axis_pts = self.scene().app.system.get_axis_path()
        if len(axis_pts) > 1:
            pen.setColor(QColor(61, 90, 254, 150))
            pen.setWidth(1)
            pen.setDashPattern([10, 5])
            painter.setPen(pen)
            for i in range(len(axis_pts)-1):
                p1 = QPointF(axis_pts[i][0]*5.0, axis_pts[i][1]*5.0)
                p2 = QPointF(axis_pts[i+1][0]*5.0, axis_pts[i+1][1]*5.0)
                painter.drawLine(p1, p2)

    def wheelEvent(self, event):
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor
        if event.angleDelta().y() > 0: self.scale(zoom_in_factor, zoom_in_factor)
        else: self.scale(zoom_out_factor, zoom_out_factor)

class SimulatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stockholm")
        self.resize(1400, 900)
        self.setStyleSheet(DARK_THEME)
        
        self.system = OpticalSystem()
        self.scene_manager = SceneManager()
        self.scene_manager.system = self.system
        self.snapping_enabled = True
        
        self.scene = QGraphicsScene(-1000, -1000, 2000, 2000)
        self.scene.app = self
        self.scene.selectionChanged.connect(self.on_selection_changed)
        
        self.view = ZoomableView(self.scene)
        self.setCentralWidget(self.view)
        
        # Side Dock
        self.dock = QDockWidget("Engineering View", self)
        self.dock.setMinimumWidth(320)
        self.props_panel = PropertyPanel(self)
        self.dock.setWidget(self.props_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock)

        self.setup_ui()
        self.load_default_scene()
        self.wave_engine = WaveEngine(res=256, size=0.03) # Real-time Wave Core

    def setup_ui(self):
        toolbar = QToolBar("Toolbox")
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)
        
        def add_action(text, icon_text, cb):
            a = QAction(f"{icon_text} {text}", self)
            a.triggered.connect(cb)
            toolbar.addAction(a)
            return a

        # Components
        add_action("Lens", "👓", lambda: self.add_comp(Lens(0, 0, 0)))
        add_action("Mirror", "🪞", lambda: self.add_comp(Mirror(50, 0, 45)))
        add_action("Grating", "🏁", lambda: self.add_comp(Grating(50, 0, 0)))
        add_action("Target", "🅰", lambda: self.add_comp(TestTarget(0, 0, 0)))
        add_action("Aperture", "⭕", lambda: self.add_comp(Aperture(50, 0, 0)))
        add_action("High-Pass", "🔴", lambda: self.add_comp(HighPassFilter(50, 0, 0)))
        add_action("Fan Src", "☀️", lambda: self.add_comp(PointSource(-80, 0, 0)))
        add_action("Detector", "📹", lambda: self.add_comp(Detector(120, 0, 0)))
        
        toolbar.addSeparator()
        add_action("Save", "💾", self.save_scene)
        add_action("Load", "📂", self.load_scene)


    def on_selection_changed(self):
        self.props_panel.sync_data()
        for item in self.scene.items():
            if isinstance(item, VisualComponent):
                item.handle.setVisible(item.isSelected())

    def load_default_scene(self):
        default_path = os.path.join(os.getcwd(), "starting_setup.json")
        if os.path.exists(default_path):
            self.scene_manager.load(default_path)
            # Clear and rebuild scene
            self.scene.clear()
            for comp in self.system.components:
                self.scene.addItem(VisualComponent(comp, self))
            self.update_rays()
            logger.info("UI: Loaded default scene from starting_setup.json")
        else:
            self.init_demo()

    def init_demo(self):
        # Collimator Setup (Default)
        # Point source at [-0.8, 0]
        # Lens at [-0.3, 0] with f=0.5
        # Detector at [0.5, 0]
        src = PointSource(-100.0, 0.0, 0)
        src.params["angle_range"] = 0.4
        
        l1 = Lens(-30, 0.0, 0)
        l1.params["f"] = 50.0
        l1.params["r"] = 25.0
        
        det = Detector(80.0, 0.0, 0)
        det.params["r"] = 30.0
        
        for c in [src, l1, det]:
            self.add_comp(c)

    def add_comp(self, comp):
        self.system.components.append(comp)
        vcomp = VisualComponent(comp, self)
        self.scene.addItem(vcomp)
        self.update_rays()

    def delete_selected(self):
        items = self.scene.selectedItems()
        for item in items:
            if isinstance(item, VisualComponent):
                self.system.components.remove(item.component)
                self.scene.removeItem(item)
                logger.info(f"UI: Component {item.component.uid} deleted.")
        self.update_rays()
        self.props_panel.sync_data()

    def update_rays(self):
        self.system.update_rays()
        
        # Fast Clear
        if not hasattr(self, 'ray_items'): self.ray_items = []
        for item in self.ray_items:
            try: self.scene.removeItem(item)
            except: pass
        self.ray_items.clear()
        
        # Premium Neon Rays
        for ray in self.system.rays:
            path = QPainterPath()
            if len(ray.points) > 1:
                pts = ray.points
                path.moveTo(pts[0][0]*5.0, pts[0][1]*5.0)
                for p in pts[1:]:
                    path.lineTo(p[0]*5.0, p[1]*5.0)
                
                color = wavelength_to_color(ray.wavelength)
                
                # Outer glow
                glow_pen = QPen(color, 4)
                glow_color = QColor(color)
                glow_color.setAlpha(30)
                glow_pen.setColor(glow_color)
                glow = self.scene.addPath(path, glow_pen)
                glow.setZValue(-1)
                self.ray_items.append(glow)
                
                # Core beam
                core = self.scene.addPath(path, QPen(color, 1.5))
                core.setZValue(-1)
                self.ray_items.append(core)

    def save_scene(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Scene", "", "JSON Files (*.json)")
        if path: self.scene_manager.save(path)

    def load_scene(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Scene", "", "JSON Files (*.json)")
        if path:
            # Important: Clear rays BEFORE scene items to avoid dangling pointers
            for item in self.ray_items:
                if item.scene():
                    try: self.scene.removeItem(item)
                    except: pass
            self.ray_items.clear()
            
            self.scene_manager.load(path)
            self.scene.clear()
            for comp in self.system.components:
                self.scene.addItem(VisualComponent(comp, self))
            self.update_rays()

    def add_combo_to_form(self, form, label, items, current_val, callback):
        from PyQt6.QtWidgets import QComboBox
        combo = QComboBox()
        combo.addItems(items)
        combo.setCurrentText(current_val)
        combo.currentTextChanged.connect(callback)
        form.addRow(label, combo)
        return combo

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            self.delete_selected()
        super().keyPressEvent(event)

def wavelength_to_color(wavelength):
    """Approximate wavelength to QColor."""
    # Simple RGB approximation for 380-780nm
    if 380 <= wavelength < 440: r, g, b = (440 - wavelength) / (440 - 380), 0.0, 1.0
    elif 440 <= wavelength < 490: r, g, b = 0.0, (wavelength - 440) / (490 - 440), 1.0
    elif 490 <= wavelength < 510: r, g, b = 0.0, 1.0, (510 - wavelength) / (510 - 490)
    elif 510 <= wavelength < 580: r, g, b = (wavelength - 510) / (580 - 510), 1.0, 0.0
    elif 580 <= wavelength < 645: r, g, b = 1.0, (645 - wavelength) / (645 - 580), 0.0
    elif 645 <= wavelength <= 780: r, g, b = 1.0, 0.0, 0.0
    else: r, g, b = 0.5, 0.5, 0.5
    return QColor(int(r*255), int(g*255), int(b*255))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Inter", 9))
    win = SimulatorApp()
    win.show()
    sys.exit(app.exec())

