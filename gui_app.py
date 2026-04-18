import sys
import os
import uuid
from PyQt6.QtWidgets import (QApplication, QMainWindow, QGraphicsView, 
                             QGraphicsScene, QGraphicsItem, QToolBar, 
                             QVBoxLayout, QWidget, QDoubleSpinBox, 
                             QLabel, QFormLayout, QPushButton, QFileDialog,
                             QGraphicsPathItem, QMenu, QHBoxLayout, QFrame,
                             QScrollArea, QDockWidget, QCheckBox, QGroupBox, QSpinBox)
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, QEvent
from PyQt6.QtGui import QPen, QBrush, QColor, QAction, QPainterPath, QTransform, QPainter, QFont, QPixmap, QRadialGradient, QImage

import numpy as np

# Local imports
from optics_engine import Ray, OpticalSystem
from components import *
from persistence import SceneManager
from wave_engine import WaveEngine
from scipy.fft import fft2, fftshift

import logging
logger = logging.getLogger("SimulatorApp")
logger.setLevel(logging.WARNING)

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
        self.setPos(self.component.x * 100, self.component.y * 100)
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
        return QRectF(-25, -50, 50, 100)

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
        r = self.component.params.get("r", 0.2) * 100 # Half-width in pixels
        
        if isinstance(self.component, Lens):
            f = self.component.params.get("f", 0.1)
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
        elif isinstance(self.component, PointSource):
            painter.setBrush(QBrush(QColor(255, 50, 50)))
            painter.drawEllipse(-6, -6, 12, 12)
            # Directional line
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            painter.drawLine(0, 0, 10, 0)
        elif isinstance(self.component, BeamSource):
            painter.setBrush(QBrush(QColor(255, 50, 50, 150)))
            w_px = self.component.params.get('width', 0.1) * 100
            painter.drawRect(-3, -w_px/2, 6, w_px)
            # Directional arrow
            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            painter.drawLine(0, 0, 10, 0)
            painter.drawLine(10, 0, 7, -3)
            painter.drawLine(10, 0, 7, 3)
        elif isinstance(self.component, Aperture):
            # Aperture: Two blocks with a gap of 2*r
            painter.setPen(QPen(QColor(150, 150, 150), 4))
            gap_i = int(round(r))
            painter.drawLine(0, -100, 0, -gap_i) # Outer part
            painter.drawLine(0, gap_i, 0, 100)
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
        elif isinstance(self.component, HighPassFilter):
            # High-Pass Filter: A central blocking disk (DC Block)
            painter.setPen(QPen(QColor(255, 100, 100), 2))
            painter.setBrush(QBrush(QColor(255, 50, 50, 150)))
            # Draw as a solid circle in the center
            ri = int(round(r)) 
            painter.drawEllipse(-ri, -ri, 2*ri, 2*ri)
        elif isinstance(self.component, Detector):
            # Scientific Camera Shape: Body + Sensor housing
            painter.setPen(QPen(QColor(100, 116, 139), 2)) # Blue-gray metallic
            ri = int(round(r))
            # Camera Body (Back)
            painter.setBrush(QBrush(QColor(30, 41, 59)))
            painter.drawRect(-12, -ri, 6, 2*ri)
            # Sensor Front Plate
            painter.setBrush(QBrush(QColor(51, 65, 85)))
            painter.drawRect(-6, -ri, 12, 2*ri)
            # Sensor Slot (Visual line)
            painter.setPen(QPen(QColor(100, 255, 100, 150), 1))
            painter.drawLine(0, -ri+5, 0, ri-5)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.scene():
            if getattr(self, '_syncing_handle', False):
                return value
            
            new_pos = value
            
            # Dynamic Optical Axis Snapping
            if self.app.snapping_enabled:
                grid = 5.0 # 0.05m = 5px
                new_pos.setX(round(new_pos.x() / grid) * grid)
                new_pos.setY(round(new_pos.y() / grid) * grid)
                
                axis_pts = self.app.system.get_axis_path()
                best_snap = None
                best_dist = 25 # 25px snapping radius
                
                # Check each segment of the optical axis path
                for i in range(len(axis_pts)-1):
                    p1 = np.array(axis_pts[i]) * 100
                    p2 = np.array(axis_pts[i+1]) * 100
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

            self.component.x = new_pos.x() / 100.0
            self.component.y = new_pos.y() / 100.0
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
            s.setRange(-5, 5)
            s.setSingleStep(0.01)
            s.setSuffix(" m")
        
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
        self.sensor_label.setStyleSheet("background: #000; border: 1px solid #444; border-radius: 4px; margin-top: 10px;")
        self.sensor_label.setFixedSize(240, 160)
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
        label_text = "Emission Direction" if isinstance(comp, (PointSource, BeamSource)) else "Rotation"
        self.form.labelForField(self.angle_spin).setText(label_text)
        
        # Clear specific form
        while self.specific_form.count():
            child = self.specific_form.takeAt(0)
            if child.widget(): child.widget().deleteLater()
            
        # 1. Lens specifics
        if isinstance(comp, Lens):
            f_spin = QDoubleSpinBox()
            f_spin.setRange(-2, 2)
            f_spin.setValue(comp.params.get('f', 0.5))
            f_spin.valueChanged.connect(lambda v: self.apply_param('f', v))
            self.specific_form.addRow("Focal Length", f_spin)
        
        # 2. Source specifics (Rays, Width, Angle)
        if isinstance(comp, (PointSource, BeamSource)):
            nr_spin = QDoubleSpinBox()
            nr_spin.setRange(1, 100)
            nr_spin.setDecimals(0)
            nr_spin.setValue(comp.params.get('n_rays', 21))
            nr_spin.valueChanged.connect(lambda v: self.apply_param('n_rays', int(v)))
            self.specific_form.addRow("Num Rays", nr_spin)
            if isinstance(comp, BeamSource):
                w_spin = QDoubleSpinBox()
                w_spin.setRange(0.01, 1.0)
                w_spin.setValue(comp.params.get('width', 0.1))
                w_spin.valueChanged.connect(lambda v: self.apply_param('width', v))
                self.specific_form.addRow("Beam Width", w_spin)
            else:
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
        if isinstance(comp, (Lens, Mirror, Detector, Aperture, Grating, HighPassFilter)):
            r_spin = QDoubleSpinBox()
            r_spin.setRange(0.001, 2.0)
            r_spin.setSuffix(" m")
            label = "Opening (R)" if isinstance(comp, Aperture) else "Radius (R)"
            self.specific_form.addRow(label, r_spin)
            r_spin.setValue(comp.params.get('r', 0.2))
            r_spin.valueChanged.connect(lambda v: self.apply_param('r', v))

        # 4. Grating Density & Preview Logic
        if isinstance(comp, Grating):
            p_combo = self.app.add_combo_to_form(self.specific_form, "Pattern", 
                                                 ["Linear", "Crossed", "Chessboard"], 
                                                 comp.params.get('pattern', 'Linear'),
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

        
        # 5. Detector specifics (Spectral View)
        if isinstance(comp, Detector):
            spec_check = QCheckBox("Spectral Analysis (Fourier)")
            spec_check.setStyleSheet("color: #3d5afe; font-weight: bold;")
            spec_check.setChecked(comp.params.get('spectral_view', False))
            spec_check.toggled.connect(lambda v: self.apply_param('spectral_view', v))
            self.specific_form.addRow(spec_check)

        del_btn = QPushButton("Delete Component")
        del_btn.setStyleSheet("background: #f44336;")
        del_btn.clicked.connect(self.app.delete_selected)
        self.specific_form.addRow(del_btn)
        
        # Update Sensor View if detector (Now dummy)
        if isinstance(comp, Detector):
            self.update_detector_view(comp)
            # TRIGGER ANALYSIS PRINT TO TERMINAL
            report = self.app.system.analyze_system()
            print(report)
        else:
            self.sensor_label.hide()
            
        self._syncing = False

    def update_detector_view(self, comp):
        if not self.sensor_label: return
        self.sensor_label.show()
        
        # Purely black sensor area as per request
        W, H = 240, 160
        canvas = QPixmap(W, H)
        canvas.fill(QColor(0, 0, 0))
        painter = QPainter(canvas)
        painter.setPen(QColor(50, 50, 50))
        painter.drawText(10, 80, "Dummy Sensor - Check Terminal")
        painter.end()
        self.sensor_label.setPixmap(canvas)
        self.sensor_label.setPixmap(canvas)

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
        self._syncing = False

    def apply_param(self, key, val):
        items = self.app.scene.selectedItems()
        if not items: return
        comp = items[0].component
        comp.params[key] = val
        logger.info(f"UI: Property '{key}' updated for {comp.uid} to {val}")
        self.app.update_rays()
        
        # Trigger analysis print if it's a detector being modified
        if isinstance(comp, Detector):
            print(self.app.system.analyze_system())

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
        
        # Minor Grid (10cm = 10px)
        left = int(rect.left()) - (int(rect.left()) % 10)
        top = int(rect.top()) - (int(rect.top()) % 10)
        
        for x in range(left, int(rect.right()), 10):
            painter.drawLine(x, int(rect.top()), x, int(rect.bottom()))
        for y in range(top, int(rect.bottom()), 10):
            painter.drawLine(int(rect.left()), y, int(rect.right()), y)
            
        # Major Grid (1m = 100px)
        pen.setColor(QColor(60, 60, 60))
        pen.setWidth(2)
        painter.setPen(pen)
        
        left_m = int(rect.left()) - (int(rect.left()) % 100)
        top_m = int(rect.top()) - (int(rect.top()) % 100)
        
        for x in range(left_m, int(rect.right()), 100):
            painter.drawLine(x, int(rect.top()), x, int(rect.bottom()))
        for y in range(top_m, int(rect.bottom()), 100):
            painter.drawLine(int(rect.left()), y, int(rect.right()), y)

        # Draw Dynamic Optical Axis
        axis_pts = self.scene().app.system.get_axis_path()
        if len(axis_pts) > 1:
            pen.setColor(QColor(61, 90, 254, 150))
            pen.setWidth(1)
            pen.setDashPattern([10, 5])
            painter.setPen(pen)
            for i in range(len(axis_pts)-1):
                p1 = QPointF(axis_pts[i][0]*100, axis_pts[i][1]*100)
                p2 = QPointF(axis_pts[i+1][0]*100, axis_pts[i+1][1]*100)
                painter.drawLine(p1, p2)

    def wheelEvent(self, event):
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor
        if event.angleDelta().y() > 0: self.scale(zoom_in_factor, zoom_in_factor)
        else: self.scale(zoom_out_factor, zoom_out_factor)

class SimulatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optics Studio")
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
        self.props_panel = PropertyPanel(self)
        self.dock.setWidget(self.props_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock)

        self.setup_ui()
        self.init_demo()
        self.wave_engine = WaveEngine(res=256, size=0.03) # Real-time Wave Core

    def setup_ui(self):
        # ...
        toolbar = QToolBar("Toolbox")
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)
        
        def add_action(text, icon_text, cb):
            a = QAction(text, self)
            a.triggered.connect(cb)
            toolbar.addAction(a)
            return a

        add_action("Lens", "L", lambda: self.add_comp(Lens(0, 0, 0)))
        add_action("Mirror", "M", lambda: self.add_comp(Mirror(0.5, 0, 45)))
        add_action("Grating", "G", lambda: self.add_comp(Grating(0.5, 0, 0)))
        add_action("High-Pass", "HP", lambda: self.add_comp(HighPassFilter(0.5, 0, 0)))
        add_action("Fan Src", "S", lambda: self.add_comp(PointSource(-0.8, 0, 0)))
        add_action("Beam Src", "B", lambda: self.add_comp(BeamSource(-0.8, -0.2, 0)))
        add_action("Aperture", "A", lambda: self.add_comp(Aperture(0.5, 0, 0)))
        add_action("Detector", "D", lambda: self.add_comp(Detector(1.2, 0, 0)))
        
        toolbar.addSeparator()
        self.snap_act = QAction("Snapping ON", self)
        self.snap_act.setCheckable(True)
        self.snap_act.setChecked(True)
        self.snap_act.toggled.connect(self.toggle_snap)
        toolbar.addAction(self.snap_act)
        
        toolbar.addSeparator()
        add_action("Save", "💾", self.save_scene)
        add_action("Load", "📂", self.load_scene)

    def toggle_snap(self, enabled):
        self.snapping_enabled = enabled
        self.snap_act.setText("Snapping ON" if enabled else "Snapping OFF")

    def on_selection_changed(self):
        self.props_panel.sync_data()
        for item in self.scene.items():
            if isinstance(item, VisualComponent):
                item.handle.setVisible(item.isSelected())

    def init_demo(self):
        # Collimator Setup (Default)
        # Point source at [-0.8, 0]
        # Lens at [-0.3, 0] with f=0.5
        # Detector at [0.5, 0]
        src = PointSource(-0.8, 0.0, 0)
        src.params["angle_range"] = 0.4
        
        l1 = Lens(-0.3, 0.0, 0)
        l1.params["f"] = 0.5
        l1.params["r"] = 0.25
        
        det = Detector(0.8, 0.0, 0)
        det.params["r"] = 0.3
        
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
                path.moveTo(pts[0][0]*100, pts[0][1]*100)
                for p in pts[1:]:
                    path.lineTo(p[0]*100, p[1]*100)
                
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

