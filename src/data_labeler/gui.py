"""PyQt5 based GUI classes for manual data labeling"""

__author__      = "Veronika Kalouskova"
__copyright__   = "Copyright 2025, FBMI CVUT"

import math 
import sys

import matplotlib as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from PyQt5 import QtWidgets, QtCore
plt.use('Qt5Agg')

from scipy.signal import find_peaks

PADDING = 1000

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=8, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.subplots_adjust(0, 0, 1, 1)
        
        self.axes = fig.add_subplot(111)
        self.axes.tick_params(left = False, labelleft = False, labelbottom = False, bottom = True, direction='in', color='blue') 

        for spine in self.axes.spines.values():
            spine.set_edgecolor('blue')

        super(MplCanvas, self).__init__(fig)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, app, dh, fs, seg_len, seg_num, filename):
        self.dh = dh
        self.DATA = self.dh.df_in
        self.FS = fs
        self.SEG_LEN = seg_len
        self.seg_curr = seg_num

        super(MainWindow, self).__init__()

        self.setWindowTitle(filename)
        self.set_styles(app)
        self.create_layout()
    
    #   Keypress eventhandler for navigating the ECG and respiratory signals
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_W or event.key() == QtCore.Qt.Key_D:
            self.on_next()
        elif event.key() == QtCore.Qt.Key_S or event.key() == QtCore.Qt.Key_A:
            self.on_prev()
        elif event.key() == QtCore.Qt.Key_1:
            self.on_toggle(self.radio_ref_yes, self.radio_ref_no)
        elif event.key() == QtCore.Qt.Key_2:
            self.on_toggle(self.radio_sig1_yes, self.radio_sig1_no)
        elif event.key() == QtCore.Qt.Key_3:
            self.on_toggle(self.radio_sig2_yes, self.radio_sig2_no)
        elif event.key() == QtCore.Qt.Key_Space:
            self.on_toggle(self.radio_seg_yes, self.radio_seg_no)
        elif event.key() == QtCore.Qt.Key_Enter or event.key() == QtCore.Qt.Key_Return:
            sys.exit()

        event.accept()

    #   Set layout styles
    def set_styles(self, app):
        app.setStyleSheet('QLabel{color: #404040;} QPushButton{color: #404040;} QRadioButton{color: #404040;}')

    #   Draw grid separator line
    def draw_line(self, x, y, layout):
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Raised)
        layout.addWidget(line, x, y, 2, 2)

    #   Define layout elements
    def create_layout(self):
        width, height = 12, 5
        layout = QtWidgets.QGridLayout()

        # Matplotlib canvas for the ECG signal
        self.canvas_ecg = MplCanvas(self, width, height)
        self.line_ecg, = self.canvas_ecg.axes.plot([], [], linewidth=0.8, color='red')

        # Flip ECG signal axes upside down
        self.line_ecg.axes.set_ylim((max(self.DATA['ECG']) + PADDING, min(self.DATA['ECG']) - PADDING))
        self.canvas_ecg.axes.text(0.01, 0.95, 'ECG', ha='left', va='top', color='blue', fontsize = 10, transform=self.canvas_ecg.axes.transAxes)

        layout.addWidget(self.canvas_ecg, 1, 0, 10, 10)

        # Matplotlib canvas for the Th reference signal
        self.canvas_th_ref = MplCanvas(self, width, height)
        self.line_th_ref, = self.canvas_th_ref.axes.plot([], [], linewidth=0.8, color='grey')
        self.line_th_ref_filt, = self.canvas_th_ref.axes.plot([], [], linewidth=0.8, color='blue')

        self.canvas_th_ref.axes.text(0.01, 0.95, 'Th Reference', ha='left', va='top', color='blue', fontsize = 10, transform=self.canvas_th_ref.axes.transAxes)
        layout.addWidget(self.canvas_th_ref, 11, 0, 10, 10)

        # Matplotlib canvas for the Th 1 signal
        self.canvas_th_1 = MplCanvas(self, width, height)
        self.line_th_1, = self.canvas_th_1.axes.plot([], [], linewidth=0.8, color='blue')

        self.canvas_th_1.axes.text(0.01, 0.95, 'Th 1 Filtered', ha='left', va='top', color='blue', fontsize = 10, transform=self.canvas_th_1.axes.transAxes)
        layout.addWidget(self.canvas_th_1, 21, 0, 10, 10)

        # Variables for Th1 semi-automatic annotation
        self.neighborhood_patches_th1 = []
        self.interval_vertical_stripes_th1 = []
        self.correct_dips_th1, = self.canvas_th_1.axes.plot([], [], marker='o', linestyle='None', color='green', markersize=7.5, zorder=50)
        self.dangerous_dips_th1, = self.canvas_th_1.axes.plot([], [], marker='X', linestyle='None', color='red', markersize=10, zorder=100)

        # Matplotlib canvas for the Th 2 signal
        self.canvas_th_2 = MplCanvas(self, width, height)
        self.line_th_2, = self.canvas_th_2.axes.plot([], [], linewidth=0.8, color='blue')

        self.canvas_th_2.axes.text(0.01, 0.95, 'Th 2 Filtered', ha='left', va='top', color='blue', fontsize = 10, transform=self.canvas_th_2.axes.transAxes)
        layout.addWidget(self.canvas_th_2, 31, 0, 10, 10)

        # Buttons for navigating the signal
        pushbutton = QtWidgets.QPushButton('<')
        pushbutton.clicked.connect(self.on_prev)
        pushbutton.setFixedSize(80, 30)
        layout.addWidget(pushbutton, 41, 0)

        pushbutton = QtWidgets.QPushButton('>')
        pushbutton.clicked.connect(self.on_next)
        pushbutton.setFixedSize(80, 30)
        layout.addWidget(pushbutton, 41, 9)

        # Grid separators
        self.draw_line(0, 11, layout)
        self.draw_line(10, 11, layout)
        self.draw_line(20, 11, layout)
        self.draw_line(30, 11, layout)
        self.draw_line(40, 11, layout)

        # -------------- Segment --------------
        # Segment label
        self.seg_label = QtWidgets.QLabel()
        self.seg_label.setFixedWidth(150)
        layout.addWidget(self.seg_label, 2, 11, 2, 2)

        self.label = QtWidgets.QLabel('Confident:')
        layout.addWidget(self.label, 4, 11, 2, 2)

        # Segment toggle buttons
        self.radio_seg_yes = QtWidgets.QRadioButton('Yes')
        self.radio_seg_yes.label = 1
        self.radio_seg_yes.toggled.connect(lambda checked: self.on_click(checked, 'Confidence'))
        layout.addWidget(self.radio_seg_yes, 6, 11, 2, 1)

        self.radio_seg_no = QtWidgets.QRadioButton('No')
        self.radio_seg_no.label = 0
        self.radio_seg_no.toggled.connect(lambda checked: self.on_click(checked, 'Confidence'))
        layout.addWidget(self.radio_seg_no, 6, 12, 2, 1)

        # Create a group for the segment set
        self.group_seg = QtWidgets.QButtonGroup()
        self.group_seg.addButton(self.radio_seg_yes)
        self.group_seg.addButton(self.radio_seg_no)
        # -------------------------------------
        
        # ------------- Reference ------------- 
        # Signal label
        self.label = QtWidgets.QLabel('Reference visible:')
        layout.addWidget(self.label, 12, 11, 2, 2)

        # Signal toggle buttons
        self.radio_ref_yes = QtWidgets.QRadioButton('Yes')
        self.radio_ref_yes.label = 1
        self.radio_ref_yes.toggled.connect(lambda checked: self.on_click(checked, 'Reference'))
        layout.addWidget(self.radio_ref_yes, 14, 11, 2, 1)

        self.radio_ref_no = QtWidgets.QRadioButton('No')
        self.radio_ref_no.label = 0
        self.radio_ref_no.toggled.connect(lambda checked: self.on_click(checked, 'Reference'))
        layout.addWidget(self.radio_ref_no, 14, 12, 2, 1)

        # Create a group for the reference set
        self.group_ref = QtWidgets.QButtonGroup()
        self.group_ref.addButton(self.radio_ref_yes)
        self.group_ref.addButton(self.radio_ref_no)
        # ------------------------------------- 

        # ------------- Signal 1 --------------
        # Signal label
        self.label = QtWidgets.QLabel('Signal visible:')
        layout.addWidget(self.label, 22, 11, 2, 2)

        # Signaltoggle buttons
        self.radio_sig1_yes = QtWidgets.QRadioButton('Yes')
        self.radio_sig1_yes.label = 1
        self.radio_sig1_yes.toggled.connect(lambda checked: self.on_click(checked, 'Signal_1'))
        layout.addWidget(self.radio_sig1_yes, 24, 11, 2, 1)

        self.radio_sig1_no = QtWidgets.QRadioButton('No')
        self.radio_sig1_no.label = 0
        self.radio_sig1_no.toggled.connect(lambda checked: self.on_click(checked, 'Signal_1'))
        layout.addWidget(self.radio_sig1_no, 24, 12, 2, 1)

        # Create a group for the signal 1 set
        self.group_sig1 = QtWidgets.QButtonGroup()
        self.group_sig1.addButton(self.radio_sig1_yes)
        self.group_sig1.addButton(self.radio_sig1_no)
        # ------------------------------------- 

        # ------------- Signal 2 --------------
        self.label = QtWidgets.QLabel('Signal visible:')
        layout.addWidget(self.label, 32, 11, 2, 2)

        # Signal toggle buttons
        self.radio_sig2_yes = QtWidgets.QRadioButton('Yes')
        self.radio_sig2_yes.label = 1
        self.radio_sig2_yes.toggled.connect(lambda checked: self.on_click(checked, 'Signal_2'))
        layout.addWidget(self.radio_sig2_yes, 34, 11, 2, 1)

        self.radio_sig2_no = QtWidgets.QRadioButton('No')
        self.radio_sig2_no.label = 0
        self.radio_sig2_no.toggled.connect(lambda checked: self.on_click(checked, 'Signal_2'))
        layout.addWidget(self.radio_sig2_no, 34, 12, 2, 1)

        # Create a group for the signal 2 set
        self.group_sig2 = QtWidgets.QButtonGroup()
        self.group_sig2.addButton(self.radio_sig2_yes)
        self.group_sig2.addButton(self.radio_sig2_no)
        # -------------------------------------

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.update_plot()
        self.show()

    #   Dynamic update of canvas limits of a segment
    def update_limits(self, axes, data, start, end, flip = False):
        y_min = min(data[start:end])
        y_max = max(data[start:end])

        # 20% padding
        padding = (y_max - y_min) * 0.2 

        if not flip:
            axes.set_ylim(y_min - padding, y_max + padding)
        else:
            axes.set_ylim(y_max + padding, y_min - padding)

    #   Update specific canvas plot
    def update_canvas(self, x, data, line, canvas, seg_start, start, end):
        line.set_data(x, data[start:end])

        # Only set limits once if they’re not changing
        line.axes.set_xlim((seg_start, seg_start + self.SEG_LEN))

        # Use blitting for faster redraws
        canvas.draw_idle()

    #   Update shading of expiratory and inspiratory phases
    def update_shading(self, x, start, end):
        # Mask for inspiration (1) and expiration (0)
        inspiration_mask = (self.DATA[start:end]['Phase'] == 1)
        expiration_mask = (self.DATA[start:end]['Phase'] == 0)

        #self.line_th_ref.axes.patches.clear()
        while self.line_th_ref.axes.patches: self.line_th_ref.axes.patches[0].remove()

        y_min, _ = self.line_th_ref.axes.get_ylim()

        # Plot shading for Inspiration (1)
        if np.any(inspiration_mask):  # Check if any inspiration phase is present
            self.line_th_ref.axes.fill_between(x,
                                               y1=y_min,
                                               y2=self.DATA['Th_Ref_Filt'][start:end], 
                                               where=inspiration_mask, 
                                               color='lightblue', 
                                               alpha=0.7, 
                                               edgecolor='none')

        # Plot shading for Expiration (0)
        if np.any(expiration_mask):  # Check if any expiration phase is present
            self.line_th_ref.axes.fill_between(x, 
                                               y1=y_min,
                                               y2=self.DATA['Th_Ref_Filt'][start:end], 
                                               where=expiration_mask, 
                                               color='mistyrose', 
                                               alpha=0.7,
                                               edgecolor='none')

    def update_th1_semi_auto_annotations(self, x, start, end, patch_height_percentage = 0.05, reference_separation = 2, separation = 1, colors = ['C0', 'C1', 'C2'], interval_range_falloff = .25, epsilon = 0.25):
        # --- SETUP ---
        ref_data_segment = self.DATA['Th_Ref_Filt'][start:end].to_numpy()
        if (len(ref_data_segment) == 0):
            n = 0
        else:
            ref_dips_indices, _ = find_peaks(-ref_data_segment, distance=self.FS*reference_separation)
            n = len(ref_dips_indices)

        data_segment = self.DATA['Th1_Filt'][start:end].to_numpy()
        if (len(data_segment) == 0):
            return
        
        minima_indices, _ = find_peaks(-data_segment, distance=self.FS*separation)
        if (len(minima_indices) == 0):
            return
        # --- END SETUP ---

        # --- PLOTTING REFERENCE DIP INTERVAL RANGES ---
        reference_minima_index_ranges = []
        for ref_dip_idx in ref_dips_indices:
            start_range = ref_dip_idx - int(self.FS * reference_separation * interval_range_falloff)
            if (start_range < 0):
                start_range = 0
            end_range = ref_dip_idx + int(self.FS * reference_separation * interval_range_falloff)
            if (end_range >= len(data_segment)):
                end_range = len(data_segment) - 1       

            reference_minima_index_ranges.append((start_range, end_range))

        while self.interval_vertical_stripes_th1:
            self.interval_vertical_stripes_th1.pop().remove()

        for reference_range in reference_minima_index_ranges:
            stripe = self.canvas_th_1.axes.axvspan(
                x[reference_range[0]],
                x[reference_range[1]],
                color='lightgrey',
                alpha=0.5,
                zorder=0
                )
            
            self.interval_vertical_stripes_th1.append(stripe)
        # --- END PLOTTING REFERENCE DIP INTERVAL RANGES ---

        # --- PLOTTING CRITICAL DIP POINTS ---
        epsilon_y = epsilon * (np.max(data_segment) - np.min(data_segment))

        correct_minima_indices = []
        dangerous_minima_indices = []

        smallest_minimum_y = np.min(data_segment[minima_indices])
        for idx in minima_indices:
            if (data_segment[idx] <= smallest_minimum_y + epsilon_y) and any(start_range <= idx <= end_range for (start_range, end_range) in reference_minima_index_ranges):
                correct_minima_indices.append(idx)
            else:
                dangerous_minima_indices.append(idx)            

        self.correct_dips_th1.set_data(x[correct_minima_indices], data_segment[correct_minima_indices])
        self.dangerous_dips_th1.set_data(x[dangerous_minima_indices], data_segment[dangerous_minima_indices])
        # --- END PLOTTING CRITICAL DIP POINTS ---

        # --- PLOTTING NEIGHBORHOOD PATCHES/BARS ---
        minima_values = data_segment[minima_indices]
        sorted_minima_indices = np.argsort(minima_values)

        while self.neighborhood_patches_th1:
            self.neighborhood_patches_th1.pop().remove()

        ymin, ymax = self.line_th_1.axes.get_ylim()
        patch_size = (ymax - ymin) * patch_height_percentage
        n_patches = min(n, len(sorted_minima_indices))
        for i, min_value in enumerate(minima_values[sorted_minima_indices[:n_patches]]):
            patch_start = min_value - patch_size / 2
            patch_end = min_value + patch_size / 2

            patch = self.canvas_th_1.axes.axhspan(
                patch_start,
                patch_end,
                color=colors[i%len(colors)],
                alpha=0.25,
                zorder=0
                )

            self.neighborhood_patches_th1.append(patch)
        # --- END PLOTTING NEIGHBORHOOD PATCHES/BARS ---

    #   Redraw all plots
    def update_plot(self):
        seg_len_pts = self.SEG_LEN * self.FS            # Length of a single segment in points
        seg_start = self.seg_curr * self.SEG_LEN        # Start of current segment in seconds
        
        start = self.seg_curr * seg_len_pts             # Start of current segment in points
        end = start + seg_len_pts                       # End of current segment in points

        x = np.linspace(seg_start, seg_start + self.SEG_LEN, seg_len_pts)

        # Update canvas for all signals
        self.update_canvas(x, self.DATA['ECG'], self.line_ecg, self.canvas_ecg, seg_start, start, end)

        self.update_canvas(x, self.DATA['Th_Ref'], self.line_th_ref, self.canvas_th_ref, seg_start, start, end)
        self.update_canvas(x, self.DATA['Th_Ref_Filt'], self.line_th_ref_filt, self.canvas_th_ref, seg_start, start, end)

        self.update_canvas(x, self.DATA['Th1_Filt'], self.line_th_1, self.canvas_th_1, seg_start, start, end)
        self.update_canvas(x, self.DATA['Th2_Filt'], self.line_th_2, self.canvas_th_2, seg_start, start, end)

        # Update axis limits
        self.update_limits(self.line_th_ref.axes, self.DATA['Th_Ref_Filt'], start, end, False)
        self.update_limits(self.line_th_1.axes, self.DATA['Th1_Filt'], start, end, False)
        self.update_limits(self.line_th_2.axes, self.DATA['Th2_Filt'], start, end, False)

        # Update Th1 semi-automatic annotation
        self.update_th1_semi_auto_annotations(x, start, end)

        # Update shading of expiratory and inspiratory phases
        self.update_shading(x, start, end)

        # Update segment label
        self.seg_label.setText('SEGMENT ' + str(self.seg_curr + 1) + ' / ' + str(math.floor(len(self.DATA) / seg_len_pts)))
        
        # Update selection of radio buttons
        self.update_selection(self.radio_seg_yes, self.radio_seg_no, self.dh.get_column_value(self.seg_curr, 'Confidence'))
        self.update_selection(self.radio_ref_yes, self.radio_ref_no, self.dh.get_column_value(self.seg_curr, 'Reference'))
        self.update_selection(self.radio_sig1_yes, self.radio_sig1_no, self.dh.get_column_value(self.seg_curr, 'Signal_1'))
        self.update_selection(self.radio_sig2_yes, self.radio_sig2_no, self.dh.get_column_value(self.seg_curr, 'Signal_2'))

     #   Update radio button selection based on data
    def update_selection(self, radio_yes, radio_no, data):
        if data == radio_yes.label:
            radio_yes.setChecked(True)
        elif data == radio_no.label:
            radio_no.setChecked(True)

    #   Handle radio button keyboard keypress
    def on_toggle(self, radio_yes, radio_no):
        if (radio_yes.isChecked()):
            radio_no.setChecked(True)
        elif (radio_no.isChecked()):
            radio_yes.setChecked(True)

    #   Handle radio button keypress
    def on_click(self, checked, column):
        if checked:
            radio_button = self.sender()
            if radio_button.isChecked():
                self.dh.set_column_value(self.seg_curr, radio_button.label, column)

    #   Handle next button keypress
    def on_next(self):
        if (self.seg_curr < math.floor(len(self.DATA) / (self.SEG_LEN * self.FS)) - 1):
            self.seg_curr += 1
            self.update_plot()

    #   Handle previous button keypress
    def on_prev(self):
        if (self.seg_curr > 0):
            self.seg_curr -= 1
            self.update_plot()


class Application():

    #   Run application loop
    def __init__(self, dh, fs, seg_len, seg_num, filename):
        app = QtWidgets.QApplication(sys.argv)
        win = MainWindow(app, dh, fs, seg_len, seg_num, filename)

        app.exec_()