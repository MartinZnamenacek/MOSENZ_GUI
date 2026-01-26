"""PyQt5 based GUI classes for manual data labeling"""

__author__      = "Veronika Kalouskova"
__copyright__   = "Copyright 2025, FBMI CVUT"

import math 
import sys
from tracemalloc import start

import matplotlib as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from PyQt5 import QtWidgets, QtCore
plt.use('Qt5Agg')

from scipy.signal import find_peaks

from enum import Enum
class AnnotationType(Enum):
    TH1, TH2 = 1, 2

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
        elif event.key() == QtCore.Qt.Key_C:
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

    def set_label_style(self, label, value):
        color = 'transparent'
        if value == 1:
            color = 'lightgreen'
        elif value == 0:
            color = 'lightcoral'
        
        label.setStyleSheet(f'QLabel {{ background-color: {color}; }}')

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
        self.line_th_1, = self.canvas_th_1.axes.plot([], [], linewidth=2.5, color='darkblue')

        self.canvas_th_1.axes.text(0.01, 0.95, 'Th 1 Filtered', ha='left', va='top', color='blue', fontsize = 10, transform=self.canvas_th_1.axes.transAxes)
        layout.addWidget(self.canvas_th_1, 21, 0, 10, 10)

        # Variables for Th1 semi-automatic annotation
        self.neighborhood_patches_th1 = []
        self.interval_vertical_stripes_th1 = []
        self.correct_dips_th1, = self.canvas_th_1.axes.plot([], [], marker='P', linestyle='None', color='darkgreen', markersize=10, zorder=50)
        self.dangerous_dips_th1, = self.canvas_th_1.axes.plot([], [], marker='X', linestyle='None', color='red', markersize=10, zorder=100)
        self.dubious_dips_th1, = self.canvas_th_1.axes.plot([], [], marker='$?!$', linestyle='None', color='black', markersize=10, zorder=25)
        self.trend_lines_th1 = []

        # Matplotlib canvas for the Th 2 signal
        self.canvas_th_2 = MplCanvas(self, width, height)
        self.line_th_2, = self.canvas_th_2.axes.plot([], [], linewidth=2.5, color='darkblue')

        self.canvas_th_2.axes.text(0.01, 0.95, 'Th 2 Filtered', ha='left', va='top', color='blue', fontsize = 10, transform=self.canvas_th_2.axes.transAxes)
        layout.addWidget(self.canvas_th_2, 31, 0, 10, 10)

        # Variables for Th2 semi-automatic annotation
        self.neighborhood_patches_th2 = []
        self.interval_vertical_stripes_th2 = []
        self.correct_dips_th2, = self.canvas_th_2.axes.plot([], [], marker='P', linestyle='None', color='darkgreen', markersize=10, zorder=50)
        self.dangerous_dips_th2, = self.canvas_th_2.axes.plot([], [], marker='X', linestyle='None', color='red', markersize=10, zorder=100)
        self.dubious_dips_th2, = self.canvas_th_2.axes.plot([], [], marker='$?!$', linestyle='None', color='black', markersize=10, zorder=25)
        self.trend_lines_th2 = []

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

        self.label_conf = QtWidgets.QLabel('Confident:')
        layout.addWidget(self.label_conf, 4, 11, 2, 2)

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
        self.label_ref = QtWidgets.QLabel('Reference visible:')
        layout.addWidget(self.label_ref, 12, 11, 2, 2)

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
        self.label_sig1 = QtWidgets.QLabel('Signal visible:')
        layout.addWidget(self.label_sig1, 22, 11, 2, 2)

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
        self.label_sig2 = QtWidgets.QLabel('Signal visible:')
        layout.addWidget(self.label_sig2, 32, 11, 2, 2)

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

        for collection in list(self.line_th_ref.axes.collections):
            collection.remove()

        y_min, _ = self.line_th_ref.axes.get_ylim()

        # Plot shading for Inspiration (1)
        if np.any(inspiration_mask):  # Check if any inspiration phase is present
            self.line_th_ref.axes.fill_between(x,
                                               y1=y_min,
                                               y2=self.DATA['Th_Ref_Filt'][start:end], 
                                               where=inspiration_mask, 
                                               color='green', 
                                               alpha=0.9, 
                                               edgecolor='none')

        # Plot shading for Expiration (0)
        if np.any(expiration_mask):  # Check if any expiration phase is present
            self.line_th_ref.axes.fill_between(x, 
                                               y1=y_min,
                                               y2=self.DATA['Th_Ref_Filt'][start:end], 
                                               where=expiration_mask, 
                                               color='red', 
                                               alpha=0.9,
                                               edgecolor='none')

    def update_semi_auto_annotations(self, x, start, end, annotation_type: AnnotationType):
        EXPIRATION_LENGTH = 0.85
        INSPIRATION_LENGTH = 0.2
        DIP_SEPARATION = .15
        REQUIRED_AMPLITUDE_PERCENTAGE = 0.05 

        match annotation_type:
            case AnnotationType.TH1:
                FILTERED_SIGNAL = 'Th1_Filt'
                SIGNAL_CANVAS = self.canvas_th_1
                VERTICAL_STRIPES = self.interval_vertical_stripes_th1
                TREND_LINES = self.trend_lines_th1 
                CORRECT_DIPS = self.correct_dips_th1
                DANGEROUS_DIPS = self.dangerous_dips_th1
                DUBIOUS_POINTS = self.dubious_dips_th1 
            case AnnotationType.TH2:
                FILTERED_SIGNAL = 'Th2_Filt'
                SIGNAL_CANVAS = self.canvas_th_2
                VERTICAL_STRIPES = self.interval_vertical_stripes_th2
                TREND_LINES = self.trend_lines_th2 
                CORRECT_DIPS = self.correct_dips_th2
                DANGEROUS_DIPS = self.dangerous_dips_th2
                DUBIOUS_POINTS = self.dubious_dips_th2
            case _:
                return
        
        # get unclapmed reference subset
        total_data_len = len(self.DATA)
        visible_length = end - start

        safe_start = max(0, start - 2 * visible_length)
        safe_end = min(total_data_len, end + 2 * visible_length)
        subset = self.DATA['Phase'].iloc[safe_start : safe_end]
        
        transition_indices = subset.index[subset.diff() == 1].tolist()
        reference_peaks = subset.index[subset.diff() == -1].tolist()

        # determine reference phase transitions within and around the visible range
        prev_candidates = [t for t in transition_indices if t < start]
        closest_prev = [prev_candidates[-1]] if prev_candidates else []
        curr_candidates = [t for t in transition_indices if start <= t < end]
        next_candidates = [t for t in transition_indices if t >= end]
        closest_next = [next_candidates[0]] if next_candidates else []

        reference_dips = closest_prev + curr_candidates + closest_next

        # get reference ranges for predicted expiration (dips) and inspiration (peaks)
        reference_ranges_dips = []
        for dip in reference_dips:
            preceding_peaks = [p for p in reference_peaks if p < dip]
            closest_preceding_peak = preceding_peaks[-1] if preceding_peaks else safe_start
            following_peaks = [p for p in reference_peaks if p > dip]
            closest_following_peak = following_peaks[0] if following_peaks else safe_end - 1

            expiration_length = dip - closest_preceding_peak
            inspiration_length = closest_following_peak - dip

            r_start = int(dip - expiration_length * EXPIRATION_LENGTH) - start
            r_end = int(dip + inspiration_length * INSPIRATION_LENGTH) - start
            reference_ranges_dips.append((r_start, r_end))

        reference_ranges_peaks = []
        if reference_ranges_dips:
            reference_ranges_dips.sort(key=lambda x: x[0])
            for i in range(len(reference_ranges_dips) - 1):
                d_curr_end = reference_ranges_dips[i][1]
                d_next_start = reference_ranges_dips[i+1][0]
                if d_curr_end < d_next_start:
                    reference_ranges_peaks.append((d_curr_end, d_next_start))

        # clean up signals for fresh plotting
        while VERTICAL_STRIPES:
            VERTICAL_STRIPES.pop().remove()
        
        while TREND_LINES:
            TREND_LINES.pop().remove()

        # draw vertical stripes for predicted reference phases (expiration/inspiration)
        def plot_stripes(ranges, color):
            for r_start, r_end in ranges:
                draw_start = max(0, min(visible_length - 1, r_start))
                draw_end = max(0, min(visible_length - 1, r_end))
                if draw_start < draw_end:
                    stripe = SIGNAL_CANVAS.axes.axvspan(
                        x[draw_start], x[draw_end],
                        color=color, alpha=0.15, zorder=0
                    )
                    VERTICAL_STRIPES.append(stripe)

        plot_stripes(reference_ranges_dips, 'red')
        plot_stripes(reference_ranges_peaks, 'green')

        if not reference_ranges_dips and not reference_ranges_peaks:
            return

        # slice the relevant signal data for processing
        all_starts = [r[0] for r in reference_ranges_dips + reference_ranges_peaks]
        all_ends = [r[1] for r in reference_ranges_dips + reference_ranges_peaks]

        if not all_starts: return

        min_rel_idx = min(all_starts)
        max_rel_idx = max(all_ends)

        abs_search_start = max(0, start + min_rel_idx)
        abs_search_end = min(total_data_len, start + max_rel_idx + 1)
        window_offset = abs_search_start - start 

        process_data = self.DATA[FILTERED_SIGNAL].iloc[abs_search_start : abs_search_end].to_numpy()
        if len(process_data) == 0: return

        # analyze trend lines (monotony) within each reference range and plot them
        def analyze_and_plot_trend(ranges, expected_trend):
            for r_start, r_end in ranges:
                idx_start = max(0, r_start - window_offset)
                idx_end = min(len(process_data), r_end - window_offset + 1)
                
                if idx_end - idx_start < 2: 
                    continue

                y_segment = process_data[idx_start:idx_end]
                x_segment = np.arange(len(y_segment)) 
                
                slope, intercept = np.polyfit(x_segment, y_segment, 1)

                is_valid = False
                if expected_trend == 'down':
                    if slope <= 0: is_valid = True
                else:
                    if slope >= 0: is_valid = True
                
                line_color = 'green' if is_valid else 'red'
                
                draw_idx_start = max(0, min(visible_length - 1, r_start))
                draw_idx_end = max(0, min(visible_length - 1, r_end))

                if draw_idx_start < draw_idx_end:
                    local_idx_start = (draw_idx_start - r_start) if r_start < 0 else 0
                    local_idx_end = (len(y_segment) - 1) - ((r_end) - draw_idx_end) if r_end >= visible_length else (len(y_segment) - 1)
                    
                    draw_y_start = slope * local_idx_start + intercept
                    draw_y_end = slope * local_idx_end + intercept

                    if (line_color == 'green'): continue 

                    line, = SIGNAL_CANVAS.axes.plot(
                        [x[draw_idx_start], x[draw_idx_end]], 
                        [draw_y_start, draw_y_end],
                        color=line_color, linewidth=3, linestyle='--', alpha=0.9
                    )
                    TREND_LINES.append(line)

        analyze_and_plot_trend(reference_ranges_dips, 'down')
        analyze_and_plot_trend(reference_ranges_peaks, 'up')

        # find local extrema within and around the segment
        seg_min = np.min(process_data)
        seg_max = np.max(process_data)
        dynamic_threshold = (seg_max - seg_min) * REQUIRED_AMPLITUDE_PERCENTAGE

        found_dips, _ = find_peaks(-process_data, distance=self.FS*DIP_SEPARATION)
        found_peaks, _ = find_peaks(process_data, distance=self.FS*DIP_SEPARATION)

        selected_points_expiration = [] 
        selected_points_inspiration = [] 

        def get_rel_idx(idx_in_window):
            return idx_in_window + window_offset

        # process expiration (dips) selecting the lowest dip in each reference range, defaulting to local min if none found
        for r_start, r_end in reference_ranges_dips:
            candidates = []
            for idx in found_dips:
                rel_idx = get_rel_idx(idx)
                if r_start <= rel_idx <= r_end:
                    candidates.append(idx)
            
            if candidates:
                best_internal = min(candidates, key=lambda i: process_data[i])
                best_rel = get_rel_idx(best_internal)
                val = process_data[best_internal]
            else:
                s_start = max(0, r_start - window_offset)
                s_end = min(len(process_data), r_end - window_offset + 1)
                
                if s_start < s_end:
                    local_min_idx = np.argmin(process_data[s_start:s_end])
                    best_internal = s_start + local_min_idx
                    best_rel = get_rel_idx(best_internal)
                    val = process_data[best_internal]
                else:
                    continue
            
            selected_points_expiration.append((best_rel, val))

        # process inspiration (peaks) selecting the highest peak in each reference range, defaulting to local max if none found
        for r_start, r_end in reference_ranges_peaks:
            candidates = []
            for idx in found_peaks:
                rel_idx = get_rel_idx(idx)
                if r_start <= rel_idx <= r_end:
                    candidates.append(idx)
            
            if candidates:
                best_internal = max(candidates, key=lambda i: process_data[i])
                best_rel = get_rel_idx(best_internal)
                val = process_data[best_internal]
            else:
                s_start = max(0, r_start - window_offset)
                s_end = min(len(process_data), r_end - window_offset + 1)

                if s_start < s_end:
                    local_max_idx = np.argmax(process_data[s_start:s_end])
                    best_internal = s_start + local_max_idx
                    best_rel = get_rel_idx(best_internal)
                    val = process_data[best_internal]
                else:
                    continue

            selected_points_inspiration.append((best_rel, val))

        # pair extrema points and categorize them as correct, or wrong
        all_points = sorted(
            [(i, v, True) for i, v in selected_points_expiration] + 
            [(i, v, False) for i, v in selected_points_inspiration],
            key=lambda x: x[0]
        )

        correct_points = []
        wrong_points = []

        for k in range(0, len(all_points) - 1):
            p1 = all_points[k]
            p2 = all_points[k+1]
            
            is_valid = False
            
            if p1[2] != p2[2]: 
                val_dip = p1[1] if p1[2] else p2[1]
                val_peak = p2[1] if p1[2] else p1[1]

                if (val_peak - val_dip) > dynamic_threshold:
                    is_valid = True
            
            if is_valid:
                correct_points.append((p1[0], p1[1]))
                correct_points.append((p2[0], p2[1]))
            else:
                wrong_points.append((p1[0], p1[1]))
                wrong_points.append((p2[0], p2[1]))

        correct_points = list(set(correct_points))
        wrong_points = list(set(wrong_points))

        # identify dubious points (peaks too high in expiration, dips too low in inspiration)
        dubious_points = []
        
        num_pairs = min(len(reference_ranges_dips), len(reference_ranges_peaks))

        for i in range(num_pairs):
            range_exp = reference_ranges_dips[i]
            range_insp = reference_ranges_peaks[i]

            # determine local min/max thresholds of expiration & inspiration for the current pair of reference ranges
            s_exp_start = max(0, range_exp[0] - window_offset)
            s_exp_end = min(len(process_data), range_exp[1] - window_offset + 1)
            
            if s_exp_start >= s_exp_end: continue
            threshold_lowest_dip = np.min(process_data[s_exp_start:s_exp_end])

            s_insp_start = max(0, range_insp[0] - window_offset)
            s_insp_end = min(len(process_data), range_insp[1] - window_offset + 1)

            if s_insp_start >= s_insp_end: continue
            threshold_highest_peak = np.max(process_data[s_insp_start:s_insp_end])

            # check for improper peaks in expiration (higher than the highest peak in relevant inspiration)
            for idx in found_peaks:
                rel_idx = get_rel_idx(idx)
                if range_exp[0] <= rel_idx <= range_exp[1]:
                    val = process_data[idx]
                    if val > threshold_highest_peak:
                        dubious_points.append((rel_idx, val))

            # check for improper dips in inspiration (lower than the lowest dips in relevant expiration)
            for idx in found_dips:
                rel_idx = get_rel_idx(idx)
                if range_insp[0] <= rel_idx <= range_insp[1]:
                    val = process_data[idx]
                    if val < threshold_lowest_dip:
                        dubious_points.append((rel_idx, val))

        # map filtered points to visible canvas coordinates and update plots to visualize them
        def filter_and_map(points_list):
            xs = []
            ys = []
            for idx, val in points_list:
                if 0 <= idx < visible_length:
                    xs.append(x[idx])
                    ys.append(val)
            return xs, ys

        cx, cy = filter_and_map(correct_points)
        wx, wy = filter_and_map(wrong_points)
        dx, dy = filter_and_map(dubious_points)

        CORRECT_DIPS.set_data(cx, cy)
        DANGEROUS_DIPS.set_data(wx, wy)
        DUBIOUS_POINTS.set_data(dx, dy)

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

        # Update semi-automatic annotations
        self.update_semi_auto_annotations(x, start, end, AnnotationType.TH1)
        self.update_semi_auto_annotations(x, start, end, AnnotationType.TH2)

        # Update shading of expiratory and inspiratory phases
        self.update_shading(x, start, end)

        # Update segment label
        self.seg_label.setText('SEGMENT ' + str(self.seg_curr + 1) + ' / ' + str(math.floor(len(self.DATA) / seg_len_pts)))
        
        # Update selection of radio buttons
        conf_val = self.dh.get_column_value(self.seg_curr, 'Confidence')
        self.update_selection(self.radio_seg_yes, self.radio_seg_no, conf_val)
        self.set_label_style(self.label_conf, conf_val)

        ref_val = self.dh.get_column_value(self.seg_curr, 'Reference')
        self.update_selection(self.radio_ref_yes, self.radio_ref_no, ref_val)
        self.set_label_style(self.label_ref, ref_val)
        
        sig1_val = self.dh.get_column_value(self.seg_curr, 'Signal_1')
        self.update_selection(self.radio_sig1_yes, self.radio_sig1_no, sig1_val)
        self.set_label_style(self.label_sig1, sig1_val)
        
        sig2_val = self.dh.get_column_value(self.seg_curr, 'Signal_2')
        self.update_selection(self.radio_sig2_yes, self.radio_sig2_no, sig2_val)
        self.set_label_style(self.label_sig2, sig2_val)

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

                value = radio_button.label
                if column == 'Confidence':
                    self.set_label_style(self.label_conf, value)
                elif column == 'Reference':
                    self.set_label_style(self.label_ref, value)
                elif column == 'Signal_1':
                    self.set_label_style(self.label_sig1, value)
                elif column == 'Signal_2':
                    self.set_label_style(self.label_sig2, value)

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