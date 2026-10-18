# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
"""Qt desktop interface. Physics and training remain independent of the UI."""
from pathlib import Path
import json
import threading
import traceback
import numpy as np
import torch
from PySide6 import QtCore as C, QtGui as G, QtWidgets as W
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from .config import Config, CHANNEL_FIELDS
from .presets import preset, PRESET_NAMES
from .objectives import EXAMPLES, PARAMETERS,recipe_name
from .training import train
from .export import export_result, runtime_directory, array
from .recommend import recommendation
from .conventions import export_explanation
from .guide import QUICK_START, GUIDE_TOPICS, document, walkthrough
from .appearance import THEMES, COLORMAPS, DEFAULTS, recolor, load_preferences
from .digits import detector_overlay

STYLE = '''
QWidget { background:#090f1b; color:#dce6f3; font-family:'Segoe UI'; font-size:13px; }
QMainWindow { background:#090f1b; }
QFrame#rail { background:#070c15; border-right:1px solid #20314a; }
QLabel#brand { color:#6af6e0; font-size:21px; font-weight:800; letter-spacing:2px; }
QLabel#eyebrow { color:#5ad5c7; font-size:11px; font-weight:600; letter-spacing:2px; }
QLabel#headline { font-size:27px; font-weight:650; }
QLabel#muted { color:#8294ae; }
QLabel#telemetry { font-family:'Consolas'; color:#6af6e0; font-size:14px; }
QPushButton { background:#142238; border:1px solid #2b415f; border-radius:6px; padding:10px 15px; font-weight:600; }
QPushButton:hover { background:#203650; border-color:#5ad5c7; }
QPushButton:disabled { color:#4b5b71; border-color:#1d293c; background:#101827; }
QPushButton#primary { color:#061719; background:#66f2d5; border:0; }
QPushButton#primary:hover { background:#a0ffe9; }
QPushButton#stop { color:#ff9b91; border-color:#61343c; }
QPushButton#nav { text-align:left; background:transparent; border:0; color:#8294ae; padding:15px 13px; }
QPushButton#nav:checked { background:#14293a; color:#78f6df; border-left:3px solid #78f6df; border-radius:0; }
QLineEdit,QSpinBox,QDoubleSpinBox,QComboBox,QPlainTextEdit,QTextBrowser { background:#111c2d; border:1px solid #293a53; border-radius:5px; padding:7px; selection-background-color:#277c80; }
QLineEdit:focus,QSpinBox:focus,QDoubleSpinBox:focus,QPlainTextEdit:focus { border:1px solid #6af6e0; }
QComboBox::drop-down { border:0; width:22px; }
QComboBox QAbstractItemView { background:#142238; selection-background-color:#226366; }
QGroupBox { border:1px solid #243650; border-radius:8px; margin-top:19px; padding:17px 12px 12px; font-weight:600; }
QGroupBox::title { subcontrol-origin:margin; left:14px; padding:0 5px; color:#69e8d3; }
QScrollArea { border:0; }
QScrollBar:vertical { background:#0b1421; width:8px; }
QScrollBar::handle:vertical { background:#2b435d; border-radius:4px; min-height:25px; }
QProgressBar { border:0; border-radius:3px; background:#1b2a40; height:7px; text-align:center; }
QProgressBar::chunk { background:#67ead5; border-radius:3px; }
QTabWidget::pane { border:1px solid #233650; }
QTabBar::tab { background:#101c2d; padding:9px 15px; color:#8196af; }
QTabBar::tab:selected { color:#6af6e0; border-bottom:2px solid #6af6e0; }
QListWidget { background:#101b2c; border:1px solid #233650; padding:7px; }
QListWidget::item { padding:8px; }
QListWidget::item:selected { background:#1c484e; color:#82ffe6; }
QSplitter::handle { background:#192b41; width:2px; }
'''


def label(text, name=None, wrap=False):
    item=W.QLabel(text)
    if name: item.setObjectName(name)
    item.setWordWrap(wrap)
    return item


class Worker(C.QObject):
    update=C.Signal(dict)
    done=C.Signal(object,object)
    failed=C.Signal(str)

    def __init__(self, config, stop, preview=False, output_base=None,plot_settings=None,existing_result=None):
        super().__init__();self.config=config;self.stop=stop;self.preview=preview;self.output_base=output_base;self.plot_settings=plot_settings
        self.existing_result=existing_result

    @C.Slot()
    def run(self):
        try:
            result=self.existing_result if self.existing_result is not None else train(self.config,self.update.emit,self.stop,self.preview)
            if not self.preview:self.update.emit({"message":"Training finished. Exporting phase maps, fields and figures…"})
            path=None if self.preview else export_result(result,self.output_base,plot_settings=self.plot_settings)
            self.done.emit(result,path)
        except Exception:
            self.failed.emit(traceback.format_exc())


class LiveCanvas(FigureCanvasQTAgg):
    def __init__(self):
        self.colors=THEMES["Deep space"];self.cmaps=DEFAULTS.copy();self.detector_artists=[];self.detector_signature=None
        self.fig=Figure(figsize=(8,6),facecolor="#0c1524",layout="constrained")
        super().__init__(self.fig)
        self.axes=self.fig.subplots(2,2)
        self.reset()

    def reset(self):
        for ax,title in zip(self.axes.flat,["INCIDENT AMPLITUDE","OUTPUT INTENSITY","SURFACE 01 / PHASE","OPTIMIZATION TRACE"]):
            ax.clear();ax.set_facecolor(self.colors["plot"]);ax.set_title(title,color=self.colors["muted"],fontsize=9,loc="left",pad=13)
            ax.tick_params(colors=self.colors["muted"],labelsize=8)
            for spine in ax.spines.values():spine.set_color(self.colors["border"])
        self.draw_idle()
        self.images=[]
        for i,(ax,cmap) in enumerate(zip(list(self.axes.flat)[:3],["magma","inferno","twilight"])):
            # Smooth only the displayed fields; keep individual metasurface cells exact.
            self.images.append(ax.imshow(np.zeros((2,2)),origin="lower",cmap=cmap,aspect="equal",
                                         interpolation="nearest" if i==2 else "bilinear",interpolation_stage="data"))
            ax.set_xticks([]);ax.set_yticks([])
        self.trace,=self.axes[1,1].plot([],[],color=self.colors["accent"],lw=1.7)

    def apply_appearance(self,prefs):
        self.colors=THEMES[prefs["theme"]];self.cmaps=prefs.copy()
        self.fig.set_facecolor(self.colors["plot"])
        for ax in self.axes.flat:
            ax.set_facecolor(self.colors["plot"]);ax.tick_params(colors=self.colors["muted"])
            ax.title.set_color(self.colors["muted"]);ax.xaxis.label.set_color(self.colors["muted"])
            ax.yaxis.label.set_color(self.colors["muted"])
            for spine in ax.spines.values():spine.set_color(self.colors["border"])
            for line in ax.get_xgridlines()+ax.get_ygridlines():line.set_color(self.colors["border"])
        self.trace.set_color(self.colors["accent"])
        self.images[0].set_cmap(prefs["input_cmap"]);self.images[1].set_cmap(prefs["intensity_cmap"]);self.images[2].set_cmap(prefs["phase_cmap"])
        for artist in self.detector_artists:
            if hasattr(artist,"set_edgecolor"):artist.set_edgecolor(self.colors["accent"])
            else:artist.set_color(self.colors["accent"])
        self.draw_idle()

    def update_data(self, data):
        for i,(ax,key) in enumerate(zip(list(self.axes.flat)[:3],["input","intensity","phase"])):
            if key in data:
                values=data[key];artist=self.images[i]
                artist.set_data(values)
                artist.set_extent((-.5,values.shape[1]-.5,-.5,values.shape[0]-.5))
                artist.set_clim(-np.pi if key=="phase" or (key=="intensity" and data.get("output_view")=="Phase") else float(np.min(values)),
                                np.pi if key=="phase" or (key=="intensity" and data.get("output_view")=="Phase") else max(float(np.max(values)),1e-16))
                ax.set_xlim(-.5,values.shape[1]-.5);ax.set_ylim(-.5,values.shape[0]-.5)
                if key=="intensity":
                    # Color interpolation avoids averaging wrapped phase angles through zero.
                    artist.set_interpolation_stage("rgba" if data.get("output_view")=="Phase" else "data")
                    artist.set_cmap(self.cmaps["phase_cmap"] if data.get("output_view")=="Phase" else self.cmaps["intensity_cmap"])
                    ax.set_title("OUTPUT / "+data.get("output_view","Intensity").upper(),color=self.colors["muted"],fontsize=9,loc="left",pad=13)
                if key=="phase":ax.set_title(f"SURFACE {data.get('surface',1):02d} / PHASE",color=self.colors["muted"],fontsize=9,loc="left",pad=13)
        signature=(tuple(tuple(x) for x in data.get("detectors",[])),tuple(data.get("intensity",np.zeros((2,2))).shape))
        if signature!=self.detector_signature:
            from matplotlib.patches import Ellipse
            for artist in self.detector_artists:artist.remove()
            self.detector_artists=[];self.detector_signature=signature
            h,w=signature[1]
            for x,y,rx,ry,digit in signature[0]:
                circle=Ellipse((x*w-.5,y*h-.5),2*rx*w,2*ry*h,fill=False,edgecolor=self.colors["accent"],linewidth=.7)
                self.axes[0,1].add_patch(circle);self.detector_artists.append(circle)
                self.detector_artists.append(self.axes[0,1].text(x*w-.5,y*h-.5,str(digit),color=self.colors["accent"],fontsize=7,ha="center",va="center"))
        history=data.get("history",[])
        if len(history):
            ax=self.axes[1,1]
            values=np.asarray(history)
            if data.get("normalized",True):values=values/max(abs(float(values[0])),1e-16)
            indices=np.linspace(0,len(values)-1,min(len(values),2500),dtype=int)
            self.trace.set_data(indices+1,values[indices])
            ax.relim();ax.autoscale_view()
            ax.set_title("NORMALIZED LOSS / |INITIAL LOSS|" if data.get("normalized",True) else "RAW OBJECTIVE LOSS",color=self.colors["muted"],fontsize=9,loc="left",pad=13)
            ax.grid(color=self.colors["border"],alpha=.5)
            ax.set_xlabel("epoch",color=self.colors["muted"],fontsize=8)
        self.draw_idle()


class SpacingsEditor(W.QWidget):
    """Explicit per-plane distances, including a separate output-plane control."""
    def __init__(self):
        super().__init__();self.form=W.QFormLayout(self);self.form.setContentsMargins(0,0,0,0)
        self.controls=[];self.set_values([1.2,10.8])

    def values(self):return [w.value() for w in self.controls]

    def set_values(self,values):
        while self.form.rowCount():self.form.removeRow(0)
        self.controls=[]
        for i,value in enumerate(values):
            w=W.QDoubleSpinBox();w.setDecimals(5);w.setRange(.00001,100000);w.setValue(value)
            title="Input → surface 1" if i==0 else (f"Surface {i} → output" if i==len(values)-1 else f"Surface {i} → {i+1}")
            w.setToolTip(title+", in wavelengths. This distance changes the propagated optical field.")
            self.form.addRow(title+" [λ]",w);self.controls.append(w)

    def set_layers(self,layers):
        old=self.values();between=old[1:-1]
        self.set_values([old[0]]+(between+[4.]*(layers-1))[:layers-1]+[old[-1]])


class Window(W.QMainWindow):
    run_complete=C.Signal()

    def __init__(self,output_base=None,preferences_path=None):
        super().__init__()
        self.output_base=output_base
        self.setWindowTitle("WAVEFORGE / Diffractive Neural Lab")
        self.resize(1460,940);self.setMinimumSize(1120,760)
        self.fields={};self.field_rows={};self.thread=None;self.result=None;self.result_dir=None;self.preview=False
        self._loading=True;self._appearance_ready=False
        self.preferences_path=Path(preferences_path) if preferences_path else runtime_directory()/"ui_preferences.json"
        self.preferences=load_preferences(self.preferences_path)
        self.live_cache={};self.active_polarization=1
        self.setStyleSheet(STYLE)
        central=W.QWidget();self.setCentralWidget(central)
        layout=W.QHBoxLayout(central);layout.setContentsMargins(0,0,0,0);layout.setSpacing(0)
        rail=W.QFrame();rail.setObjectName("rail");rail.setFixedWidth(205)
        r=W.QVBoxLayout(rail);r.setContentsMargins(18,30,18,20);r.setSpacing(7)
        r.addWidget(label("//", "brand"));r.addWidget(label("WAVEFORGE","brand"));r.addWidget(label("D2NN / OPTICAL DESIGN","eyebrow"))
        r.addSpacing(35)
        self.nav=[]
        for i,title in enumerate(["01   Optical system","02   Input & target","03   Python objective","04   Experiments","05   Results archive","06   Guide & examples","07   Appearance"]):
            b=W.QPushButton(title.replace("&","&&"));b.setObjectName("nav");b.setCheckable(True)
            b.clicked.connect(lambda checked=False,j=i:self.navigate(j));r.addWidget(b);self.nav.append(b)
        r.addStretch();r.addWidget(label("COMPUTE ENGINE","eyebrow"))
        gpu=torch.cuda.is_available()
        self.device_label=label("CUDA / ONLINE" if gpu else "CPU / READY","telemetry");r.addWidget(self.device_label)
        gpu_name=torch.cuda.get_device_name(0) if gpu else "Automatic CPU fallback"
        r.addWidget(label(gpu_name,"muted",True));r.addSpacing(18)
        r.addWidget(label("RS INTEGRAL\nPHASE-ONLY SURFACES\nLOCAL • OFFLINE • v1.1.2","muted",True))
        r.addWidget(label("Made by Ali Nadarloo","muted",True))
        layout.addWidget(rail)
        main=W.QVBoxLayout();main.setContentsMargins(27,25,27,20);main.setSpacing(16);layout.addLayout(main,1)
        head=W.QHBoxLayout();title=W.QVBoxLayout()
        title.addWidget(label("PHOTONIC DESIGN / RESEARCH STATION","eyebrow"));title.addWidget(label("Shape light. Train possibility.","headline"))
        head.addLayout(title);head.addStretch()
        self.preset_combo=W.QComboBox();self.preset_combo.addItems(PRESET_NAMES);self.preset_combo.setMinimumWidth(220)
        self.preset_combo.activated.connect(lambda:self.load_preset(self.preset_combo.currentText()))
        head.addWidget(self.preset_combo);main.addLayout(head)
        toolbar=W.QHBoxLayout();self.config_buttons=[]
        for text,fn in [("Save experiment",self.save_config),("Load experiment",self.load_config),("Import phases",self.import_phases)]:
            b=W.QPushButton(text);b.clicked.connect(fn);toolbar.addWidget(b);self.config_buttons.append(b)
        toolbar.addStretch();toolbar.addWidget(label("λ / SI-AWARE     •     AUTOGRAD","eyebrow"));main.addLayout(toolbar)
        toolbar.addWidget(label("Editing","muted"))
        self.channel_combo=W.QComboBox();self.channel_combo.addItems(["Polarization 1","Polarization 2"])
        self.channel_combo.setToolTip("Shared geometry; separate inputs, target, objective and phase maps for each scalar polarization.")
        self.channel_combo.activated.connect(self.switch_channel);toolbar.addWidget(self.channel_combo)
        split=W.QSplitter();main.addWidget(split,1)
        self.pages=W.QStackedWidget();self.pages.setMinimumWidth(355);split.addWidget(self.pages)
        self.make_system();self.make_inputs();self.make_objective();self.make_experiments();self.make_results();self.make_guide();self.make_appearance()
        live=W.QWidget();self.live_panel=live;lv=W.QVBoxLayout(live);lv.setContentsMargins(18,0,0,0)
        line=W.QHBoxLayout();line.addWidget(label("LIVE WAVEFRONT","eyebrow"));line.addStretch();self.epoch_label=label("READY / AWAITING RUN","muted");line.addWidget(self.epoch_label);lv.addLayout(line)
        controls=W.QHBoxLayout()
        self.live_pol=W.QComboBox();self.live_pol.addItems(["Follow active","Polarization 1","Polarization 2"])
        self.live_sample=W.QComboBox();self.live_sample.addItem("input")
        self.live_surface=W.QComboBox();self.live_surface.addItem("Surface 1")
        self.live_view=W.QComboBox();self.live_view.addItems(["Intensity","Phase","Amplitude"])
        self.live_loss=W.QComboBox();self.live_loss.addItems(["Normalized loss","Raw loss"])
        for control in [self.live_pol,self.live_sample,self.live_surface,self.live_view,self.live_loss]:
            control.setSizeAdjustPolicy(W.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            control.setMinimumContentsLength(5);control.currentIndexChanged.connect(self.render_live);controls.addWidget(control)
        lv.addLayout(controls)
        self.canvas=LiveCanvas();lv.addWidget(self.canvas,1)
        self.metrics_label=label("Preview the optical fields, then launch training.","telemetry",True);lv.addWidget(self.metrics_label)
        self.status=label("Results are saved beside the application in results/.","muted",True);lv.addWidget(self.status)
        split.addWidget(live);split.setSizes([440,760])
        self.progress=W.QProgressBar();self.progress.setTextVisible(False);main.addWidget(self.progress)
        actions=W.QHBoxLayout();self.summary=label("32 × 32 CELLS  /  1 SURFACE  /  1 THz","muted");actions.addWidget(self.summary);actions.addStretch()
        self.preview_btn=W.QPushButton("Preview / validate loss");self.preview_btn.clicked.connect(lambda:self.start(True));actions.addWidget(self.preview_btn)
        self.stop_btn=W.QPushButton("Stop && export");self.stop_btn.setObjectName("stop");self.stop_btn.setEnabled(False);self.stop_btn.clicked.connect(self.stop_run);actions.addWidget(self.stop_btn)
        self.train_btn=W.QPushButton("▶  Launch training");self.train_btn.setObjectName("primary");self.train_btn.clicked.connect(lambda:self.start(False));actions.addWidget(self.train_btn);main.addLayout(actions)
        self.load_preset(PRESET_NAMES[0]);self.navigate(0)
        self._appearance_ready=True;self.apply_appearance(False)
        for key in ("aperture","beam","target","propagation","optimizer"):
            self.fields[key].currentTextChanged.connect(lambda text,k=key:self.context_changed(k))
        self.fields["layers"].valueChanged.connect(lambda count:self.fields["distances_wl"].set_layers(count))
        self.refresh_context()
        self.advice_timer=C.QTimer(self);self.advice_timer.timeout.connect(self.update_advice);self.advice_timer.start(750)

    def page(self, title, description):
        scroll=W.QScrollArea();scroll.setWidgetResizable(True)
        inner=W.QWidget();v=W.QVBoxLayout(inner);v.setContentsMargins(0,0,12,12);v.setSpacing(12)
        v.addWidget(label(title,"headline"));v.addWidget(label(description,"muted",True))
        scroll.setWidget(inner);self.pages.addWidget(scroll)
        return v

    def group(self,layout,title):
        box=W.QGroupBox(title);form=W.QFormLayout(box);form.setFieldGrowthPolicy(W.QFormLayout.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(W.QFormLayout.WrapLongRows)
        form.setVerticalSpacing(11);layout.addWidget(box);return form

    def field(self,form,key,title,kind="float",minimum=0,maximum=100000,choices=None,tip=""):
        if kind=="spacings":w=SpacingsEditor()
        elif choices:
            w=W.QComboBox();w.addItems(choices)
        elif kind=="text":w=W.QLineEdit()
        elif kind=="int":w=W.QSpinBox();w.setRange(int(minimum),int(maximum))
        else:
            w=W.QDoubleSpinBox();w.setDecimals(5);w.setRange(minimum,maximum);w.setSingleStep(.1)
        w.setToolTip(tip or title);w.setObjectName(key);self.fields[key]=w;form.addRow(title,w)
        self.field_rows[key]=(form,w)
        return w

    def file_field(self,form,key,title):
        row=W.QWidget();h=W.QHBoxLayout(row);h.setContentsMargins(0,0,0,0)
        edit=W.QLineEdit();self.fields[key]=edit;edit.setObjectName(key);h.addWidget(edit)
        b=W.QPushButton("…");b.setFixedWidth(38);b.clicked.connect(lambda:self.pick_file(edit));h.addWidget(b);form.addRow(title,row)
        self.field_rows[key]=(form,row)

    def make_system(self):
        v=self.page("Optical system","Define the sampled metasurfaces and propagation geometry. Every distance is explicit.")
        f=self.group(v,"01 / Geometry")
        mode=self.field(f,"polarization_count","Scalar polarizations","int",1,2)
        mode.valueChanged.connect(self.update_polarization_mode)
        for key,title in [("nx","Cells along x"),("ny","Cells along y")]:self.field(f,key,title,"int",8,256)
        self.field(f,"layers","Metasurfaces","int",1,8)
        self.field(f,"dx_um","Cell pitch x [µm]",minimum=.001)
        self.field(f,"dy_um","Cell pitch y [µm]",minimum=.001)
        self.field(f,"frequency_thz","Frequency [THz]",minimum=.00001)
        self.field(f,"distances_wl","Plane distances","spacings")
        self.field(f,"output_scale","Output window scale",minimum=1,maximum=3)
        self.field(f,"cst_axis","CST propagation axis",choices=["+z","-z","+x","-x","+y","-y"])
        f=self.group(v,"02 / Solver & optimization")
        self.field(f,"propagation","Propagation",choices=["Dense RS","FFT RS","ASM • evanescent decay","ASM • propagating only"])
        self.field(f,"samples_per_cell","ASM samples per cell","int",1,8)
        self.field(f,"padding_factor","ASM zero padding factor","int",1,8)
        self.field(f,"device","Compute",choices=["Auto (CUDA / CPU)","CPU","CUDA preferred"])
        self.field(f,"epochs","Epochs","int",1,100000)
        self.field(f,"learning_rate","Learning rate",minimum=.000001,maximum=1)
        self.fields["learning_rate"].setDecimals(6)
        self.field(f,"optimizer","Training method",choices=["Adam","AdamW","SGD + momentum","RMSprop","Adagrad"])
        self.field(f,"momentum","Momentum (SGD / RMSprop)",minimum=0,maximum=.999)
        self.field(f,"weight_decay","Weight decay",minimum=0,maximum=1)
        self.field(f,"seed","Random seed","int",0,2147483647)
        self.field(f,"phase_initialization","Initial phase",choices=["Zeros (paper)","Small random","Random [-pi, pi]"])
        self.advice_label=label("Solver recommendation updates with your geometry and task.","muted",True);v.addWidget(self.advice_label)
        advice_button=W.QPushButton("Apply recommended propagation");advice_button.clicked.connect(self.apply_advice);v.addWidget(advice_button)
        v.addWidget(label("Dense RS matches the paper. FFT RS retains its sampled kernel. Padded ASM resolves each constant-phase cell with subcell samples and uses an enlarged FFT window. All use exp(+iωt).","muted",True));v.addStretch()

    def update_advice(self):
        try:
            data=self.read_fields();c=Config(**data)
            _,text=recommendation(c);self.advice_label.setText(text)
        except Exception:
            self.advice_label.setText("Complete the numeric geometry to see the solver recommendation.")

    def apply_advice(self):
        try:
            method,text=recommendation(Config(**self.read_fields()))
            self.fields["propagation"].setCurrentText(method);self.advice_label.setText(text)
        except Exception as exc:self.show_error(str(exc))

    def make_inputs(self):
        v=self.page("Input & target","Aperture × illumination defines the complex input. Targets guide your objective.")
        self.context_label=label("Only relevant controls are shown.","muted",True);v.addWidget(self.context_label)
        self.show_all=W.QCheckBox("Show all controls for custom Python work");self.show_all.toggled.connect(self.refresh_context);v.addWidget(self.show_all)
        f=self.group(v,"01 / Aperture")
        self.field(f,"aperture","Type",choices=["Open","Circle","Double slit","Logic gates","Text / number","Image / array","Handwritten digits"])
        self.field(f,"aperture_radius_wl","Radius / slit scale [λ]",minimum=.001)
        self.field(f,"text","Text or number","text")
        self.file_field(f,"aperture_file","Image / array")
        self.field(f,"gates","Gates (comma-separated)","text",tip="OR,AND,XOR,BUFFER or NOR,NAND,XNOR,NOT; complementary gates use separate scalar polarizations.")
        f=self.group(v,"02 / Incident beam")
        self.field(f,"beam","Profile",choices=["Plane wave","Gaussian","OAM (LG p=0)","Bessel–Gauss"])
        self.field(f,"waist_wl","Waist [λ]",minimum=.001)
        self.field(f,"charge","OAM charge ℓ","int",-8,8)
        self.field(f,"tilt_x_deg","Tilt x [degrees]",minimum=-79,maximum=79)
        self.field(f,"tilt_y_deg","Tilt y [degrees]",minimum=-79,maximum=79)
        f=self.group(v,"03 / Output target")
        self.field(f,"target","Type",choices=["Focus","Logic detectors","OAM mode","Image intensity","Text intensity","Digit detectors"])
        self.file_field(f,"target_file","Target image / array")
        self.field(f,"target_text","Target text","text")
        self.field(f,"target_charge","Target charge ℓ","int",-8,8)
        self.field(f,"target_waist_wl","Target waist [λ]",minimum=.001)
        self.field(f,"target_x_wl","Target x [λ]",minimum=-100,maximum=100)
        self.field(f,"target_y_wl","Target y [λ]",minimum=-100,maximum=100)
        f=self.group(v,"04 / Logic detectors")
        self.logic_group=f.parentWidget()
        self.field(f,"detector_radius_wl","Radius [λ]",minimum=.001)
        self.field(f,"detector_offset_wl","Center offset ±x [λ]",minimum=.001)
        self.field(f,"detector_one","Logic 1 detector","int",0,1)
        self.field(f,"desired_fraction","Desired energy fraction",minimum=.001,maximum=1)
        self.field(f,"outside_weight","Outside penalty",minimum=0,maximum=100)
        f=self.group(v,"05 / Handwritten digit recognition");self.digit_group=f.parentWidget()
        self.field(f,"digit_batch_size","Mini-batch size","int",1,256)
        self.field(f,"digit_samples_per_class","Train samples / digit (0 = all)","int",0,140)
        self.field(f,"digit_ring_radius_wl","Detector ring radius [λ]",minimum=.01,maximum=100)
        self.field(f,"digit_detector_radius_wl","Each circle radius [λ]",minimum=.01,maximum=100)
        self.digit_hint=label("Handwritten digits: a fixed 80/20 training / held-out split; ten labeled output circles. Output distance is the last control in Optical system → Plane distances.","muted",True);v.addWidget(self.digit_hint)
        self.file_hint=label("Files: PNG/JPEG/TIFF/BMP/GIF/WebP and numeric NPY/CSV/TXT. Numeric arrays must match [Ny,Nx]; complex arrays encode amplitude and phase. Raster files are resized. See the objective reference for conventions.","muted",True);v.addWidget(self.file_hint);v.addStretch()

    def make_objective(self):
        v=self.page("Python objective","Write loss(ctx) using the prepared fields. Add metrics(ctx) and plots(ctx) for custom reports.")
        self.example_combo=W.QComboBox();self.example_combo.addItems(EXAMPLES);v.addWidget(self.example_combo)
        b=W.QPushButton("Insert selected example");b.clicked.connect(self.insert_example);v.addWidget(b)
        tabs=W.QTabWidget();v.addWidget(tabs,1)
        self.editor=W.QPlainTextEdit();self.editor.setFont(G.QFont("Consolas",10));self.editor.setMinimumHeight(360);self.editor.setLineWrapMode(W.QPlainTextEdit.NoWrap);tabs.addTab(self.editor,"Code")
        reference=W.QPlainTextEdit(PARAMETERS);reference.setReadOnly(True);reference.setFont(G.QFont("Consolas",10));tabs.addTab(reference,"Parameters")
        v.addWidget(label("Local Python executes with your normal privileges. Preview validates the function and its gradients. Target settings do not replace a custom loss; insert an example to choose a new objective.","muted",True))

    def make_experiments(self):
        v=self.page("Experiment library","Start with a complete example, then adjust its geometry, inputs and objective.")
        descriptions=["A Gaussian beam focused to a small spot.","Original OR, AND, XOR, BUFFER; 3,000 epochs.","Original NOR, NAND, XNOR, NOT; 3,000 epochs.","Both paper channels with separate inputs and phase maps.","Two surfaces convert a Gaussian into a helical mode.","Shape the output intensity into letters or digits.","A bundled picture is ready to run; replace it with your own target.","Steer a double-slit beam by its intensity centroid.","Train concentration at two distinct observation planes.","Padded angular spectrum: four samples per cell, 3× padding, independent cell phases.","Four phase layers recognize handwritten 0–9 and route light to ten numbered output circles. Mini-batches, held-out accuracy, confusion matrix and adjustable output distance."]
        for name,desc in zip(PRESET_NAMES,descriptions):
            b=W.QPushButton(name);b.clicked.connect(lambda checked=False,n=name:self.load_preset(n));v.addWidget(b);v.addWidget(label(desc,"muted",True))
        v.addWidget(label("Choose one or two independent scalar polarizations. The unit-cell design determines their physical basis. The scalar model does not include polarization coupling.","muted",True));v.addStretch()

    def make_results(self):
        v=self.page("Results archive","Each run exports phases, complex fields, loss history, metrics and figures.")
        self.result_text=W.QPlainTextEdit();self.result_text.setReadOnly(True);self.result_text.setMinimumHeight(130);v.addWidget(self.result_text)
        b=W.QPushButton("Open results folder");b.clicked.connect(self.open_results);v.addWidget(b)
        self.reexport_btn=W.QPushButton("Export again with current appearance")
        self.reexport_btn.setEnabled(False);self.reexport_btn.clicked.connect(lambda:self.start(reexport=True));v.addWidget(self.reexport_btn)
        self.plot_list=W.QListWidget();self.plot_list.itemDoubleClicked.connect(self.open_plot);v.addWidget(self.plot_list,1)
        v.addWidget(label("Double-click a figure to view it. Stop & export saves the current phase maps and marks the run as interrupted. Import phases reloads phases.npz for fine-tuning.","muted",True))

    def make_guide(self):
        v=self.page("Guide & examples","Offline help, complete walkthroughs and copyable Python recipes.")
        self.wide_guide=W.QCheckBox("Wide reading area (hide the live plots while reading)")
        self.wide_guide.toggled.connect(lambda checked:self.live_panel.setVisible(not checked) if self.pages.currentIndex()==5 else None)
        v.addWidget(self.wide_guide)
        tabs=W.QTabWidget();v.addWidget(tabs,1)

        def browser():
            view=W.QTextBrowser();view.setMinimumHeight(430)
            view.setOpenExternalLinks(False)
            return view

        quick=browser();quick.setHtml(self.help_html(QUICK_START));tabs.addTab(quick,"How to use")
        manual=W.QWidget();mv=W.QVBoxLayout(manual);mv.setContentsMargins(5,5,5,5)
        self.guide_topic=W.QComboBox();self.guide_topic.addItems(GUIDE_TOPICS);mv.addWidget(self.guide_topic)
        self.guide_text=browser();mv.addWidget(self.guide_text,1)
        self.guide_topic.currentTextChanged.connect(self.show_guide_topic)
        self.show_guide_topic(self.guide_topic.currentText())
        tabs.addTab(manual,"Guide")

        examples=W.QWidget();ev=W.QVBoxLayout(examples);ev.setContentsMargins(5,5,5,5)
        ev.addWidget(label("COMPLETE EXPERIMENTS","eyebrow"))
        self.guide_preset=W.QComboBox();self.guide_preset.addItems(PRESET_NAMES);ev.addWidget(self.guide_preset)
        details=browser();details.setMinimumHeight(180);details.setHtml(self.help_html(walkthrough(PRESET_NAMES[0])));ev.addWidget(details)
        self.guide_preset.currentTextChanged.connect(lambda name:details.setHtml(self.help_html(walkthrough(name))))
        load=W.QPushButton("Load complete example");load.clicked.connect(self.load_guide_preset);ev.addWidget(load)
        ev.addWidget(label("PYTHON LOSS RECIPES","eyebrow"))
        self.guide_loss=W.QComboBox();self.guide_loss.addItems(EXAMPLES);ev.addWidget(self.guide_loss)
        self.guide_code=W.QPlainTextEdit();self.guide_code.setReadOnly(True)
        self.guide_code.setFont(G.QFont("Consolas",10));self.guide_code.setLineWrapMode(W.QPlainTextEdit.NoWrap)
        self.guide_code.setMinimumHeight(260);self.guide_code.setPlainText(next(iter(EXAMPLES.values())));ev.addWidget(self.guide_code)
        self.guide_loss.currentTextChanged.connect(lambda name:self.guide_code.setPlainText(EXAMPLES[name]))
        insert=W.QPushButton("Use recipe in current polarization");insert.clicked.connect(self.load_guide_loss);ev.addWidget(insert)
        ev.addWidget(label("Inserting replaces only the current channel's Python code. Match its target settings, then Preview. The Python objective Parameters tab contains the full field reference.","muted",True))
        tabs.addTab(examples,"Examples")
        workshop=browser();manual=runtime_directory()/"LOSS_GUIDE.md"
        workshop.setMarkdown(manual.read_text(encoding="utf-8") if manual.exists() else "See Python objective → Parameters and the copyable loss recipes.")
        tabs.addTab(workshop,"Loss lab")
        self.guide_tabs=tabs

    def help_html(self,html):return recolor(html,self.preferences["theme"])

    def show_guide_topic(self,topic):
        content=GUIDE_TOPICS[topic]
        if content.startswith("@file:"):
            file=runtime_directory()/content.split(":",1)[1]
            self.guide_text.setMarkdown(file.read_text(encoding="utf-8") if file.exists() else "This guide will be included beside the executable.")
            self.guide_text.document().setBaseUrl(C.QUrl.fromLocalFile(str(runtime_directory())+"/"))
        else:self.guide_text.setHtml(self.help_html(document(content)))

    def make_appearance(self):
        v=self.page("Appearance","Choose a comfortable workspace and independent color maps for each kind of field.")
        form=self.group(v,"Workspace theme");self.appearance_controls={}
        for key,title,choices in [("theme","Theme",list(THEMES)),("input_cmap","Input amplitude",COLORMAPS),
                                  ("intensity_cmap","Output intensity / amplitude",COLORMAPS),("phase_cmap","Surface and output phase",COLORMAPS)]:
            combo=W.QComboBox();combo.addItems(choices);combo.setCurrentText(self.preferences[key])
            form.addRow(title,combo);self.appearance_controls[key]=combo
            combo.currentTextChanged.connect(lambda text:self.apply_appearance())
        v.addWidget(label("Changes apply immediately to the live plots and are remembered beside the app in ui_preferences.json. Phase always keeps its −π…+π scale. Hot, jet and winter are available for every field type.","muted",True))
        v.addWidget(label("Saved figures use the current theme and color maps when export begins, with the same field interpolation as the live plots. Metasurface phase cells stay unsmoothed. After a completed run, use Results archive → Export again with current appearance to save a new results folder without retraining.","muted",True))
        v.addStretch()

    def apply_appearance(self,save=True):
        if not self._appearance_ready:return
        previous=self.preferences["theme"]
        self.preferences={key:combo.currentText() for key,combo in self.appearance_controls.items()}
        self.setStyleSheet(recolor(STYLE,self.preferences["theme"]))
        if self.preferences["theme"]=="Daylight":
            self.setStyleSheet(self.styleSheet()+"QPushButton:hover{background:#e0f2f0;} QPushButton#primary{color:#ffffff;} QPushButton#primary:hover{background:#006560;color:#ffffff;}")
        for view in self.findChildren(W.QTextBrowser):
            view.setHtml(recolor(view.toHtml(),self.preferences["theme"],previous))
        self.canvas.apply_appearance(self.preferences);self.render_live()
        if self.thread and hasattr(self,"worker"):self.worker.plot_settings=self.preferences.copy()
        if save:
            try:self.preferences_path.write_text(json.dumps(self.preferences,indent=2),encoding="utf-8")
            except OSError:self.status.setText("Appearance changed for this session; the settings folder is not writable.")

    def refresh_context(self,*args):
        if not hasattr(self,"show_all"):return
        advanced=self.show_all.isChecked();f=self.fields
        aperture=f["aperture"].currentText();beam=f["beam"].currentText();target=f["target"].currentText()
        visible=dict(aperture_radius_wl=aperture in ("Circle","Double slit"),text=aperture=="Text / number",
            aperture_file=aperture=="Image / array",gates=aperture=="Logic gates",waist_wl=beam!="Plane wave",
            charge=beam=="OAM (LG p=0)",target_file=target=="Image intensity",target_text=target=="Text intensity",
            target_charge=target=="OAM mode",target_waist_wl=target in ("Focus","OAM mode"),
            target_x_wl=target in ("Focus","OAM mode"),target_y_wl=target in ("Focus","OAM mode"),
            samples_per_cell=f["propagation"].currentText().startswith("ASM"),padding_factor=f["propagation"].currentText().startswith("ASM"),
            momentum=f["optimizer"].currentText() in ("SGD + momentum","RMSprop"),
            output_scale=f["propagation"].currentText()!="FFT RS")
        for key,show in visible.items():
            form,row=self.field_rows[key];form.setRowVisible(row,advanced or show)
        self.logic_group.setVisible(advanced or target=="Logic detectors")
        self.digit_group.setVisible(advanced or aperture=="Handwritten digits")
        self.digit_hint.setVisible(advanced or aperture=="Handwritten digits")
        self.file_hint.setVisible(advanced or aperture=="Image / array" or target=="Image intensity")
        self.context_label.setText(f"{aperture} / {beam} → {target}. "+("All controls are visible." if advanced else "Unrelated controls are hidden; their values are preserved."))

    def context_changed(self,key):
        if self._loading:return
        f=self.fields
        if key=="aperture":
            old=f["target"].currentText();aperture=f["aperture"].currentText()
            choices=["Focus","OAM mode","Image intensity","Text intensity"]
            if aperture=="Logic gates":choices.append("Logic detectors");target="Logic detectors"
            elif aperture=="Handwritten digits":choices=["Digit detectors"];target="Digit detectors"
            else:target=old if old in choices else "Focus"
            f["target"].blockSignals(True);f["target"].clear();f["target"].addItems(choices);f["target"].setCurrentText(target);f["target"].blockSignals(False)
        if key=="propagation" and f["propagation"].currentText()=="FFT RS":f["output_scale"].setValue(1.)
        if key in ("target","aperture"):
            suggested={"Logic detectors":"Paper logic gates","Digit detectors":"Digit recognition / ten detectors",
                       "OAM mode":"OAM / complex mode overlap"}.get(f["target"].currentText(),"Focus / image intensity")
            self.example_combo.setCurrentText(suggested)
            if recipe_name(self.editor.toPlainText()):self.editor.setPlainText(EXAMPLES[suggested])
        self.refresh_context()

    def load_guide_preset(self):
        self.load_preset(self.guide_preset.currentText());self.navigate(0)

    def load_guide_loss(self):
        self.example_combo.setCurrentText(self.guide_loss.currentText());self.insert_example();self.navigate(2)

    def navigate(self,index):
        self.pages.setCurrentIndex(index)
        if hasattr(self,"live_panel"):self.live_panel.setVisible(not(index==5 and self.wide_guide.isChecked()))
        for i,b in enumerate(self.nav):b.setChecked(i==index)

    def load_preset(self,name):
        self.set_config(preset(name));self.preset_combo.setCurrentText(name)
        if hasattr(self,"status"):self.status.setText(f"Loaded {name}. Preview validates the fields and loss gradients.")

    def set_config(self,c):
        self._loading=True
        if c.name in PRESET_NAMES:self.preset_combo.setCurrentText(c.name)
        else:
            self.preset_combo.setCurrentIndex(-1);self.preset_combo.setPlaceholderText(c.name)
        self.channel_index=0
        channels=c.channels()
        self.channel_cache=[{k:v for k,v in channels[0].to_dict().items() if k in CHANNEL_FIELDS},
                            {k:v for k,v in channels[-1].to_dict().items() if k in CHANNEL_FIELDS}]
        self.channel_combo.blockSignals(True);self.channel_combo.setCurrentIndex(0);self.channel_combo.blockSignals(False)
        self.config_name=c.name;self.initial_phases=c.initial_phases
        self.reset_targets(c.aperture,c.target)
        for key,w in self.fields.items():
            value=getattr(c,key)
            if isinstance(w,SpacingsEditor):w.set_values(value)
            elif isinstance(w,W.QComboBox):w.setCurrentText(str(value))
            elif isinstance(w,W.QLineEdit):w.setText(", ".join(map(str,value)) if isinstance(value,list) else str(value))
            else:w.setValue(value)
        self.editor.setPlainText(c.loss_code)
        self.select_recipe(c.loss_code)
        self.update_polarization_mode(c.polarization_count)
        self._loading=False;self.refresh_context()

    def reset_targets(self,aperture,target):
        choices=["Focus","OAM mode","Image intensity","Text intensity"]
        if aperture=="Logic gates":choices.append("Logic detectors")
        if aperture=="Handwritten digits":choices=["Digit detectors"]
        combo=self.fields["target"];combo.clear();combo.addItems(choices);combo.setCurrentText(target)

    def select_recipe(self,code):
        matching=recipe_name(code)
        if matching:self.example_combo.setCurrentText(matching)
        else:self.example_combo.setCurrentIndex(-1);self.example_combo.setPlaceholderText("Custom objective — choose a recipe to replace it")

    def read_fields(self):
        data=dict(name=self.config_name,initial_phases=self.initial_phases,loss_code=self.editor.toPlainText())
        for key,w in self.fields.items():
            if isinstance(w,SpacingsEditor):value=w.values()
            elif isinstance(w,W.QComboBox):value=w.currentText()
            elif isinstance(w,W.QLineEdit):value=w.text()
            else:value=w.value()
            if key=="gates":value=[v.strip().upper() for v in value.split(",") if v.strip()]
            data[key]=value
        return data

    def store_channel(self):
        data=self.read_fields()
        self.channel_cache[self.channel_index]={k:v for k,v in data.items() if k in CHANNEL_FIELDS}

    def get_config(self):
        self.store_channel();data=self.read_fields()
        data.update(self.channel_cache[0]);data["channel2"]=self.channel_cache[1]
        c=Config(**data);c.validate();return c

    def switch_channel(self,index):
        if index==self.channel_index:return
        try:self.store_channel()
        except Exception as exc:
            self.channel_combo.setCurrentIndex(self.channel_index);self.show_error(str(exc));return
        self.channel_index=index
        data=self.channel_cache[index]
        self._loading=True;self.reset_targets(data["aperture"],data["target"])
        for key,value in data.items():
            if key not in self.fields:continue
            w=self.fields[key]
            if isinstance(w,W.QComboBox):w.setCurrentText(str(value))
            elif isinstance(w,W.QLineEdit):w.setText(", ".join(map(str,value)) if isinstance(value,list) else str(value))
            else:w.setValue(value)
        self.editor.setPlainText(data["loss_code"]);self.initial_phases=data.get("initial_phases","")
        self.select_recipe(data["loss_code"])
        self._loading=False;self.refresh_context()
        self.status.setText(f"Editing Polarization {index+1}: input, target and loss are independent. Optical geometry is shared.")

    def update_polarization_mode(self,count):
        if not hasattr(self,"channel_cache"):return
        if count==1 and self.channel_index==1:self.switch_channel(0);self.channel_combo.setCurrentIndex(0)
        self.channel_combo.setEnabled(count==2)

    def insert_example(self):
        name=self.example_combo.currentText()
        if name in EXAMPLES:self.editor.setPlainText(EXAMPLES[name])

    def pick_file(self,edit):
        path,_=W.QFileDialog.getOpenFileName(self,"Choose image or numeric array",str(runtime_directory()),"Inputs (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.gif *.webp *.npy *.csv *.txt);;All files (*)")
        if path:edit.setText(path)

    def save_config(self):
        try:c=self.get_config()
        except Exception as exc:self.show_error(str(exc));return
        path,_=W.QFileDialog.getSaveFileName(self,"Save experiment",str(runtime_directory()/"experiment.json"),"JSON (*.json)")
        if path:Path(path).write_text(json.dumps(c.to_dict(),indent=2,ensure_ascii=False),encoding="utf-8")

    def load_config(self):
        path,_=W.QFileDialog.getOpenFileName(self,"Load experiment",str(runtime_directory()),"JSON (*.json)")
        if path:
            try:
                c=Config(**json.loads(Path(path).read_text(encoding="utf-8"))).resolve_paths(Path(path).parent);c.validate();self.set_config(c)
            except Exception as exc:self.show_error(str(exc))

    def import_phases(self):
        path,_=W.QFileDialog.getOpenFileName(self,"Import trained phases",str(runtime_directory()/"results"),"Phase stack (*.npz)")
        if path:self.initial_phases=path;self.status.setText(f"Initial phases: {path}. Load the matching experiment configuration first.")

    def start(self,preview=False,reexport=False):
        if self.thread:return
        if reexport and self.result is None:return
        try:c=self.result.config if reexport else self.get_config()
        except Exception as exc:self.show_error(str(exc));return
        self.preview=preview;self.stop_event=threading.Event()
        self.live_cache={};self.live_pol.setCurrentIndex(0)
        self.train_btn.setEnabled(False);self.preview_btn.setEnabled(False);self.stop_btn.setEnabled(True)
        self.reexport_btn.setEnabled(False)
        for i in range(4):self.pages.widget(i).setEnabled(False)
        for button in self.config_buttons:button.setEnabled(False)
        self.preset_combo.setEnabled(False);self.channel_combo.setEnabled(False)
        self.progress.setValue(0);self.status.setText("Preparing propagation operators and validating Python objective…")
        self.summary.setText(f"{c.nx} × {c.ny} CELLS  /  {c.layers} SURFACE(S)  /  {c.frequency_thz:g} THz")
        self.thread=C.QThread();self.worker=Worker(c,self.stop_event,preview,self.output_base,self.preferences.copy(),
                                                 existing_result=self.result if reexport else None);self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run);self.worker.update.connect(self.update_progress)
        self.worker.done.connect(self.finished);self.worker.failed.connect(self.show_error)
        self.worker.done.connect(self.thread.quit);self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.cleanup);self.thread.start()

    @C.Slot(dict)
    def update_progress(self,data):
        if "message" in data:self.status.setText(data["message"]);return
        self.progress.setValue(round(100*data["epoch"]/data["epochs"]))
        prefix=f"POLARIZATION {data['polarization']} / " if "polarization" in data else ""
        self.epoch_label.setText(prefix+f"EPOCH {data['epoch']:04d} / {data['epochs']:04d}")
        self.metrics_label.setText(f"LOSS {data['loss']:.6g}     /     {data['seconds']:.1f}s     /     {data['device'].upper()}")
        self.device_label.setText(data["device"].upper()+" / ACTIVE")
        self.active_polarization=data.get("polarization",1);self.live_cache[self.active_polarization]=data
        elapsed=data["seconds"];eta=elapsed/max(data["epoch"],1)*(data["epochs"]-data["epoch"])
        initial=abs(data["history"][0]);normalized=data["loss"]/max(initial,1e-16)
        extra="   /   ".join(f"{k}: {v:.4g}" for k,v in data.get("metrics",{}).items())
        self.metrics_label.setText(f"LOSS {data['loss']:.6g}  /  NORMALIZED {normalized:.5f}\nELAPSED {elapsed:.1f}s  /  ETA {eta:.1f}s  /  {data['device'].upper()}"+("\n"+extra if extra else ""))
        self.render_live()
        self.status.setText("Checking fields and gradients…" if self.preview else "Training phase surfaces. Results export automatically when complete.")

    def render_live(self,*args):
        channel=self.active_polarization if self.live_pol.currentIndex()==0 else self.live_pol.currentIndex()
        if channel not in self.live_cache:return
        data=self.live_cache[channel].copy()
        names=data.get("sample_names",["input"])
        if [self.live_sample.itemText(i) for i in range(self.live_sample.count())]!=names:
            self.live_sample.blockSignals(True);self.live_sample.clear();self.live_sample.addItems(names);self.live_sample.blockSignals(False)
        phases=data.get("all_surface_phases",np.asarray([data["phase"]]))
        if self.live_surface.count()!=len(phases):
            self.live_surface.blockSignals(True);self.live_surface.clear();self.live_surface.addItems([f"Surface {i+1}" for i in range(len(phases))]);self.live_surface.blockSignals(False)
        sample=max(0,self.live_sample.currentIndex());surface=max(0,self.live_surface.currentIndex())
        if "all_inputs" in data:data["input"]=data["all_inputs"][sample]
        view=self.live_view.currentText()
        key={"Intensity":"all_intensity","Phase":"all_output_phase","Amplitude":"all_amplitude"}[view]
        if key in data:data["intensity"]=data[key][sample]
        data["phase"]=np.angle(np.exp(1j*phases[surface]));data["surface"]=surface+1
        data["normalized"]=self.live_loss.currentIndex()==0;data["output_view"]=view
        self.canvas.update_data(data)

    @C.Slot(object,object)
    def finished(self,result,path):
        self.result=result
        channels=result.channels if hasattr(result,"channels") else [result]
        for i,ch in enumerate(channels,1):
            self.live_cache[i]=dict(input=abs(array(ch.context["input_field"])[0]),
                intensity=array(ch.context["intensity"])[0],phase=array(ch.model.phases[0]),history=ch.history,
                all_inputs=abs(array(ch.context["input_field"])),all_intensity=array(ch.context["intensity"]),
                all_output_phase=array(ch.context["phase"]),all_amplitude=array(ch.context["amplitude"]),
                all_surface_phases=np.stack([array(p) for p in ch.model.phases]),sample_names=ch.context["sample_names"],
                detectors=detector_overlay(ch.context,ch.config))
        self.render_live()
        self.metrics_label.setText("DEVICE: "+result.device.upper()+"\n"+"   /   ".join(f"{k}: {v:.5g}" for k,v in result.metrics.items()))
        if self.preview:
            self.status.setText(f"Preview passed: finite scalar loss and surface gradients on {result.device.upper()}.")
        else:
            self.result_dir=Path(path);self.result_text.setPlainText(json.dumps(result.metrics,indent=2)+f"\n\n{path}")
            self.plot_list.clear()
            for p in sorted(self.result_dir.rglob("*.png")):self.plot_list.addItem(str(p.relative_to(self.result_dir)))
            warning=" Some custom plots failed; see summary.json." if json.loads((path/"summary.json").read_text())["plot_errors"] else ""
            self.status.setText(("Interrupted run exported: " if result.cancelled else "Training complete: ")+str(path)+warning)
            self.convention_popup=W.QMessageBox(self)
            self.convention_popup.setWindowTitle("CST phase export / axes and convention")
            self.convention_popup.setIcon(W.QMessageBox.Information)
            self.convention_popup.setText("Phase maps and CST coordinate tables were exported.")
            self.convention_popup.setInformativeText(export_explanation(result.config))
            self.convention_popup.setStandardButtons(W.QMessageBox.Ok)
            self.convention_popup.open()  # nonblocking: training and GUI cleanup finish normally

    def cleanup(self):
        self.worker.deleteLater();self.thread.deleteLater();self.thread=None
        self.train_btn.setEnabled(True);self.preview_btn.setEnabled(True);self.stop_btn.setEnabled(False)
        self.reexport_btn.setEnabled(self.result is not None and not self.preview)
        for i in range(self.pages.count()):self.pages.widget(i).setEnabled(True)
        for button in self.config_buttons:button.setEnabled(True)
        self.preset_combo.setEnabled(True);self.update_polarization_mode(self.fields["polarization_count"].value())
        device=self.result.device.split(",")[0].upper() if self.result else ("CUDA" if torch.cuda.is_available() else "CPU")
        self.device_label.setText(device+" / READY")
        self.run_complete.emit()

    def stop_run(self):
        if self.thread:self.stop_event.set();self.status.setText("Stopping after the current epoch, then exporting…");self.stop_btn.setEnabled(False)

    @C.Slot(str)
    def show_error(self,message):
        self.status.setText("Check the experiment settings or Python objective.")
        W.QMessageBox.critical(self,"Experiment needs attention",message)

    def open_results(self):
        path=self.result_dir or runtime_directory()/"results"
        path.mkdir(parents=True,exist_ok=True);G.QDesktopServices.openUrl(C.QUrl.fromLocalFile(str(path)))

    def open_plot(self,item):
        if self.result_dir:G.QDesktopServices.openUrl(C.QUrl.fromLocalFile(str(self.result_dir/item.text())))

    def closeEvent(self,event):
        if self.thread:
            self.stop_run();event.ignore();self.status.setText("Stopping and exporting. Close the window once the run has finished.")
        else:event.accept()
