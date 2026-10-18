# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
"""Show a responsive lightweight launch window before importing the numerical stack."""
import json
from pathlib import Path
import sys
import traceback
import time
from PySide6 import QtCore as C,QtGui as G,QtWidgets as W
from .paths import application_directory


class Loader(C.QThread):
    ready=C.Signal(object)
    failed=C.Signal(str)

    def run(self):
        try:
            from .ui import Window
            self.ready.emit(Window)
        except Exception:self.failed.emit(traceback.format_exc())


def launch(args):
    launch_start=time.perf_counter()
    app=W.QApplication(sys.argv[:1]);app.setApplicationName("WaveForge")
    app.setQuitOnLastWindowClosed(False)
    for font in ["segoeui.ttf","seguisb.ttf","consola.ttf","seguisym.ttf"]:
        path=Path("C:/Windows/Fonts")/font
        if path.exists():G.QFontDatabase.addApplicationFont(str(path))
    app.setFont(G.QFont("Segoe UI",10))
    splash=W.QWidget();splash.setWindowTitle("WaveForge / initializing")
    splash.setFixedSize(470,230)
    splash.setStyleSheet("QWidget{background:#090f1b;color:#dce6f3;font-family:'Segoe UI';} QLabel#brand{font-size:27px;font-weight:700;color:#6af6e0;letter-spacing:3px;} QProgressBar{background:#19283e;border:0;height:5px;} QProgressBar::chunk{background:#6af6e0;}")
    layout=W.QVBoxLayout(splash);layout.setContentsMargins(35,35,35,35)
    brand=W.QLabel("// WAVEFORGE");brand.setObjectName("brand");layout.addWidget(brand)
    layout.addWidget(W.QLabel("DIFFRACTIVE NEURAL LAB"));layout.addStretch()
    status=W.QLabel("Loading numerical engine…");layout.addWidget(status)
    bar=W.QProgressBar();bar.setRange(0,0);bar.setTextVisible(False);layout.addWidget(bar)
    splash.show();app.splash=splash
    splash_seconds=time.perf_counter()-launch_start
    messages=["Loading numerical engine…","Preparing CUDA / CPU runtime…","Preparing optical workspace…"]
    timer=C.QTimer();timer.setInterval(700);tick=[0]
    def animate():
        tick[0]+=1;status.setText(messages[min(tick[0]//5,2)])
    timer.timeout.connect(animate);timer.start();app.launch_timer=timer
    loader=Loader();app.loader=loader

    def loaded(Window):
        try:
            if not splash.isVisible():
                if loader.isRunning():loader.finished.connect(app.quit)
                else:app.quit()
                return
            window=Window(output_base=args.output);app.window=window
            if args.config:
                from .config import Config
                c=Config(**json.loads(args.config.read_text(encoding="utf-8"))).resolve_paths(args.config.resolve().parent)
                c.validate();window.set_config(c)
            elif args.preset!="Gaussian focus":window.load_preset(args.preset)
            if args.cpu:window.fields["device"].setCurrentText("CPU")
            if args.epochs:window.fields["epochs"].setValue(args.epochs)
            window.show();splash.close();timer.stop();app.setQuitOnLastWindowClosed(True)
            if args.smoke_gui:
                ready_seconds=time.perf_counter()-launch_start
                gaps=[];last_tick=[time.perf_counter()]
                heartbeat=C.QTimer(window);heartbeat.setInterval(50)
                def beat():
                    now=time.perf_counter();gaps.append(now-last_tick[0]);last_tick[0]=now
                heartbeat.timeout.connect(beat);heartbeat.start();window.qa_heartbeat=heartbeat
                def done():
                    args.smoke_gui.parent.mkdir(parents=True,exist_ok=True)
                    # Public screenshots show the actual app without machine-specific folder names.
                    window.status.setText("Training complete. Phase maps, fields and reports are saved in results/ beside the app.")
                    window.grab().save(str(args.smoke_gui))
                    popup=getattr(window,"convention_popup",None)
                    if popup:
                        popup.grab().save(str(args.smoke_gui.with_name(args.smoke_gui.stem+"_cst_popup.png")))
                    window.navigate(1)
                    window.grab().save(str(args.smoke_gui.with_name(args.smoke_gui.stem+"_inputs.png")))
                    if window.result and window.result.config.target=="Digit detectors":
                        window.live_surface.setCurrentIndex(window.result.config.layers-1)
                        window.grab().save(str(args.smoke_gui.with_name(args.smoke_gui.stem+"_last_surface.png")))
                    window.navigate(5)
                    window.grab().save(str(args.smoke_gui.with_name(args.smoke_gui.stem+"_guide.png")))
                    if args.smoke_gui.stem=="frozen_ui":
                        window.guide_tabs.setCurrentIndex(3)
                        window.grab().save(str(args.smoke_gui.with_name("loss_workshop.png")))
                        window.navigate(6)
                        original=window.preferences.copy()
                        from .appearance import THEMES
                        for index,theme in enumerate(THEMES):
                            combo=window.appearance_controls["theme"];combo.blockSignals(True);combo.setCurrentText(theme);combo.blockSignals(False)
                            window.apply_appearance(False)
                            window.grab().save(str(args.smoke_gui.with_name(f"theme_{index+1}.png")))
                        combo=window.appearance_controls["theme"];combo.blockSignals(True);combo.setCurrentText(original["theme"]);combo.blockSignals(False)
                        window.apply_appearance(False)
                    if window.result:
                        receipt=dict(device=window.result.device,metrics=window.result.metrics,
                                     result_folder=str(window.result_dir),config=window.result.config.to_dict(),
                                     application_module=str(sys.modules[Window.__module__].__file__),
                                     first_window_seconds=splash_seconds,workspace_ready_seconds=ready_seconds,
                                     ui_heartbeat_ticks=len(gaps),ui_gap_p95_ms=sorted(gaps)[min(len(gaps)-1,int(.95*len(gaps)))]*1000 if gaps else None,
                                     ui_gap_max_ms=max(gaps)*1000 if gaps else None)
                        args.smoke_gui.with_suffix(".json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
                    app.quit()
                window.run_complete.connect(done)
                C.QTimer.singleShot(400,lambda:window.start(not bool(args.epochs)))
                C.QTimer.singleShot(max(90000,(args.epochs or 1)*150+30000),lambda:app.exit(3))
        except Exception:failed(traceback.format_exc())

    def failed(message):
        root=application_directory()
        (root/"waveforge_error.log").write_text(message,encoding="utf-8")
        W.QMessageBox.critical(splash,"WaveForge could not start",message)
        app.exit(1)

    loader.ready.connect(loaded);loader.failed.connect(failed)
    C.QTimer.singleShot(0,loader.start)
    return app.exec()
