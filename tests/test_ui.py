# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
"""Exercise our own Qt widgets and worker signals without controlling other apps."""
import os
os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
import time
from pathlib import Path
from PySide6 import QtWidgets as W,QtGui as G,QtCore as C
from PySide6.QtTest import QTest
from onnlab.ui import Window


def test_ui_channel_editing_and_training(monkeypatch,tmp_path):
    import onnlab.ui as ui
    from onnlab.export import export_result
    monkeypatch.setattr(ui,"export_result",lambda result,base=None,**kwargs:export_result(result,tmp_path,**kwargs))
    app=W.QApplication.instance() or W.QApplication([])
    for font in ["segoeui.ttf","seguisb.ttf","consola.ttf","seguisym.ttf"]:
        p=Path("C:/Windows/Fonts")/font
        if p.exists():G.QFontDatabase.addApplicationFont(str(p))
    window=Window(preferences_path=tmp_path/"preferences.json");errors=[];window.show_error=lambda msg:errors.append(msg)
    window.show()
    window.navigate(5);assert window.pages.currentIndex()==5
    assert window.guide_tabs.count()==4
    window.guide_topic.setCurrentText("Results, CST & continuing a run")
    assert "−z toward +z" in window.guide_text.toPlainText()
    window.guide_preset.setCurrentText("OAM conversion");window.load_guide_preset()
    assert window.fields["target"].currentText()=="OAM mode"
    window.guide_loss.setCurrentText("Circular phase matching");window.load_guide_loss()
    assert "torch.cos" in window.editor.toPlainText()
    assert window.pages.currentIndex()==2
    window.load_preset("Paper • two polarizations")
    window.navigate(1)
    assert window.fields["charge"].isHidden()
    assert window.fields["target_charge"].isHidden()
    assert window.fields["aperture_file"].parentWidget().isHidden()
    assert window.logic_group.isVisible()
    window.fields["beam"].setCurrentText("OAM (LG p=0)")
    assert not window.fields["charge"].isHidden()
    window.fields["beam"].setCurrentText("Plane wave")
    window.show_all.setChecked(True);assert not window.fields["charge"].isHidden()
    window.show_all.setChecked(False)
    custom="def loss(ctx):\n    return ctx['intensity'].mean()\n"
    window.editor.setPlainText(custom);window.fields["target"].setCurrentText("OAM mode")
    assert window.editor.toPlainText()==custom
    window.load_preset("Paper • two polarizations")
    window.appearance_controls["phase_cmap"].setCurrentText("jet")
    window.appearance_controls["intensity_cmap"].setCurrentText("winter")
    window.appearance_controls["theme"].setCurrentText("Daylight")
    assert window.canvas.images[2].get_cmap().name=="jet"
    assert (tmp_path/"preferences.json").is_file()
    window.appearance_controls["theme"].setCurrentText("Deep space")
    window.appearance_controls["phase_cmap"].setCurrentText("twilight")
    window.appearance_controls["intensity_cmap"].setCurrentText("inferno")
    window.fields["layers"].setValue(4)
    assert len(window.fields["distances_wl"].controls)==5
    window.fields["distances_wl"].controls[-1].setValue(17.)
    assert window.read_fields()["distances_wl"][-1]==17.
    window.load_preset("Paper • two polarizations")
    window.channel_combo.setCurrentIndex(1);window.switch_channel(1)
    assert window.fields["gates"].text()=="NOR, NAND, XNOR, NOT"
    window.fields["text"].setText("SECOND")
    window.channel_combo.setCurrentIndex(0);window.switch_channel(0)
    assert window.fields["gates"].text()=="OR, AND, XOR, BUFFER"
    window.fields["epochs"].setValue(2)
    config=window.get_config()
    assert config.channel2["text"]=="SECOND"
    assert config.polarization_count==2
    # Use two small independent samples for the actual end-to-end UI run.
    # Full paper truth-table export is covered separately in test_engine.
    from onnlab.presets import preset
    config=preset("Gaussian focus");config.polarization_count=2;config.epochs=2
    config.channel2={"text":"SECOND","beam":"OAM (LG p=0)"}
    window.set_config(config)
    QTest.mouseClick(window.train_btn,C.Qt.LeftButton)
    # Appearance remains editable while the worker trains; export uses the latest choice.
    window.appearance_controls["theme"].setCurrentText("Solar amber")
    window.appearance_controls["input_cmap"].setCurrentText("winter")
    # Qt's native loop releases the GIL for the Python training worker.
    loop=C.QEventLoop()
    window.run_complete.connect(loop.quit)
    C.QTimer.singleShot(90000,loop.quit)
    if window.thread:loop.exec()
    assert window.thread is None
    assert not errors,errors
    assert window.result_dir is not None
    assert (window.result_dir/"Polarization_2/phases.npz").is_file()
    import json
    saved=json.loads((window.result_dir/"plot_settings.json").read_text())
    assert saved["theme"]=="Solar amber" and saved["input_cmap"]=="winter"
    assert window.train_btn.isEnabled()
    assert window.device_label.text().startswith(window.result.device.split(",")[0].upper())
    assert window.metrics_label.text().startswith("DEVICE:")
    assert window.convention_popup.isVisible()
    assert "−z toward +z" in window.convention_popup.informativeText()
    window.convention_popup.close()
    assert window.fields["optimizer"].currentText()=="Adam"
    window.live_pol.setCurrentIndex(2)
    window.live_loss.setCurrentIndex(0)
    assert abs(float(window.canvas.trace.get_ydata()[0])-1.0)<1e-6
    window.live_view.setCurrentText("Phase")
    assert window.canvas.images[1].get_cmap().name=="twilight"
    window.navigate(2);assert window.pages.currentIndex()==2
    window.channel_combo.setCurrentIndex(1);window.switch_channel(1)
    assert window.fields["text"].text()=="SECOND"
    # Re-export uses the finished model and creates a fresh folder without training.
    previous_result=window.result;previous_dir=window.result_dir
    monkeypatch.setattr(ui,"train",lambda *a,**k:(_ for _ in ()).throw(AssertionError("Unexpected retraining")))
    window.appearance_controls["theme"].setCurrentText("Daylight")
    window.start(reexport=True)
    loop=C.QEventLoop();window.run_complete.connect(loop.quit);C.QTimer.singleShot(90000,loop.quit)
    if window.thread:loop.exec()
    assert not errors,errors
    assert window.result is previous_result and window.result_dir!=previous_dir
    assert json.loads((window.result_dir/"plot_settings.json").read_text())["theme"]=="Daylight"
    import numpy as np
    np.testing.assert_array_equal(np.load(previous_dir/"Polarization_1/phases.npz")["phases_rad"],
                                  np.load(window.result_dir/"Polarization_1/phases.npz")["phases_rad"])
    window.convention_popup.close()
    window.close();app.processEvents()
