# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
"""Numerical regression against untouched reference snapshots, plus actual training.

Run from application: python -m pytest tests -q
The original project is never imported or written to.
"""
from dataclasses import replace
from pathlib import Path
import importlib.util
import sys
import threading
import numpy as np
import pytest
import torch

from onnlab.config import Config
from onnlab.presets import preset
from onnlab.physics import DiffractiveNetwork, Propagator, choose_device
from onnlab.inputs import build_inputs, gate_aperture, load_array, text_image, GATES
from onnlab.objectives import EXAMPLES, Objective
from onnlab.training import train, prepare
from onnlab.export import export_result

ROOT=Path(__file__).resolve().parents[1]


def reference():
    if not (ROOT/"reference/train.py").is_file():
        pytest.skip("Optional original-paper snapshots are not part of the clean public checkout.")
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(ROOT/"reference"))
    spec=importlib.util.spec_from_file_location("paper_reference",ROOT/"reference/train.py")
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("channel",["x","y"])
def test_original_apertures(channel):
    ref=reference()
    gates=["OR","AND","XOR","BUFFER"] if channel=="x" else ["NOR","NAND","XNOR","NOT"]
    for gate in gates:
        for bits in ([[0],[1]] if gate in ["NOT","BUFFER"] else [[0,0],[0,1],[1,0],[1,1]]):
            old=ref.generate_aperture(gate,bits,0,0,32,32,0).numpy()
            np.testing.assert_array_equal(old,gate_aperture(gate,bits,32,32))


@pytest.mark.parametrize("device",["cpu"]+(["cuda"] if torch.cuda.is_available() else []))
def test_paper_forward_loss_gradient_and_adam(device):
    ref=reference();ref.DEVICE=torch.device(device)
    c=preset("Paper • x polarization");c.device="CPU" if device=="cpu" else "CUDA preferred"
    new,builder,objective=prepare(c,torch.device(device));old=ref.DiffractiveNetwork()
    samples,_=ref.build_samples();masks=ref.detector_masks(*ref.make_grid(32,32))
    # Check a nonzero phase, not only the all-zero trivial case.
    torch.manual_seed(123)
    phase=torch.randn(32,32,device=device)*.2
    with torch.no_grad():old.phases[0].copy_(phase);new.phases[0].copy_(phase)
    torch.testing.assert_close(old.H_0,new.propagators[0].H,rtol=0,atol=0)
    torch.testing.assert_close(old.H_1,new.propagators[1].H,rtol=0,atol=0)
    ctx=builder()
    expected=torch.stack([old(s["field"]).reshape(32,32) for s in samples])
    torch.testing.assert_close(ctx["field"],expected,rtol=3e-6,atol=3e-7)
    opt_old=torch.optim.Adam([{"params":list(old.phases),"lr":c.learning_rate}])
    opt_new=torch.optim.Adam([{"params":list(new.phases),"lr":c.learning_rate}])
    for _ in range(3):
        opt_old.zero_grad();opt_new.zero_grad()
        loss=torch.zeros((),device=device)
        for sample in samples:
            e0,e1,out=ref.energy_fractions(old,sample,masks)
            t0,t1=(0.,ref.DESIRED_FRAC) if sample["expected"] else (ref.DESIRED_FRAC,0.)
            loss=loss+sample["weight"]*((e0-t0)**2+(e1-t1)**2+ref.LAMBDA_OUTSIDE*out**2)
        loss=loss/len(samples)
        new_loss=objective.checked_loss(builder())
        torch.testing.assert_close(new_loss,loss,rtol=2e-6,atol=2e-7)
        loss.backward();new_loss.backward()
        torch.testing.assert_close(new.phases[0].grad,old.phases[0].grad,rtol=5e-4,atol=5e-8)
        torch.nn.utils.clip_grad_norm_(old.parameters(),1.);torch.nn.utils.clip_grad_norm_(new.parameters(),1.)
        opt_old.step();opt_new.step()
    torch.testing.assert_close(new.phases[0],old.phases[0],rtol=1e-4,atol=2e-6)


def test_fft_matches_dense_anisotropic_forward_and_gradient():
    c=Config(nx=18,ny=14,dx_um=110,dy_um=173)
    dense=Propagator(c,(14,18),(14,18),3*c.wavelength,torch.device("cpu"))
    fft=Propagator(replace(c,propagation="FFT RS"),(14,18),(14,18),3*c.wavelength,torch.device("cpu"))
    u=torch.randn(14,18,dtype=torch.complex64,requires_grad=True)
    a,b=dense(u),fft(u)
    torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-5)
    ga=torch.autograd.grad(a.abs().square().mean(),u,retain_graph=True)[0]
    gb=torch.autograd.grad(b.abs().square().mean(),u)[0]
    torch.testing.assert_close(ga,gb,rtol=2e-3,atol=1e-6)


def test_no_cuda_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    assert choose_device().type=="cpu"
    assert choose_device("CUDA preferred").type=="cpu"


@pytest.mark.parametrize("name",list(EXAMPLES))
def test_every_custom_objective_has_finite_gradient(name):
    c=preset("OAM conversion" if name=="OAM / complex mode overlap" else "Gaussian focus")
    if name=="Paper logic gates":c=preset("Paper • x polarization")
    if name=="Digit recognition / ten detectors":c=preset("Advanced • four-layer digit router");c.digit_samples_per_class=1
    c.loss_code=EXAMPLES[name];c.device="CPU";c.epochs=1
    result=train(c,validate_only=True)
    assert np.isfinite(result.metrics["final_loss"])


@pytest.mark.parametrize("name",["Gaussian focus","OAM conversion","Text hologram","Double-slit steering","Two-plane focus"])
def test_examples_really_learn(name):
    c=preset(name);c.epochs=20;c.device="Auto (CUDA / CPU)"
    result=train(c)
    assert result.metrics["final_loss"] < result.metrics["initial_loss"]


def test_export_and_phase_reload(tmp_path,monkeypatch):
    import onnlab.export as exporter
    calls=[];original=exporter.save_plot
    def capture(*args,**kwargs):
        calls.append((str(args[0]),kwargs.get("cmap")));return original(*args,**kwargs)
    monkeypatch.setattr(exporter,"save_plot",capture)
    c=preset("Gaussian focus");c.epochs=3;c.device="CPU"
    result=train(c);out=export_result(result,tmp_path,plot_settings=dict(input_cmap="winter",intensity_cmap="hot",phase_cmap="jet"))
    assert any("output_phase" in name and cmap=="jet" for name,cmap in calls)
    assert any("input_amplitude" in name and cmap=="winter" for name,cmap in calls)
    assert any("output_intensity" in name and cmap=="hot" for name,cmap in calls)
    assert (out/"summary.json").is_file()
    table=np.loadtxt(out/"best_meta_layer_1.txt",skiprows=1)
    assert table.shape==(1024,4)
    c.initial_phases=str(out/"phases.npz")
    loaded=train(c,validate_only=True)
    np.testing.assert_allclose(loaded.context["field"].numpy(),result.context["field"].numpy(),rtol=0,atol=0)


def test_cancellation():
    c=preset("Gaussian focus");c.epochs=20;c.device="CPU"
    stop=threading.Event()
    result=train(c,progress=lambda data:stop.set(),stop=stop)
    assert result.cancelled and len(result.history)==1


def test_import_files_and_validation(tmp_path):
    a=np.ones((16,20),dtype=np.complex64)*np.exp(1j*.8)
    np.save(tmp_path/"field.npy",a)
    np.testing.assert_array_equal(load_array(tmp_path/"field.npy",a.shape),a)
    with pytest.raises(ValueError):load_array(tmp_path/"field.npy",(20,16))
    with pytest.raises(ValueError):load_array(tmp_path/"field.npy",a.shape,False)
    c=preset("Gaussian focus");c.loss_code="def loss(ctx):\n    return ctx['intensity'].mean().item()"
    with pytest.raises(ValueError,match="scalar"):train(c,validate_only=True)
    c=Config(layers=2)
    with pytest.raises(ValueError,match="spacings"):c.validate()
    c=preset("Paper • x polarization");c.gates=["OR","NOR"]
    with pytest.raises(ValueError,match="Complementary"):c.validate()


def test_oam_sign_and_purity():
    c=preset("OAM conversion");c.nx=c.ny=64;c.propagation="FFT RS";c.device="CPU"
    _,builder,_=prepare(c,torch.device("cpu"));ctx=builder()
    ell,power=ctx["oam_spectrum"](ctx["target_field"][None])
    assert int(ell[power[0].argmax()])==c.target_charge
    assert float(power[0,ell==c.target_charge])>.95


def test_two_independent_polarizations(tmp_path):
    c=preset("Paper • two polarizations");c.epochs=3;c.device="CPU"
    result=train(c)
    assert len(result.channels)==2
    assert result.channels[0].context["sample_names"][0]=="OR_00"
    assert result.channels[1].context["sample_names"][0]=="NOR_00"
    assert not torch.equal(result.channels[0].model.phases[0],result.channels[1].model.phases[0])
    out=export_result(result,tmp_path)
    assert (out/"Polarization_1/phases.npz").is_file()
    assert (out/"Polarization_2/phases.npz").is_file()
    c=preset("Gaussian focus");c.polarization_count=2;c.epochs=2;c.device="CPU"
    c.channel2={"beam":"OAM (LG p=0)","charge":-2,"target":"OAM mode","target_charge":-1,
                "loss_code":EXAMPLES["OAM / complex mode overlap"]}
    result=train(c)
    assert not torch.equal(result.channels[0].context["input_field"],result.channels[1].context["input_field"])
    assert result.channels[1].config.target_charge==-1


@pytest.mark.parametrize("index",[1,2])
def test_archived_paper_solution(index):
    if not (ROOT/f"reference/paper_polarization_{index}.txt").is_file():
        pytest.skip("Optional archived paper phases are not part of the clean public checkout.")
    ref=reference();ref.DEVICE=torch.device("cpu")
    c=preset(f"Paper • Polarization {index}");c.device="CPU"
    ref.GATE_TYPES=c.gates
    new,builder,objective=prepare(c,torch.device("cpu"));old=ref.DiffractiveNetwork()
    phase=np.loadtxt(ROOT/f"reference/paper_polarization_{index}.txt",skiprows=1)[:,2].reshape(32,32)
    with torch.no_grad():
        old.phases[0].copy_(torch.tensor(phase));new.phases[0].copy_(torch.tensor(phase))
        ctx=builder();samples,_=ref.build_samples()
        expected=torch.stack([old(s["field"]).reshape(32,32) for s in samples])
        torch.testing.assert_close(ctx["field"],expected,rtol=3e-6,atol=3e-7)
        e=ctx["detector_fractions"]
        hot=ctx["labels"]
        assert (e[:,:2].argmax(1)==hot).all()
        contrast=10*torch.log10((e[torch.arange(len(e)),hot]+1e-16)/(e[torch.arange(len(e)),1-hot]+1e-16))
        assert abs(float(contrast.mean())-(24.10 if index==1 else 20.40))<.01


@pytest.mark.parametrize("method",["Adam","AdamW","SGD + momentum","RMSprop","Adagrad"])
def test_training_methods(method):
    c=preset("Gaussian focus");c.device="CPU";c.epochs=10;c.optimizer=method
    result=train(c)
    assert result.metrics["final_loss"]<result.metrics["initial_loss"]
