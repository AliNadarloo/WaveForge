# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
"""Independent NumPy and analytic checks for padded angular-spectrum propagation."""
import numpy as np
import pytest
import torch
from onnlab.config import Config
from onnlab.physics import Propagator
from onnlab.training import train
from onnlab.presets import preset
from onnlab.recommend import recommendation


@pytest.mark.parametrize("method",["ASM • evanescent decay","ASM • propagating only"])
def test_asm_against_numpy_reference(method):
    c=Config(nx=18,ny=14,dx_um=70,dy_um=90,propagation=method,samples_per_cell=1,padding_factor=3)
    shape=(14,18);z=2.4*c.wavelength
    rng=np.random.default_rng(13);u=(rng.normal(size=shape)+1j*rng.normal(size=shape)).astype(np.complex64)
    # Independent NumPy formulation of the spectral propagation transfer function.
    ny,nx=shape;big=np.zeros((ny*3,nx*3),complex);sy,sx=ny,nx
    big[sy:sy+ny,sx:sx+nx]=u
    KX,KY=np.meshgrid(2*np.pi*np.fft.fftfreq(nx*3,d=c.dx),2*np.pi*np.fft.fftfreq(ny*3,d=c.dy))
    k=2*np.pi/c.wavelength;q=KX*KX+KY*KY
    H=np.exp(-1j*np.sqrt(np.maximum(k*k-q,0))*z)
    H*=np.exp(-np.sqrt(np.maximum(q-k*k,0))*abs(z)) if "decay" in method else q<k*k
    expected=np.fft.ifft2(np.fft.fft2(big)*H)[sy:sy+ny,sx:sx+nx]
    actual=Propagator(c,shape,shape,z,torch.device("cpu"))(torch.tensor(u))
    np.testing.assert_allclose(actual.numpy(),expected,rtol=3e-4,atol=3e-6)


@pytest.mark.parametrize("order",[1,7])
def test_asm_plane_wave_phase_and_evanescent_decay(order):
    c=Config(nx=32,ny=32,dx_um=30,dy_um=30,propagation="ASM • evanescent decay",samples_per_cell=1,padding_factor=1)
    x=torch.arange(32,dtype=torch.float32)
    u=torch.exp(2j*torch.pi*order*x/32)[None].repeat(32,1)
    z=.15*c.wavelength;k=2*np.pi/c.wavelength;kx=2*np.pi*order/(32*c.dx)
    coefficient=np.exp(-1j*np.sqrt(max(k*k-kx*kx,0))*z)*np.exp(-np.sqrt(max(kx*kx-k*k,0))*z)
    actual=Propagator(c,(32,32),(32,32),z,torch.device("cpu"))(u)
    torch.testing.assert_close(actual,u*coefficient,rtol=1e-4,atol=2e-6)


@pytest.mark.parametrize("method",["ASM • evanescent decay","ASM • propagating only"])
def test_subcell_training_fields_and_power(method):
    c=preset("OAM conversion");c.nx=18;c.ny=14;c.epochs=12;c.propagation=method;c.samples_per_cell=2
    result=train(c)
    ctx=result.context
    assert ctx["field"].shape==(1,28,36)
    assert ctx["input_field"].shape==(1,28,36)
    assert result.model.phases[0].shape==(14,18)
    assert ctx["dx"]==c.dx/2 and ctx["pixel_area"]==c.dx*c.dy/4
    assert (ctx["total_power"]<=ctx["input_power"]*1.00001).all()
    assert result.metrics["final_loss"]<result.metrics["initial_loss"]


def test_asm_extra_plane_and_expanded_output():
    c=preset("Two-plane focus");c.nx=c.ny=16;c.propagation="ASM • evanescent decay"
    c.samples_per_cell=2;c.output_scale=2;c.epochs=2
    result=train(c)
    assert result.context["field"].shape==(1,64,64)
    assert result.context["at_z"](3*c.wavelength).shape==(1,64,64)


def test_solver_recommendations():
    method,_=recommendation(preset("Paper • Polarization 1"));assert method=="Dense RS"
    method,_=recommendation(preset("OAM conversion"));assert method.startswith("ASM")
    c=Config(nx=128,ny=128,dx_um=100,dy_um=100,distances_wl=[3.,10.])
    method,text=recommendation(c);assert method=="FFT RS" and "MiB" in text
    c.propagation="ASM • evanescent decay";c.samples_per_cell=8;c.padding_factor=8
    c.layers=2;c.distances_wl=[1.,2.,3.]
    with pytest.raises(ValueError,match="GiB"):c.validate()


def test_cst_coordinates_and_phase_sign(tmp_path):
    from onnlab.conventions import BASES,cst_coordinates
    from onnlab.export import export_result
    c=preset("Gaussian focus");c.epochs=2
    result=train(c);folder=export_result(result,tmp_path)
    table=np.loadtxt(folder/"CST_phase_layer_1.txt",skiprows=1)
    assert table.shape==(1024,6)
    xyz=cst_coordinates(c,0).reshape(-1,3)
    np.testing.assert_allclose(table[:,:3],xyz,rtol=1e-8)
    assert xyz[0,0]<0 and xyz[-1,0]>0
    assert xyz[0,1]<0 and xyz[-1,1]>0
    assert (xyz[:,2]>0).all()
    phase=result.model.phases[0].detach().cpu().numpy().ravel()
    np.testing.assert_allclose(np.exp(1j*table[:,3]),np.exp(1j*phase),rtol=1e-6,atol=1e-7)
    for axis,basis in BASES.items():
        assert np.linalg.det(np.asarray(basis))==1
        np.testing.assert_array_equal(np.cross(basis[0],basis[1]),basis[2])


@pytest.mark.parametrize("method",["Dense RS","FFT RS","ASM • evanescent decay"])
def test_batched_propagation_matches_individual_fields_and_gradients(method):
    from onnlab.physics import DiffractiveNetwork
    c=Config(name="Batch validation",nx=16,ny=12,propagation=method,samples_per_cell=2)
    model=DiffractiveNetwork(c,torch.device("cpu"))
    inputs=torch.randn(4,12,16,dtype=torch.complex64)
    fast=model(inputs)[0]
    fast.abs().square().mean().backward()
    gradient=model.phases[0].grad.clone();model.phases[0].grad=None
    model.paper_order=True
    individual=model(inputs)[0]
    individual.abs().square().mean().backward()
    torch.testing.assert_close(fast,individual,rtol=2e-5,atol=2e-6)
    torch.testing.assert_close(gradient,model.phases[0].grad,rtol=2e-4,atol=2e-7)
