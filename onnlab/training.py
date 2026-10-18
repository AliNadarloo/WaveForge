# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
"""Training loop with cancellation, gradient checks and CPU fallback."""
from dataclasses import dataclass
import time
import numpy as np
import torch
from .physics import choose_device, DiffractiveNetwork
from .inputs import build_inputs
from .objectives import Objective, ContextBuilder


@dataclass
class Result:
    config: object
    model: object
    context: dict
    history: list
    metrics: dict
    plots: dict
    seconds: float
    device: str
    cancelled: bool
    classification: object = None


@dataclass
class PolarizationResult:
    config: object
    channels: list
    seconds: float

    @property
    def metrics(self):
        return {f"P{i+1}_{key}":value for i,ch in enumerate(self.channels) for key,value in ch.metrics.items()}

    @property
    def cancelled(self):
        return len(self.channels)<2 or any(ch.cancelled for ch in self.channels)

    @property
    def device(self):
        return ", ".join(ch.device for ch in self.channels)

    # Default live display is channel 1; all channels are separately exported.
    @property
    def context(self):return self.channels[0].context
    @property
    def model(self):return self.channels[0].model
    @property
    def history(self):return self.channels[0].history


def prepare(c, device):
    c.validate()
    torch.manual_seed(c.seed)
    np.random.seed(c.seed)
    inputs,names,labels,weights = build_inputs(c,device)
    model = DiffractiveNetwork(c,device)
    if c.initial_phases:
        with np.load(c.initial_phases,allow_pickle=False) as saved:
            phases = saved["phases_rad"]
        if phases.shape != (c.layers,c.ny,c.nx) or not np.isfinite(phases).all():
            raise ValueError("Loaded phase stack must match [layers, Ny, Nx] and be finite.")
        with torch.no_grad():
            for p,a in zip(model.phases,phases):
                p.copy_(torch.as_tensor(a,device=device))
    context = ContextBuilder(c,model,inputs,names,labels,weights)
    return model,context,Objective(c.loss_code)


def make_optimizer(c,model):
    parameters=[{"params":list(model.phases),"lr":c.learning_rate}]
    options={"weight_decay":c.weight_decay}
    if c.optimizer=="Adam":return torch.optim.Adam(parameters,**options)
    if c.optimizer=="AdamW":return torch.optim.AdamW(parameters,**options)
    if c.optimizer=="SGD + momentum":return torch.optim.SGD(parameters,momentum=c.momentum,**options)
    if c.optimizer=="RMSprop":return torch.optim.RMSprop(parameters,momentum=c.momentum,**options)
    return torch.optim.Adagrad(parameters,**options)


def train(c, progress=None, stop=None, validate_only=False):
    start = time.perf_counter()
    if c.polarization_count==2:
        c.validate()
        channels=[]
        for index,channel in enumerate(c.channels()):
            if index and stop and stop.is_set():break
            def channel_progress(data):
                data=data.copy();data["polarization"]=index+1
                data["seconds"]=time.perf_counter()-start
                if "epoch" in data:
                    data["channel_epoch"]=data["epoch"]
                    limit=1 if validate_only else c.epochs
                    data["epoch"]+=index*limit;data["epochs"]=2*limit
                if progress:progress(data)
            channels.append(train(channel,channel_progress,stop,validate_only))
        return PolarizationResult(c,channels,time.perf_counter()-start)
    if c.aperture=="Handwritten digits":
        from .digit_training import train_digits
        return train_digits(c,progress,stop,validate_only)
    device = choose_device(c.device)
    try:
        model,context,objective = prepare(c,device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        device = torch.device("cpu")
        if progress:
            progress({"message":"GPU memory exhausted during setup; using CPU."})
        model,context,objective = prepare(c,device)
    optimizer = make_optimizer(c,model)
    history=[]; cancelled=False; last_refresh=0.0
    limit=1 if validate_only else c.epochs
    for epoch in range(1,limit+1):
        if stop and stop.is_set():
            cancelled=True
            break
        optimizer.zero_grad()
        ctx=context(epoch)
        loss=objective.checked_loss(ctx)
        loss.backward()
        gradients=[p.grad for p in model.phases if p.grad is not None]
        if not gradients:
            raise ValueError("No surface gradient. Check your Python loss.")
        # One norm check also detects nonfinite gradients, avoiding a GPU sync
        # per surface before the same norm is computed again for clipping.
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        value=float(loss.detach())
        history.append(value)
        if not validate_only:
            optimizer.step()
        now=time.perf_counter()
        if progress and (epoch==1 or epoch==limit or now-last_refresh>=0.15):
            last_refresh=now
            with torch.no_grad():
                display=context(epoch)
                live_metrics={str(k):float(v) for k,v in objective.metrics(display).items()}
            def preview_array(tensor):
                # Downsample only the preview before copying from GPU. Full
                # resolution stays in the model and is exported after training.
                sy=max(1,(tensor.shape[-2]+255)//256);sx=max(1,(tensor.shape[-1]+255)//256)
                return tensor[...,::sy,::sx].detach().cpu().numpy()
            progress({"epoch":epoch,"epochs":limit,"loss":value,"history":history.copy(),
                      "device":str(device),"seconds":time.perf_counter()-start,
                      "intensity":preview_array(display["intensity"][0]),
                      "phase":model.phases[0].detach().cpu().numpy(),
                      "input":preview_array(display["input_field"][0].abs()),
                      "all_intensity":preview_array(display["intensity"]),
                      "all_output_phase":preview_array(display["phase"]),
                      "all_amplitude":preview_array(display["amplitude"]),
                      "all_inputs":preview_array(display["input_field"].abs()),
                      "all_surface_phases":np.stack([p.detach().cpu().numpy() for p in model.phases]),
                      "sample_names":display["sample_names"],"metrics":live_metrics})
            del display
        del ctx,loss
    with torch.no_grad():
        ctx=context()
        final_loss=objective.loss(ctx)
        if not isinstance(final_loss,torch.Tensor) or final_loss.numel()!=1 or not torch.isfinite(final_loss):
            raise ValueError("Final loss is not a finite scalar.")
        metrics={"final_loss":float(final_loss),"initial_loss":history[0] if history else float(final_loss)}
        for key,value in objective.metrics(ctx).items():
            scalar=float(value)
            if not np.isfinite(scalar):
                raise ValueError(f"Metric {key} is not finite.")
            metrics[str(key)]=scalar
        plots=objective.plots(ctx)
        if not isinstance(plots,dict):
            raise ValueError("plots(ctx) must return a dictionary.")
    return Result(c,model,ctx,history,metrics,plots,time.perf_counter()-start,str(device),cancelled)
