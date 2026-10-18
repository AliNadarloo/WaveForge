# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
"""Mini-batch optical classification, with a fixed held-out set and bounded memory."""
import time
import numpy as np
import torch
from .physics import choose_device
from .digits import split_indices, detector_overlay
from .training import Result, prepare, make_optimizer


def train_digits(c,progress=None,stop=None,validate_only=False):
    start=time.perf_counter();device=choose_device(c.device)
    try:model,context,objective=prepare(c,device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache();device=torch.device("cpu")
        if progress:progress({"message":"GPU memory exhausted during setup; using CPU."})
        model,context,objective=prepare(c,device)
    optimizer=make_optimizer(c,model)
    train_ids,test_ids=split_indices(context.labels,c.seed,c.digit_samples_per_class)
    all_labels=np.asarray(context.label_values)
    representatives=[int(test_ids[all_labels[test_ids]==digit][0]) for digit in range(10)]
    batch_size=c.digit_batch_size

    @torch.no_grad()
    def evaluate(ids):
        losses=[];scores=[]
        for offset in range(0,len(ids),batch_size):
            selection=ids[offset:offset+batch_size].tolist()
            ctx=context(indices=selection)
            value=objective.loss(ctx)
            if not isinstance(value,torch.Tensor) or value.numel()!=1 or not bool(torch.isfinite(value)):
                raise ValueError("Digit objective must return one finite real scalar.")
            losses.append(float(value)*len(selection))
            scores.append(ctx["digit_fractions"].cpu().numpy())
        scores=np.concatenate(scores)
        accuracy=float(np.mean(scores.argmax(1)==all_labels[ids]))
        return sum(losses)/len(ids),accuracy,scores

    initial,_,_=evaluate(train_ids)
    history=[];learning=[];cancelled=False;last_refresh=0.;heldout_accuracy=0.
    rng=np.random.default_rng(c.seed)
    limit=1 if validate_only else c.epochs
    for epoch in range(1,limit+1):
        shuffled=rng.permutation(train_ids);weighted=0.;count=0
        correct=torch.zeros((),device=device)
        for offset in range(0,len(shuffled),batch_size):
            if stop and stop.is_set():cancelled=True;break
            selection=shuffled[offset:offset+batch_size].tolist()
            optimizer.zero_grad();ctx=context(epoch,selection)
            loss=objective.checked_loss(ctx);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            if not validate_only:optimizer.step()
            weighted+=float(loss.detach())*len(selection);count+=len(selection)
            correct+=(ctx["digit_fractions"].argmax(1)==ctx["labels"]).sum().detach()
            del ctx,loss
            if validate_only:break
        if not count:break
        value=weighted/count;history.append(value);training_accuracy=float(correct)/count
        if epoch==1 or epoch%5==0 or epoch==limit or cancelled:
            _,heldout_accuracy,_=evaluate(test_ids)
            learning.append([epoch,value,training_accuracy,heldout_accuracy])
        now=time.perf_counter()
        if progress and (epoch==1 or epoch==limit or now-last_refresh>=.15 or cancelled):
            last_refresh=now
            with torch.no_grad():
                display=context(epoch,representatives)
                live_metrics={"preview_"+str(k):float(v) for k,v in objective.metrics(display).items()}
            def preview(t):
                sy=max(1,(t.shape[-2]+255)//256);sx=max(1,(t.shape[-1]+255)//256)
                return t[...,::sy,::sx].detach().cpu().numpy()
            phases=np.stack([p.detach().cpu().numpy() for p in model.phases])
            progress(dict(epoch=epoch,epochs=limit,loss=value,history=history.copy(),device=str(device),
                seconds=now-start,intensity=preview(display["intensity"][0]),phase=phases[0],
                input=preview(display["input_field"][0].abs()),all_intensity=preview(display["intensity"]),
                all_output_phase=preview(display["phase"]),all_amplitude=preview(display["amplitude"]),
                all_inputs=preview(display["input_field"].abs()),all_surface_phases=phases,
                sample_names=display["sample_names"],detectors=detector_overlay(display,c),
                metrics=dict(training_accuracy=training_accuracy,heldout_accuracy=heldout_accuracy,**live_metrics)))
            del display
        if cancelled:break
    final,train_accuracy,train_scores=evaluate(train_ids)
    _,heldout_accuracy,test_scores=evaluate(test_ids)
    confusion=np.zeros((10,10),dtype=int)
    np.add.at(confusion,(all_labels[test_ids],test_scores.argmax(1)),1)
    with torch.no_grad():
        ctx=context(indices=representatives)
        custom_metrics={str(k):float(v) for k,v in objective.metrics(ctx).items()}
        plots=objective.plots(ctx)
        if not isinstance(plots,dict):raise ValueError("plots(ctx) must return a dictionary.")
    if not all(np.isfinite(v) for v in custom_metrics.values()):raise ValueError("A custom digit metric is not finite.")
    metrics=dict(initial_loss=initial,final_loss=final,training_accuracy=train_accuracy,
                 heldout_accuracy=heldout_accuracy,training_samples=len(train_ids),heldout_samples=len(test_ids),
                 mean_correct_detector_fraction=float(test_scores[np.arange(len(test_ids)),all_labels[test_ids]].mean()))
    # Custom metrics refer to the ten representative held-out previews.
    metrics.update({"preview_"+k:v for k,v in custom_metrics.items()})
    report=dict(train_indices=train_ids,heldout_indices=test_ids,labels=all_labels,
                train_scores=train_scores,heldout_scores=test_scores,confusion=confusion,
                learning_curve=np.asarray(learning).reshape(-1,4),representative_indices=np.asarray(representatives))
    return Result(c,model,ctx,history,metrics,plots,time.perf_counter()-start,str(device),cancelled,report)
