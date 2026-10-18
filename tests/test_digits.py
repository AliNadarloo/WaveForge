# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
"""Dataset separation, numbered detector geometry and trainable optical routing."""
import numpy as np
import pytest
import torch
from onnlab.presets import preset
from onnlab.training import prepare,train
from onnlab.digits import split_indices


def test_heldout_split_is_disjoint_reproducible_and_stratified():
    c=preset("Advanced • four-layer digit router")
    _,context,_=prepare(c,torch.device("cpu"))
    training,heldout=split_indices(context.labels,c.seed)
    assert (len(training),len(heldout))==(1438,359)
    assert not set(training)&set(heldout)
    assert len(set(training)|set(heldout))==1797
    np.testing.assert_array_equal(training,split_indices(context.labels,c.seed)[0])
    assert set(context.labels[heldout].tolist())==set(range(10))
    small,test=split_indices(context.labels,c.seed,3)
    assert len(small)==30;np.testing.assert_array_equal(test,heldout)


def test_four_surfaces_detector_labels_and_output_distance():
    c=preset("Advanced • four-layer digit router");c.device="CPU"
    model,context,objective=prepare(c,torch.device("cpu"))
    ctx=context(indices=[0,1,2])
    assert len(model.phases)==4 and ctx["digit_fractions"].shape==(3,10)
    assert (ctx["digit_detector_masks"].sum(0)<=1).all()
    assert ctx["digit_detector_centers"][0,1]>0
    loss=objective.checked_loss(ctx);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.phases)
    c.distances_wl[-1]+=2
    _,other,_=prepare(c,torch.device("cpu"))
    assert not torch.allclose(ctx["field"],other(indices=[0,1,2])["field"])


def test_digit_minibatches_train_and_report_all_splits():
    c=preset("Advanced • four-layer digit router");c.digit_samples_per_class=3;c.epochs=8
    result=train(c)
    assert result.metrics["final_loss"]<result.metrics["initial_loss"]
    assert result.classification["confusion"].sum()==359
    assert result.context["field"].shape[0]==10
    assert result.classification["train_scores"].shape==(30,10)
    assert result.classification["heldout_scores"].shape==(359,10)


def test_invalid_digit_circles_are_rejected():
    c=preset("Advanced • four-layer digit router");c.digit_detector_radius_wl=2
    with pytest.raises(ValueError,match="nonoverlapping"):c.validate()
