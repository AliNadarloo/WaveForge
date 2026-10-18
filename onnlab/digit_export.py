# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
"""Compact, inspectable classification reports; avoid thousands of redundant figures."""
import csv
import json
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from .appearance import THEMES,style_figure


def export_classification(result,out,prefs):
    report=result.classification;c=result.config
    np.savez_compressed(out/"digit_predictions.npz",**report)
    with (out/"digit_predictions.csv").open("w",newline="",encoding="utf-8") as file:
        writer=csv.writer(file)
        writer.writerow(["sample_id","split","expected","predicted","correct"]+[f"detector_{i}_fraction" for i in range(10)])
        for split,ids,scores in [("training",report["train_indices"],report["train_scores"]),
                                 ("heldout",report["heldout_indices"],report["heldout_scores"])]:
            for sample,score in zip(ids,scores):
                expected=int(report["labels"][sample]);predicted=int(score.argmax())
                writer.writerow([int(sample),split,expected,predicted,int(expected==predicted),*score])
    np.savetxt(out/"digit_confusion.csv",report["confusion"],fmt="%d",delimiter=",")
    np.savetxt(out/"digit_learning_curve.csv",report["learning_curve"],delimiter=",",
               header="epoch,mean_pre_update_batch_loss,training_batch_accuracy,heldout_accuracy",comments="")
    centers=result.context["digit_detector_centers"].cpu().numpy()
    table=np.column_stack((np.arange(10),centers,np.full(10,c.digit_detector_radius_wl*c.wavelength)))
    np.savetxt(out/"digit_detector_layout.csv",table,delimiter=",",header="digit,x_m,y_m,radius_m",comments="")
    fig=Figure(figsize=(6.4,5.5),layout="constrained");FigureCanvasAgg(fig);ax=fig.add_subplot()
    counts=report["confusion"];im=ax.imshow(counts,cmap=prefs["intensity_cmap"],origin="upper",interpolation="nearest")
    fig.colorbar(im,ax=ax,label="Held-out samples")
    ax.set(xticks=range(10),yticks=range(10),xlabel="Predicted circle / digit",ylabel="Actual digit",title="Held-out confusion matrix")
    for y in range(10):
        for x in range(10):
            red,green,blue,_=im.cmap(im.norm(counts[y,x]))
            lightness=.2126*red+.7152*green+.0722*blue
            ax.text(x,y,str(counts[y,x]),ha="center",va="center",fontsize=8,color="black" if lightness>.5 else "white")
    style_figure(fig,prefs)
    fig.savefig(out/"digit_confusion.png",dpi=160);fig.clear()
    fig=Figure(figsize=(6.4,4.8),layout="constrained");FigureCanvasAgg(fig);ax=fig.add_subplot()
    curve=report["learning_curve"]
    colors=THEMES[prefs["theme"]]
    ax.plot(curve[:,0],curve[:,2],label="Training mini-batches",color=colors["accent"])
    ax.plot(curve[:,0],curve[:,3],label="Held-out (no updates)",color=colors["muted"],linestyle="--")
    ax.set(xlabel="Epoch",ylabel="Accuracy",ylim=(0,1.02),title="Digit recognition accuracy");ax.grid(alpha=.2);ax.legend()
    style_figure(fig,prefs)
    fig.savefig(out/"digit_accuracy.png",dpi=160);fig.clear()
    (out/"DIGIT_README.txt").write_text(
        "Four-layer preset: handwritten 0–9 -> ten circular detectors.\n"
        "The strongest integrated circle intensity determines the predicted digit.\n"
        "Training and held-out indices are disjoint and saved in digit_predictions.npz.\n"
        "Held-out samples never enter gradient updates. This is a split of the bundled\n"
        "UCI / scikit-learn digit images, not the original UCI benchmark split or MNIST.\n"
        "fields.npz and individual field figures contain one held-out example per digit.\n"
        "All evaluated training/held-out predictions are in digit_predictions.csv.\n"
        "Circle fractions divide by sampled output power, not input power.\n"
        "The last configured plane spacing is last surface -> output; changing it changes training.\n"
        "See examples/digits/DATASET.md for dataset attribution.\n",encoding="utf-8")
