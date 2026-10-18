# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
"""Measure batched vs individual propagation after warmup, with CUDA synchronization."""
from pathlib import Path
import json
import time
import torch
from onnlab.config import Config
from onnlab.physics import DiffractiveNetwork,choose_device


def main():
    device=choose_device();reports=[]
    for method in ["Dense RS","FFT RS","ASM • evanescent decay"]:
        c=Config(name="Speed measurement",propagation=method,samples_per_cell=2)
        model=DiffractiveNetwork(c,device)
        inputs=torch.randn(14,32,32,dtype=torch.complex64,device=device)
        measurements={}
        for mode in ["individual","batched"]:
            model.paper_order=mode=="individual"
            for _ in range(5):
                model.zero_grad();model(inputs)[0].abs().square().mean().backward()
            if device.type=="cuda":torch.cuda.synchronize()
            start=time.perf_counter()
            for _ in range(30):
                model.zero_grad();model(inputs)[0].abs().square().mean().backward()
            if device.type=="cuda":torch.cuda.synchronize()
            measurements[mode+"_ms_per_forward_backward"]=(time.perf_counter()-start)*1000/30
        reports.append(dict(method=method,device=str(device),batch=14,**measurements,
                            speedup=measurements["individual_ms_per_forward_backward"]/measurements["batched_ms_per_forward_backward"]))
    Path("verification/performance.json").write_text(json.dumps(reports,indent=2))
    print(json.dumps(reports,indent=2))


if __name__=="__main__":main()
