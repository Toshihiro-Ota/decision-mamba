import csv
import os
from collections import OrderedDict

import random
import numpy as np
import torch


try:
    import wandb
except ImportError:
    pass

try:
    import mlflow
except ImportError:
    pass


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def get_outdir(path, *paths, inc=False):
    outdir = os.path.join(path, *paths)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    elif inc:
        count = 1
        outdir_inc = outdir + '-' + str(count)
        while os.path.exists(outdir_inc):
            count = count + 1
            outdir_inc = outdir + '-' + str(count)
            assert count < 100
        outdir = outdir_inc
        os.makedirs(outdir)
    return outdir

def update_summary(
        epoch,
        logs,
        filename,
        args_dir=None,
        lr=None,
        write_header=False,
        log_wandb=False,
        log_mlflow=False,
        ):
    rowd = OrderedDict(epoch=epoch)
    rowd.update([(k, v) for k, v in logs.items()])
    if lr is not None:
        rowd['lr'] = lr
    if log_wandb:
        wandb.log(rowd)
    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:  # first iteration (epoch == 1 can't be used)
            dw.writeheader()
        dw.writerow(rowd)

    if log_mlflow:
        if write_header:
            mlflow.log_artifact(args_dir)
        mlflow.log_metrics({k: v for k, v in logs.items()}, step=epoch)
        mlflow.log_artifact(filename)
