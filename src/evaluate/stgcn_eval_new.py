from einops.einops import repeat
import torch
import os
from tqdm import tqdm
import numpy as np
from contextlib import redirect_stdout

from src.utils.fixseed import fixseed

from src.evaluate.stgcn.evaluate import Evaluation as STGCNEvaluation

# from src.evaluate.othermetrics.evaluation import Evaluation

from torch.utils.data import DataLoader
from src.utils.tensors import collate

import sys

from .tools import save_metrics, format_metrics
from src.models.get_model import get_model as get_gen_model
from src.datasets.get_dataset import get_datasets
import src.utils.rotation_conversions as geometry
from einops import rearrange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

do_save_metrics = True


def convert_x_to_rot6d(x, pose_rep):
    # convert rotation to rot6d
    if pose_rep == "rotvec":
        x = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(x))
    elif pose_rep == "rotmat":
        x = x.reshape(*x.shape[:-1], 3, 3)
        x = geometry.matrix_to_rotation_6d(x)
    elif pose_rep == "rotquat":
        x = geometry.matrix_to_rotation_6d(geometry.quaternion_to_matrix(x))
    elif pose_rep == "rot6d":
        x = x
    else:
        raise NotImplementedError("No geometry for this one.")
    return x


class NewDataloader:
    def __init__(self, mode, model, parameters, dataiterator, device):
        assert mode in ["gen", "gt"]

        pose_rep = parameters["pose_rep"]
        translation = parameters["translation"]
        my_model = parameters["model"]
        top_k = parameters["eval_top_k"]
        top_p = parameters["eval_top_p"]
        temperature = parameters["eval_temperature"]
        self.batches = []
        starting_pose = torch.Tensor(torch.load("starting_pose.pkl"))

        with torch.no_grad():
            for databatch in tqdm(
                dataiterator, desc=f"Construct dataloader: {mode}..", file=sys.stdout
            ):
                if mode == "gen":
                    classes = databatch["y"]
                    b = classes.shape[0]

                    base = databatch["x"]
                    base = base[:, :24, :, :1].to(device)
                    base_motion = rearrange(base, "b j s l -> b l (j s)")
                    # base_motion = repeat(starting_pose, "j d->b 1 (j d)", b=b).to( device)

                    batch = {key: val.to(device) for key, val in databatch.items()}
                    feats = "output"
                    classes = classes.to("cuda")
                    gen_motion = my_model.generate_motion(
                        classes, 60, base_motion, k=top_k, temp=temperature, top_p=top_p
                    )

                    gen_motion = gen_motion[:, 0:60]
                    batch["output"] = rearrange(
                        gen_motion, "b l (joint s)-> b joint s l", s=6
                    )
                    translation = False
                elif mode == "gt":
                    batch = {key: val.to(device) for key, val in databatch.items()}
                    feats = "x"
                else:
                    assert False

                batch = {key: val.to(device) for key, val in batch.items()}

                if translation:
                    x = batch[feats][:, :-1]
                else:
                    x = batch[feats]

                x = x.permute(0, 3, 1, 2)
                x = convert_x_to_rot6d(x, pose_rep)
                x = x.permute(0, 2, 3, 1)

                batch["x"] = x

                self.batches.append(batch)

    def __iter__(self):
        return iter(self.batches)


class EvalMetrics:
    def __init__(self):
        parameters = {}

        parameters["batch_size"] = 128
        parameters["cuda"] = True
        parameters["dataset"] = "uestc"
        parameters["debug"] = False
        parameters["glob"] = True
        parameters["glob_rot"] = [3.1415926535, 0, 0]
        parameters["jointstype"] = "vertices"
        parameters["pose_rep"] = "rot6d"
        parameters["max_len"] = -1
        parameters["min_len"] = -1
        parameters["num_seq_max"] = 3000
        parameters["translation"] = True
        parameters["num_classes"] = 40
        parameters["num_frames"] = 60
        parameters["sampling_step"] = 1
        parameters["sampling"] = "conseq"
        parameters["snapshot"] = 100
        parameters["nfeats"] = 6
        parameters["njoints"] = 25
        parameters["device"] = device
        recogparameters = parameters.copy()
        recogparameters["pose_rep"] = "rot6d"
        recogparameters["nfeats"] = 6

        self.parameters = parameters
        datasets = get_datasets(parameters)
        self.datasetGT = {"test": datasets["test"], "train": datasets["train"]}
        self.stgcnevaluation = STGCNEvaluation(
            parameters["dataset"], recogparameters, device
        )

    def evaluate(
        self, niter, model, top_k, temperature, eval_on_train: bool = False, top_p=None
    ):
        parameters = self.parameters.copy()
        datasetGT = self.datasetGT
        parameters["eval_top_k"] = top_k
        parameters["eval_top_p"] = top_p
        parameters["eval_temperature"] = temperature

        if isinstance(model, str):
            model = torch.load(model)

        model.eval()
        model = model.to(device)
        parameters["model"] = model

        stgcn_metrics = {}

        print("Dataset loaded")
        allseeds = list(range(niter))

        for seed in allseeds:
            # fixseed(seed)
            dname = "train" if eval_on_train else "test"
            datasetGT[dname].reset_shuffle()
            datasetGT[dname].shuffle()

            dataiterator = DataLoader(
                datasetGT[dname],
                batch_size=parameters["batch_size"],
                shuffle=False,
                num_workers=0,
                collate_fn=collate,
            )

            gtLoaders = {
                "test": NewDataloader("gt", None, parameters, dataiterator, device)
            }

            genLoaders = {
                "test": NewDataloader("gen", None, parameters, dataiterator, device)
            }

            loaders = {"gen": genLoaders, "gt": gtLoaders}

            stgcn_metrics[seed] = self.stgcnevaluation.evaluate(None, loaders)
            del loaders

        metrics = {"seeds": [format_metrics(stgcn_metrics[seed]) for seed in allseeds]}
        metrics = {
            key: [m[key] for m in metrics["seeds"]]
            for key in metrics["seeds"][0].keys()
        }
        return metrics


def format_stdmean(values):
    _, mean, std = values
    return f"{mean:5.2f}+-{std:5.2f}"


def stdmean(values, mult=1.0):
    values = [float(v) * mult for v in values]
    values = np.array(values)
    return values, np.mean(values), np.std(values)


def evaluate(
    parameters, folder, checkpointname, epoch, niter, evaluator, model: str = None
):
    print(f"Evaluating ... top_k={top_k}, temperature={temperature}")

    evalpath = "eval_" + model.split(".")[0] + f"{top_k}_{temperature}_{niter}.yml"
    print("EVALUATING: ", evalpath)
    # with open(os.devnull, "w") as ignore_out:
    #    with redirect_stdout(ignore_out):
    metrics = evaluator.evaluate(niter, model=model)

    acc = stdmean(metrics["accuracy_gen_test"], mult=100.0)
    fid = stdmean(metrics["fid_gen_test"])
    mmod = stdmean(metrics["multimodality_gen_test"])
    div = stdmean(metrics["diversity_gen_test"])

    print(f"Acc:        {format_stdmean(acc)}")
    print(f"FID:        {format_stdmean(fid)}")
    print(f"MultiMod:   {format_stdmean(mmod)}")
    print(f"Div:        {format_stdmean(div)}")

    print(f"Saving evaluation: {evalpath}")
    if do_save_metrics:
        save_metrics(evalpath, metrics)
    return {"acc": acc, "fid": fid, "mmod": mmod, "div": div}


import pandas as pd

if __name__ == "__main__":
    if True:  # Best
        top_k = 7
        temperature = 2.0
        evaluator = EvalMetrics(top_k, temperature)
        models = [
            #  "results/march_tok_motion_auto/conv_pass_large_s4_sampled-v8.torch",
            #  "results/march_tok_motion_auto/conv_pass_large_s4-v8.torch",
            #  "results/march_tok_motion_auto/conv_pass_medium_s4_sampled-v8.torch",
            #  "results/march_tok_motion_auto/conv_pass_medium_s4-v8.torch",
            #  "results/march_tok_motion_auto/attn_pass_large_s4-v8.torch",
            #  "results/march_tok_motion_auto/attn_pass_medium_s4-v8.torch",
            #  "results/march_tok_motion_auto/attn_pass_large_s4_sampled-v8.torch",
            #  "results/march_tok_motion_auto/attn_pass_medium_s4_sampled-v8.torch",
            # Best
            # "results/march_tok_motion_auto/attn_pass_large_s4_standingW-v8-epoch200.torch",
            # "results/april_tok_motion_auto/attn_pass_large_s4-v1.torch",
            # "results/april_tok_motion_auto/attn_pass_large_s8-v1-epoch200.torch",
            # "results/april_tok_motion_auto/attn_pass_large_s4_finetune_gold25-v2.torch",
            # "results/april_tok_motion_auto/attn_pass_large_s4_finetune_gold_drop-v2.torch",
            # AG
            # "results/april_tok_motion_auto_ag/attn_pass_large_s4_uestc-v2_drop-epoch25.torch",
            # "results/april_tok_motion_auto_ag/attn_pass_large_s8_uestc-v2_drop-epoch30.torch",
            # MAY
            # "results/may_tok_motion_auto/large_s4_auto_uestc-v1.torch",
            # "results/may_tok_motion_auto/large_s4_recauto_uestc-v1-epoch225.torch",
            # "results/may_tok_motion_auto/large_s4_recauto_rec_uestc-v1-epoch25.torch",
            # "results/may_tok_motion_auto/large_s8_auto_uestc-v1-epoch225.torch",
            # "results/may_tok_motion_auto/large_s8_recauto_uestc-v1-epoch125.torch",
            # "results/may_tok_motion_auto/large_s8_recauto_rec_uestc-v1-epoch25.torch",
            # "results/may_tok_motion_auto/large_s4_recauto_rec_uestc-v2.torch",
            # "results/may_tok_motion_auto/large_s4_auto_uestc-v2.torch",
            # "results/may_tok_motion_auto/large_s8_auto_uestc-v2.torch",
            "results/may_tok_motion_auto/large_s4_auto_uestc-v3.torch",
        ]
        for model in models:
            metrics = evaluate(
                None, None, None, None, niter=10, evaluator=evaluator, model=model
            )
        exit(0)
    print(80 * "=")
    print("SAMPLING SURVEY")
    print(80 * "=")

    do_save_metrics = False
    data = []
    values = [
        (k, temp)
        for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100, 200, 512]
        # for temp in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        # for temp in [0.1, 0.2, 0.5, 1.0]
        # for temp in [2.0, 5.0, 10.0, 20.0]
        # for temp in [0.1, 0.2]
        # for temp in [0.5, 1.0]
        # for temp in [2.0, 5.0]
        for temp in [10.0, 20.0]
    ]
    for k, temp in tqdm(values):
        top_k = k
        temperature = temp
        metrics = evaluate(
            None,
            None,
            None,
            None,
            niter=3,
            evaluator=evaluator,
            model="results/may_tok_motion_auto/large_s8_auto_uestc-v3.torch",
        )
        means = {k + "_mean": v[1] for k, v in metrics.items()}
        stds = {k + "_std": v[2] for k, v in metrics.items()}
        data.append({"k": top_k, "temp": temperature, **means, **stds})

    df = pd.DataFrame(data)
    df.to_csv("eval_results/may_s8_sampling_survey4.csv")
