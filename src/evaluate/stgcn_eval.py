import torch
from tqdm import tqdm

from src.utils.fixseed import fixseed

from src.evaluate.stgcn.evaluate import Evaluation as STGCNEvaluation

# from src.evaluate.othermetrics.evaluation import Evaluation

from torch.utils.data import DataLoader
from src.utils.tensors import collate

import os

from .tools import save_metrics, format_metrics
from src.models.get_model import get_model as get_gen_model
from src.datasets.get_dataset import get_datasets
import src.utils.rotation_conversions as geometry


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
        assert mode in ["gen", "rc", "gt"]

        pose_rep = parameters["pose_rep"]
        translation = parameters["translation"]

        self.batches = []

        with torch.no_grad():
            for databatch in tqdm(dataiterator, desc=f"Construct dataloader: {mode}.."):
                if mode == "gen":
                    classes = databatch["y"]
                    gendurations = databatch["lengths"]
                    batch = model.generate(classes, gendurations)
                    feats = "output"
                elif mode == "gt":
                    batch = {key: val.to(device) for key, val in databatch.items()}
                    feats = "x"
                elif mode == "rc":
                    databatch = {key: val.to(device) for key, val in databatch.items()}
                    batch = model(databatch)
                    feats = "output"

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


FRAMES = 60


def _gen_data(kind: str, num_batches: int, parameters, model, folder):
    data = []
    pose_rep = parameters["pose_rep"]
    translation = parameters["translation"]
    batch_size = parameters["batch_size"]
    device = parameters["device"]
    with torch.no_grad():
        for c in tqdm(
            range(parameters["num_classes"]), desc=f"[{kind}]Generating class"
        ):
            batches = []
            for _ in tqdm(
                range(num_batches), desc=f"[{kind}]Generating batch", leave=False
            ):
                classes = torch.full((batch_size,), c).to(device)
                gendurations = torch.full((batch_size,), FRAMES).to(device)
                batch = model.generate(classes, gendurations)

                if translation:
                    x = batch["output"][:, :-1]
                else:
                    x = batch["output"]

                x = x.permute(0, 3, 1, 2)
                batches.append(x)
            batches = torch.cat(batches)
            data.append(batches)
        data = torch.stack(data)
        dest = os.path.join(folder, f"{kind}_gen_dataset.npy")
        print(f"Saving {data.shape} to {dest}")
        with open(dest, "wb") as f:
            torch.save(data, f)
        return data


def generate(parameters, folder, checkpointname, epoch, niter):
    torch.multiprocessing.set_sharing_strategy("file_system")

    pose_rep = parameters["pose_rep"]
    translation = parameters["translation"]
    batch_size = parameters["batch_size"]

    device = parameters["device"]

    parameters["num_classes"] = 40
    parameters["nfeats"] = 6
    parameters["njoints"] = 25

    model = get_gen_model(parameters)
    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    state_dict = torch.load(checkpointpath, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    model.outputxyz = False

    fixseed(1)
    _gen_data("train", niter, parameters, model, folder=folder)
    fixseed(2)
    _gen_data("test", max(1, niter // 10), parameters, model, folder=folder)


def reconstruct(parameters, folder, checkpointname, epoch, niter):
    torch.multiprocessing.set_sharing_strategy("file_system")

    device = parameters["device"]
    translation = parameters["translation"]
    bs = parameters["batch_size"]

    parameters["num_classes"] = 40
    parameters["nfeats"] = 6
    parameters["njoints"] = 25

    model = get_gen_model(parameters)
    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    state_dict = torch.load(checkpointpath, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    model.outputxyz = False

    datasets = {key: get_datasets(parameters)[key] for key in ["train", "test"]}
    for data in datasets.values():
        data.reset_shuffle()
        data.shuffle()

    for kind in ["train", "test"]:
        print("Processing DATA")
        orig_motions = list(datasets[kind])
        print("PROCESSED DATA")
        outs = []
        outs_lab = []
        print("TR", translation)
        fixseed(12)
        for i in tqdm(range(20)):
            motions, labels = process_batch(
                model, orig_motions, bs, orig_motions, device, translation, i * 30, 60
            )
            outs.append(motions)
            outs_lab.append(labels)
        dest = os.path.join(folder, f"{kind}_recon_motions.npy")
        print(f"Saving to {dest}")
        with open(dest, "wb") as f:
            torch.save(outs, f)
        dest = os.path.join(folder, f"{kind}_recon_labels.npy")
        print(f"Saving to {dest}")
        with open(dest, "wb") as f:
            torch.save(outs_lab, f)


def process_batch(
    model, motions, bs, dataset, device, translation, start: int, frames: int
):
    motions = filter(lambda x: (x[0].shape[-1] >= frames + start), dataset)
    batches = batchify(motions, bs)
    smooth_b = []
    smooth_b_c = []
    with torch.no_grad():
        for batch in batches:
            batch = [(b[0][:, :, start : (start + frames)], b[1]) for b in batch]
            batch = collate(batch)
            classes = batch["y"]
            batch = {key: val.to(device) for key, val in batch.items()}
            batch = model(batch)

            if translation:
                x = batch["output"][:, :-1]
            else:
                x = batch["output"]

            x = x.permute(0, 3, 1, 2).to("cpu")
            smooth_b.append(x)
            smooth_b_c.append(classes)
    return smooth_b, smooth_b_c


def batchify(iterable, n):
    iterable = iter(iterable)
    while True:
        chunk = []
        for _ in range(n):
            try:
                chunk.append(next(iterable))
            except StopIteration:
                yield chunk
                return
        yield chunk


def evaluate(parameters, folder, checkpointname, epoch, niter):
    torch.multiprocessing.set_sharing_strategy("file_system")

    bs = parameters["batch_size"]
    doing_recons = False

    device = parameters["device"]
    dataname = parameters["dataset"]

    # dummy => update parameters info
    # get_datasets(parameters)
    # faster: hardcode value for uestc

    parameters["num_classes"] = 40
    parameters["nfeats"] = 6
    parameters["njoints"] = 25

    model = get_gen_model(parameters)
    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    state_dict = torch.load(checkpointpath, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    model.outputxyz = False

    recogparameters = parameters.copy()
    recogparameters["pose_rep"] = "rot6d"
    recogparameters["nfeats"] = 6

    # Action2motionEvaluation
    stgcnevaluation = STGCNEvaluation(dataname, recogparameters, device)

    stgcn_metrics = {}
    # joints_metrics = {}
    # pose_metrics = {}

    compute_gt_gt = False
    if compute_gt_gt:
        datasetGT = {
            key: [get_datasets(parameters)[key], get_datasets(parameters)[key]]
            for key in ["train", "test"]
        }
    else:
        datasetGT = {key: [get_datasets(parameters)[key]] for key in ["train", "test"]}

    print("Dataset loaded")

    allseeds = list(range(niter))

    for seed in allseeds:
        fixseed(seed)
        for key in ["train", "test"]:
            for data in datasetGT[key]:
                data.reset_shuffle()
                data.shuffle()

        dataiterator = {
            key: [
                DataLoader(
                    data,
                    batch_size=bs,
                    shuffle=False,
                    num_workers=8,
                    collate_fn=collate,
                )
                for data in datasetGT[key]
            ]
            for key in ["train", "test"]
        }

        if doing_recons:
            reconsLoaders = {
                key: NewDataloader(
                    "rc", model, parameters, dataiterator[key][0], device
                )
                for key in ["train", "test"]
            }

        gtLoaders = {
            key: NewDataloader("gt", model, parameters, dataiterator[key][0], device)
            for key in ["train", "test"]
        }

        if compute_gt_gt:
            gtLoaders2 = {
                key: NewDataloader(
                    "gt", model, parameters, dataiterator[key][1], device
                )
                for key in ["train", "test"]
            }

        genLoaders = {
            key: NewDataloader("gen", model, parameters, dataiterator[key][0], device)
            for key in ["train", "test"]
        }

        loaders = {"gen": genLoaders, "gt": gtLoaders}
        if doing_recons:
            loaders["recons"] = reconsLoaders

        if compute_gt_gt:
            loaders["gt2"] = gtLoaders2

        stgcn_metrics[seed] = stgcnevaluation.evaluate(model, loaders)
        del loaders

        # joints_metrics = evaluation.evaluate(model, loaders, xyz=True)
        # pose_metrics = evaluation.evaluate(model, loaders, xyz=False)

    metrics = {
        "feats": {
            key: [format_metrics(stgcn_metrics[seed])[key] for seed in allseeds]
            for key in stgcn_metrics[allseeds[0]]
        }
    }
    # "xyz": {key: [format_metrics(joints_metrics[seed])[key] for seed in allseeds] for key in joints_metrics[allseeds[0]]},
    # model.pose_rep: {key: [format_metrics(pose_metrics[seed])[key] for seed in allseeds] for key in pose_metrics[allseeds[0]]}}

    epoch = checkpointname.split("_")[1].split(".")[0]
    metricname = "evaluation_metrics_{}_all.yaml".format(epoch)

    evalpath = os.path.join(folder, metricname)
    print(f"Saving evaluation: {evalpath}")
    save_metrics(evalpath, metrics)
