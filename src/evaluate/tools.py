import yaml
import torch


def format_row(lst):
    return ",".join(f"{e:02d}" for e in lst)


def format_conf_matrix(matrix):
    header = {f"label --": format_row(range(40))}
    rows = {
        f"label {i:02d}": str(format_row(row.to("cpu").detach().tolist()))
        for i, row in enumerate(matrix)
    }
    header.update(rows)
    return header


def format_cat_acc(matrix):
    cat_acc = torch.diag(matrix) / matrix.sum(dim=-1)
    rows = {f"label {i:02d}": v.item() for i, v in enumerate(cat_acc)}
    return rows


def format_metrics(metrics, formatter="{:.6}"):
    return metrics


def save_metrics(path, metrics):
    with open(path, "w") as yfile:
        yaml.dump(metrics, yfile)


def load_metrics(path):
    with open(path, "r") as yfile:
        string = yfile.read()
        return yaml.load(string, yaml.loader.BaseLoader)
