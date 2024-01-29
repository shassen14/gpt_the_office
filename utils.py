import os
import pickle
import tiktoken
import numpy as np
import torch


def get_batch(cfg, dataset_type: str) -> tuple[torch.Tensor, torch.Tensor]:
    # Obtain file path
    file_path = get_file_path(cfg.dataset_dir, dataset_type)

    # save data
    data = np.memmap(file_path, dtype=np.uint16, mode="r")

    # random block chunk generator TODO: figure out what is happening here
    ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + cfg.block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + cfg.block_size]).astype(np.int64))
            for i in ix
        ]
    )

    if cfg.device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(cfg.device_type, non_blocking=True), y.pin_memory().to(
            cfg.device_type, non_blocking=True
        )
    else:
        x, y = x.to(cfg.device_type), y.to(cfg.device_type)
    return x, y


###############################################################################


def get_folder_path(folder: str) -> str:
    folder_path = os.path.join(os.path.dirname(__file__), folder)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


###############################################################################


def get_file_path(folder: str, file: str) -> str:
    file_path = get_folder_path(folder) + "/" + file
    return file_path


###############################################################################


def abstract_pickle(pickle_path: str):
    # obtain vocabulary size from pkl
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            meta = pickle.load(f)
        meta_vocab_size = meta["vocab_size"]
        if meta["is_tiktoken"]:
            enc = tiktoken.get_encoding(meta["tiktoken_model"])
            meta_encode = lambda s: enc.encode(
                s, allowed_special={"<|endoftext|>"}
            )  # encoder: take a string, output a list of integers
            meta_decode = lambda l: enc.decode(
                l
            )  # decoder: take a list of integers, output a string
        else:
            meta_encode = lambda s: [
                meta["stoi"][c] for c in s
            ]  # encoder: take a string, output a list of integers
            meta_decode = lambda l: "".join(
                [meta["itos"][i] for i in l]
            )  # decoder: take a list of integers, output a string
        # print(f"found vocab_size = {meta_vocab_size} (inside {pickle_path})")
        return meta_vocab_size, meta_encode, meta_decode
    else:
        print(pickle_path + " doesn't exist. Please give a valid pkl file.")
        exit()


###############################################################################


@torch.no_grad()
def estimate_loss(model, cfg) -> dict:
    out = {}
    model.eval()
    for split in cfg.file_array:
        losses = torch.zeros(cfg.eval_iterations)
        for k in range(cfg.eval_iterations):
            X, Y = get_batch(cfg, split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


###############################################################################


def calculate_learning_rate(cfg, iteration: int) -> float:
    if iteration < cfg.warmup_iterations:
        return cfg.max_learning_rate
    elif iteration > cfg.decay_iterations:
        return cfg.min_learning_rate

    dif_lr = cfg.max_learning_rate - cfg.min_learning_rate
    dif_iter = cfg.decay_iterations - cfg.warmup_iterations

    coeff = dif_lr * (iteration - cfg.warmup_iterations) / dif_iter
    return cfg.max_learning_rate - coeff
