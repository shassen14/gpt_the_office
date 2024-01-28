import utils
import config.base
from models import self_attention as sa

import sys
import torch

torch.set_printoptions(precision=2, linewidth=100)
"""
Purpose is to train language models, save the model, and then generate a sample text
to the terminal during checkpoints
"""
# Globals
iteration = 0
best_val_loss = sys.float_info.max

if __name__ == "__main__":
    # obtain all config parameters as an object
    cfg = config.base.Config()

    # obtain file paths
    pickle_path = utils.get_file_path(cfg.dataset_dir, cfg.pkl_file)
    pt_path = utils.get_file_path(cfg.param_dir, cfg.pt_file)

    # obtain metadata from pkl
    meta_vocab_size, meta_encode, meta_decode = utils.abstract_pickle(pickle_path)

    # Initializing model
    if cfg.initialize == "start":
        print(f"Initializing from the {cfg.initalize}")
        # create model using config params
        # convert model to the device. important if using cuda
        model = sa.Model(meta_vocab_size, cfg)
        model.to(cfg.device_type)

        # print the number of parameters in the model
        print(sum(p.numel() for p in model.parameters()) / 1e6, "million parameters")

        # create a PyTorch optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.max_learning_rate)
    elif cfg.initialize == "resume":
        print(f"{cfg.initialize} from {pt_path} model")
        # load pytorch model
        torch_model = torch.load(pt_path)

        # update iteration step
        iteration = torch_model["iteration"]

        # TODO: how do I ensure vocab size is same if characters change in dataset
        model = sa.Model(meta_vocab_size, cfg)
        model.to(cfg.device_type)
        model.load_state_dict(torch_model["model"])

        # print the number of parameters in the model
        print(sum(p.numel() for p in model.parameters()) / 1e6, "million parameters")

        # Load pytorch optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.max_learning_rate)
        optimizer.load_state_dict(torch_model["optimizer"])
    else:
        print(
            f"Change cfg.initialize to 'start' or 'resume'. The value is {cfg.initialize}"
        )
        exit()

    # iterate max iteration amount of times
    while iteration < cfg.max_iterations:
        # print loss ever so often to check our model's progress
        # save our model and optimzer ever so often
        if iteration % cfg.eval_iterations == 0 or iteration == cfg.max_iterations - 1:
            losses = utils.estimate_loss(model, cfg)
            print(
                f"\nstep {iteration}: train loss {losses[cfg.train_file]:.4f}, validation loss {losses[cfg.val_file]:.4f}"
            )

            # save model if the current validation loss is less than the current best
            if losses[cfg.val_file] < best_val_loss:
                best_val_loss = losses[cfg.val_file]
                torch_model = {
                    "iteration": iteration,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": cfg,
                    "best_val_loss": best_val_loss,
                }
                print(f"Saving model to {pt_path}")
                torch.save(torch_model, pt_path)

            # example generation
            print("CHECKPOINT EXAMPLE TEXT:")
            context = torch.zeros((1, 1), dtype=torch.long, device=cfg.device_type)
            print(meta_decode(model.generate(context, max_new_tokens=200)[0].tolist()))

        # get batch to train on using context length and batch size
        xb, yb = utils.get_batch(cfg, cfg.train_file)

        # train model with a single step of backward propagation
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        iteration += 1
