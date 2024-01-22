import utils
import config.base as cfg
import gpt.model as gm

import torch

if __name__ == '__main__':
    # obtain file paths
    pickle_path = utils.get_file_path(cfg.dataset_dir,
                                      cfg.pkl_file)
    pt_path = utils.get_file_path(cfg.param_dir, cfg.pt_file)
    sample_path = utils.get_file_path(cfg.sample_dir, cfg.sample_file)

    # obtain metadata from pkl
    meta_vocab_size, meta_encode, meta_decode = utils.abstract_pickle(pickle_path)

    # obtain model and optimizer
    model = gm.GPT(meta_vocab_size,
                   cfg.num_layers,
                   cfg.block_size,
                   cfg.num_embeddings,
                   cfg.head_size,
                   cfg.num_heads,
                   cfg.dropout,
                   cfg.device_type)
    model.to(cfg.device_type)
    torch_model = torch.load(pt_path)
    model.load_state_dict(torch_model['model'])

    # TODO: read from a context file if given one and use that as the start
    start = '\n'
    start_ids = meta_encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=cfg.device_type)[None, ...])

    with torch.no_grad():
        print('Generating text and writing to ' + sample_path)
        output_ids = model.generate(x, max_new_tokens=cfg.max_new_tokens)[0].tolist()
        open(sample_path, 'w').write(meta_decode(output_ids))