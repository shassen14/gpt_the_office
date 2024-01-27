import utils
import config.base
from models import self_attention as sa

import torch

if __name__ == '__main__':
    # obtain all config parameters as an object. Only using the cfg for file paths
    cfg = config.base.Config()

    # obtain file paths
    pickle_path = utils.get_file_path(cfg.dataset_dir,
                                      cfg.pkl_file)
    pt_path = utils.get_file_path(cfg.param_dir, cfg.pt_file)
    sample_path = utils.get_file_path(cfg.sample_dir, cfg.sample_file)

    # obtain metadata from pkl
    meta_vocab_size, meta_encode, meta_decode = utils.abstract_pickle(pickle_path)

    # obtain model and optimizer
    model = sa.Model(meta_vocab_size, cfg)
    model.to(cfg.device_type)
    
    torch_model = torch.load(pt_path)
    model.load_state_dict(torch_model['model'])
    model.eval()

    # update cfg for model parameters
    # TODO: there probably is some bug by doing it this way. Need to figure out
    # a better way to load config that reduces a bug happening from
    # differentiating config parameters
    cfg = torch_model['config'] 

    # TODO: read from a context file if given one and use that as the start
    start = '\n'
    start_ids = meta_encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=cfg.device_type)[None, ...])

    with torch.no_grad():
        print('Generating text and writing to ' + sample_path)
        print(meta_decode(start_ids), end ="")
        open(sample_path, 'w').write(meta_decode(start_ids))
        for _ in range(cfg.max_new_tokens):
            x, x_next = model.generate2(x)
            token = meta_decode(x_next[0].tolist())
            print(token, end = "")
            open(sample_path, 'a').write(token)
    print()
