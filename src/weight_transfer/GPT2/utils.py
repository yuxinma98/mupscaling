import torch
import torch.nn as nn
from gpt2_model import GPT, GPTConfig
import copy
from mup.coord_check import get_coord_data, plot_coord_data
from mup import set_base_shapes
from torch.nn.parallel import DistributedDataParallel as DDP
import os


def naming_conversion(lr_exponent: float):
    if lr_exponent == int(lr_exponent):
        return str(int(lr_exponent))
    else:
        return f"{lr_exponent:.1f}".replace('.', 'p')


def get_optimizer_group(model: nn.Module, weight_decay: float):
    param_dict = {n: p for n, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    return optim_groups


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, 
                                   device: torch.device) -> None:
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device, non_blocking=True)


def unwrap_model(m: nn.Module) -> nn.Module:
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod 
    if isinstance(m, DDP):
        m = m.module
    return m


def check_config_equivalence(source_model_args, target_model_args):
    assert source_model_args['n_layer'] == target_model_args['n_layer'], "n_layer mismatch"
    assert source_model_args['n_head'] == target_model_args['n_head'], "n_head mismatch"
    assert source_model_args['block_size'] == target_model_args['block_size'], "block_size mismatch"
    assert source_model_args['vocab_size'] == target_model_args['vocab_size'], "vocab_size mismatch"
    assert source_model_args['bias'] == target_model_args['bias'], "bias mismatch"
    

def run_coord_check(model_args, 
                    mup_shape_pth,
                    widths, 
                    dataloader):
    
    class GPTForCoordCheck(nn.Module):
        def __init__(self, gpt):
            super().__init__()
            self.gpt = gpt

        def forward(self, idx, targets):
            _, loss = self.gpt(idx, targets)
            return {'loss': loss}

    def gen(n_embd):
        def f():
            ma = copy.deepcopy(model_args)
            ma['n_embd'] = n_embd
            gpt = GPT(GPTConfig(**ma))
            set_base_shapes(gpt, mup_shape_pth)
            return GPTForCoordCheck(gpt).to('cuda')
        return f

    models = {w: gen(w) for w in widths}
    legend = {w: f"n_embd={w}" for w in widths}

    df = get_coord_data(
        models,
        dataloader,
        optimizer='adamw',
        lr=0.1,
        mup=mup_shape_pth is not None,
        dict_in_out=True,
        nsteps=3,)
    
    prm = 'μP' if mup_shape_pth is not None else 'SP'

    df.to_csv(os.path.join('coord_checks', f'coord_check_{prm}.csv'), index=False)
    return plot_coord_data(df, 
                           legend=False, 
                           save_to=os.path.join('coord_checks', f'coord_check_{prm}.png'),
                           suptitle=f'{prm} Transformer adamw lr=0.1 nseeds=1337',
                           face_color='xkcd:light grey' if mup_shape_pth is None else None)
