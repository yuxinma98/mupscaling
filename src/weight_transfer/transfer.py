"""
Universal function for transferring weights from a narrow model to a wider model.
Work for MLP, ResNet, and Transformer.
"""

import torch

normalization_layers = [
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.LayerNorm,
    torch.nn.GroupNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
]

def _is_norm_module(module: torch.nn.Module) -> bool:
    if any(isinstance(module, layer) for layer in normalization_layers):
        return True
    cls_name = module.__class__.__name__.lower()
    return cls_name.endswith("norm")


def transfer_weights(source_model, target_model, verbose=False):
    """
    Transfer weights from a narrow source model to a wider target model.
    Handles both parameters (weights, biases) and buffers (BatchNorm running stats).
    """
    with torch.no_grad():
        # Zero out normalization layer parameters in target model
        if hasattr(source_model, "_orig_mod"):
            source_model = source_model._orig_mod

        for name, module in target_model.named_modules():
            if _is_norm_module(module):
                for p in module.parameters():
                    p.zero_()

        for name, p in source_model.named_parameters():
            if name not in dict(source_model.named_parameters()):
                raise ValueError(f"Parameter {name} not found in target model.")
            target_p = dict(target_model.named_parameters())[name]
            if p.infshape.ninf() == 0:  # in case source model is base model
                reference_p = target_p
            else:
                reference_p = p

            if reference_p.infshape.ninf() == 2:  # hidden weight, inf * inf
                assert (
                    target_p.shape[0] % p.shape[0] == 0 and target_p.shape[1] % p.shape[1] == 0
                ), f"Cannot expand parameter of shape {p.shape} to shape {target_p.shape}"
                target_p.add_(p.repeat_interleave(target_p.shape[0] // p.shape[0], dim=0).repeat_interleave(target_p.shape[1] // p.shape[1], dim=1))
                target_p *= p.shape[1] / target_p.shape[1]  # rescale
                if verbose:
                    print(f"Transferred parameter {name} from shape {p.shape} to {target_p.shape}: inf * inf")
            elif reference_p.infshape.ninf() == 1:
                if len(p.shape) == 1 or reference_p.infshape[0].isinf() and not reference_p.infshape[1].isinf():  # inf * fin or inf(vector)
                    assert target_p.shape[0] % p.shape[0] == 0, f"Cannot expand parameter of shape {p.shape} to shape {target_p.shape}"
                    target_p.add_(p.repeat_interleave(target_p.shape[0] // p.shape[0], dim=0))  # no rescale
                    if verbose:
                        print(f"Transferred parameter {name} from shape {p.shape} to {target_p.shape}: inf * fin")
                elif not reference_p.infshape[0].isinf() and reference_p.infshape[1].isinf():  # output weight, fin * inf
                    assert target_p.shape[1] % p.shape[1] == 0, f"Cannot expand parameter of shape {p.shape} to shape {target_p.shape}"
                    target_p.add_(p.repeat_interleave(target_p.shape[1] // p.shape[1], dim=1))  # no rescale
                    if verbose:
                        print(f"Transferred parameter {name} from shape {p.shape} to {target_p.shape}: fin * inf")
            elif reference_p.infshape.ninf() == 0:  # output bias, fin(vector)
                assert target_p.shape == p.shape, f"Cannot expand parameter of shape {p.shape} to shape {target_p.shape}"
                target_p.add_(p)  # no rescale
                if verbose:
                    print(f"Transferred parameter {name} from shape {p.shape} to {target_p.shape}: fin * fin")
            else:
                NotImplementedError(f"Parameter {name} with shape {p.shape} not supported.")
        for name, p in source_model.named_buffers():  # for batchnorm running mean/var, and positional encoding in transformer
            if name not in dict(source_model.named_buffers()):
                raise ValueError(f"Parameter {name} not found in target model.")
            target_p = dict(target_model.named_buffers())[name]
            if len(p.shape) == 1:
                expand_ratio = target_p.shape[0] // p.shape[0]
                target_p.copy_(p.repeat_interleave(expand_ratio))
                if verbose:
                    print(f"Transferred buffer {name} from shape {p.shape} to {target_p.shape}: vector")
            elif len(p.shape) > 1:
                expand_ratio = target_p.shape[-1] // p.shape[-1]
                target_p.copy_(p.repeat_interleave(expand_ratio, dim=-1))
                if verbose:
                    print(f"Transferred buffer {name} from shape {p.shape} to {target_p.shape}: matrix")
            elif len(p.shape) == 0:
                target_p.copy_(p)
                if verbose:
                    print(f"Transferred buffer {name} from shape {p.shape} to {target_p.shape}: scalar value.")
            else:
                raise NotImplementedError(f"Buffer {name} with shape {p.shape} not supported.")


def transfer_optimizer(source_optimizer, target_optimizer, source_model, target_model):
    """
    Transfer optimizer states (momentum buffers, etc.) from narrow model optimizer to wide model optimizer.
    Currently support SGD, Adam, and AdamW optimizers in Torch.
    """

    for name, p in source_model.named_parameters():
        n_state = source_optimizer.state.get(p, None)
        if not n_state:  # no buffers for this parameter
            continue
        target_p = dict(target_model.named_parameters())[name]
        target_state = target_optimizer.state.setdefault(target_p, {})

        for k, v in n_state.items():
            if isinstance(v, torch.Tensor) and len(v.shape) == 0:  # e.g., 'step' count in Adam/AdamW
                target_state[k] = v

            if isinstance(v, torch.Tensor) and v.shape == p.shape:  # e.g. momentum in SGD, exp_avg/exp_avg_sq in Adam
                target_state.setdefault(k, torch.zeros_like(target_p))
                if p.infshape.ninf() == 0:  # in case source model is base model
                    reference_p = target_p
                else:
                    reference_p = p
                if k == "exp_avg_sq" or k == "max_exp_avg_sq":  # second moment requires different scaling
                    width_scale = 2.0
                else:
                    width_scale = 1.0

                if reference_p.infshape.ninf() == 2:  # hidden weight, inf * inf
                    assert (
                        target_p.shape[0] % p.shape[0] == 0 and target_p.shape[1] % p.shape[1] == 0
                    ), f"Cannot expand buffer for parameter of shape {p.shape} to shape {target_p.shape}"
                    target_state[k].copy_(
                        v.repeat_interleave(target_p.shape[0] // p.shape[0], dim=0).repeat_interleave(target_p.shape[1] // p.shape[1], dim=1)
                    )
                    target_state[k] *= (p.shape[0] / target_p.shape[0]) ** width_scale  # rescale
                elif reference_p.infshape.ninf() == 1:
                    if len(p.shape) == 1 or reference_p.infshape[0].isinf() and not reference_p.infshape[1].isinf():  # inf * fin or inf(vector)
                        assert target_p.shape[0] % p.shape[0] == 0, f"Cannot expand buffer for parameter of shape {p.shape} to shape {target_p.shape}"
                        target_state[k].copy_(v.repeat_interleave(target_p.shape[0] // p.shape[0], dim=0))
                        target_state[k] *= (p.shape[0] / target_p.shape[0]) ** width_scale  # rescale
                    elif not reference_p.infshape[0].isinf() and reference_p.infshape[1].isinf():  # output weight, fin * inf
                        assert target_p.shape[1] % p.shape[1] == 0, f"Cannot expand buffer for parameter of shape {p.shape} to shape {target_p.shape}"
                        target_state[k].copy_(v.repeat_interleave(target_p.shape[1] // p.shape[1], dim=1))
                        target_state[k] *= (p.shape[1] / target_p.shape[1]) ** width_scale  # rescale
                elif reference_p.infshape.ninf() == 0:  # output bias, fin(vector)
                    assert target_p.shape == p.shape, f"Cannot expand buffer for parameter of shape {p.shape} to shape {target_p.shape}"
                    target_state[k].copy_(v)

    # ensure tensor values are on correct device/dtype
    for p, st in list(target_optimizer.state.items()):
        for k, v in list(st.items()):
            if isinstance(v, torch.Tensor):
                st[k] = v.to(device=p.device, dtype=p.dtype)

def transfer_weights_sp(source_model, target_model, verbose=False):
    """
    Transfer weights from a narrow source model to a wider target model.
    Used for standard parametrization (only for comparison).
    """
    with torch.no_grad():
        # Zero out normalization layer parameters in target model
        if hasattr(source_model, "_orig_mod"):
            source_model = source_model._orig_mod

        for name, module in target_model.named_modules():
            if any(isinstance(module, layer) for layer in normalization_layers):
                for p in module.parameters():
                    p.zero_()

        for name, p in source_model.named_parameters():
            if name not in dict(source_model.named_parameters()):
                raise ValueError(f"Parameter {name} not found in target model.")
            target_p = dict(target_model.named_parameters())[name]
            if len(p.shape) > 1:
                assert (
                    target_p.shape[0] % p.shape[0] == 0 and target_p.shape[1] % p.shape[1] == 0
                ), f"Cannot expand parameter of shape {p.shape} to shape {target_p.shape}"
                target_p.add_(p.repeat_interleave(target_p.shape[0] // p.shape[0], dim=0).repeat_interleave(target_p.shape[1] // p.shape[1], dim=1))
                target_p *= p.shape[1] / target_p.shape[1]  # rescale
                if verbose:
                    print(f"Transferred parameter {name} from shape {p.shape} to {target_p.shape}: inf * inf")
            elif len(p.shape) == 1: 
                assert target_p.shape[0] % p.shape[0] == 0, f"Cannot expand parameter of shape {p.shape} to shape {target_p.shape}"
                target_p.add_(p.repeat_interleave(target_p.shape[0] // p.shape[0], dim=0))  # no rescale
                if verbose:
                    print(f"Transferred parameter {name} from shape {p.shape} to {target_p.shape}: inf * fin")
            elif len(p.shape) == 0: 
                assert target_p.shape == p.shape, f"Cannot expand parameter of shape {p.shape} to shape {target_p.shape}"
                target_p.add_(p)  # no rescale
                if verbose:
                    print(f"Transferred parameter {name} from shape {p.shape} to {target_p.shape}: fin * fin")
            else:
                NotImplementedError(f"Parameter {name} with shape {p.shape} not supported.")
        for name, p in source_model.named_buffers():  # for batchnorm running mean/var, and positional encoding in transformer
            if name not in dict(source_model.named_buffers()):
                raise ValueError(f"Parameter {name} not found in target model.")
            target_p = dict(target_model.named_buffers())[name]
            if len(p.shape) == 1:
                expand_ratio = target_p.shape[0] // p.shape[0]
                target_p.copy_(p.repeat_interleave(expand_ratio))
                if verbose:
                    print(f"Transferred buffer {name} from shape {p.shape} to {target_p.shape}: vector")
            elif len(p.shape) > 1:
                expand_ratio = target_p.shape[-1] // p.shape[-1]
                target_p.copy_(p.repeat_interleave(expand_ratio, dim=-1))
                if verbose:
                    print(f"Transferred buffer {name} from shape {p.shape} to {target_p.shape}: matrix")
            elif len(p.shape) == 0:
                target_p.copy_(p)
                if verbose:
                    print(f"Transferred buffer {name} from shape {p.shape} to {target_p.shape}: scalar value.")
            else:
                raise NotImplementedError(f"Buffer {name} with shape {p.shape} not supported.")