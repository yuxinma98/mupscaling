"""
The implementation of muAdam and muAdam in muP package has wrong weight decay scaling.
This file correct the weight_decay scaling, and add eps scaling.
"""

from collections import defaultdict

from torch.optim import Adam, AdamW


def process_param_groups(params, **kwargs):
    param_groups = list(params)
    if not isinstance(param_groups[0], dict):
        param_groups = [{"params": param_groups}]
    for param_group in param_groups:
        if "lr" not in param_group:
            param_group["lr"] = kwargs["lr"]
        if "weight_decay" not in param_group:
            param_group["weight_decay"] = kwargs.get("weight_decay", 0.0)
        if "eps" not in param_group:
            param_group["eps"] = kwargs.get("eps", 1e-8)
    return param_groups


def MuAdam(params, decoupled_weight_decay=False, **kwargs):
    """Adam with μP scaling.

    Note for this to work properly, your model needs to have its base shapes set
    already using `mup.set_base_shapes`.

    Inputs:
        impl: the specific Adam-like optimizer implementation from torch.optim or
            elsewhere
        decoupled_weight_decay: whether to use decoupled weight decay (AdamW style)
    Outputs:
        An instance of `impl` with refined parameter groups, each of which has the correctly
        scaled learning rate according to mup.
    """
    new_param_groups = []
    for param_group in process_param_groups(params, **kwargs):
        # For every existing param group, we split into several new groups
        def new_group():
            new_g = {k: v for k, v in param_group.items() if k != "params"}
            new_g["params"] = []
            return new_g

        # The matrix-like weights might need multiple groups since weights
        # might have different width multipliers
        matrix_like_p = defaultdict(new_group)  # key is width_mult and shape_ratio
        vector_like_p = defaultdict(new_group)
        fixed_p = new_group()
        for p in param_group["params"]:
            assert hasattr(p, "infshape"), (
                f"A parameter with shape {p.shape} does not have `infshape` attribute. " "Did you forget to call `mup.set_base_shapes` on the model?"
            )
            if p.infshape.ninf() == 1:
                vector_like_p[p.infshape.width_mult()]["params"].append(p)
            elif p.infshape.ninf() == 2:
                matrix_like_p[(p.infshape.width_mult(), p.infshape.fanin_fanout_mult_ratio())]["params"].append(p)
            elif p.infshape.ninf() > 2:
                raise NotImplementedError("more than 2 inf dimensions")
            else:
                fixed_p["params"].append(p)

        for width_mult, group in vector_like_p.items():
            group["eps"] /= width_mult
            if not decoupled_weight_decay:
                group["weight_decay"] /= width_mult

        for (width_mult, shape_ratio), group in matrix_like_p.items():
            group["lr"] /= width_mult
            group["eps"] /= width_mult / shape_ratio
            if decoupled_weight_decay:
                group["weight_decay"] *= width_mult
            else:
                group["weight_decay"] *= shape_ratio

        new_param_groups.extend(list(matrix_like_p.values()) + list(vector_like_p.values()) + [fixed_p])
    return Adam(new_param_groups, decoupled_weight_decay=decoupled_weight_decay, **kwargs)


def MuAdamW(params, **kwargs):
    """AdamW with μP scaling.

    Note for this to work properly, your model needs to have its base shapes set
    already using `mup.set_base_shapes`.
    """
    # in torch, Adam(decoupled_weight_decay=True) behaves exactly the same as AdamW
    return MuAdam(params, decoupled_weight_decay=True, **kwargs)
