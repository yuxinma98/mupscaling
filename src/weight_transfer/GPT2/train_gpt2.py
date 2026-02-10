import sys
sys.path.insert(1, "/home/nchen38/weight_transfer/src")

import argparse
import copy
import logging
import os

import gc
import math
import numpy as np
from tqdm import tqdm
import wandb

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from gpt2_model import GPT, GPTConfig
from mup import make_base_shapes, set_base_shapes, MuAdamW
from utils import get_optimizer_group, move_optimizer_state_to_device, naming_conversion, unwrap_model, check_config_equivalence
from weight_transfer.transfer import transfer_weights, transfer_optimizer


def main(args):
    # set up for data parallel training
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK']) 
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        device = 'cuda'
    
    # set up logging
    if master_process:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",)
        logger = logging.getLogger(__name__)
        logger.info(f"args: {args}")

        tokens_per_iter = args.gradient_accumulation_steps * ddp_world_size * args.batch_size * args.block_size
        logger.info(f"tokens per iteration will be: {tokens_per_iter:,}")
        
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs('mup_shapes', exist_ok=True)

        wandb.init(project='fineweb-edu-main',
                   name=args.run_name,
                   config=vars(args),
                   mode='online' if args.enable_log else 'disabled',)

    # set random seed
    torch.manual_seed(args.seed + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # poor man's data loader
    data_dir = os.path.join('data', args.dataset)
    def get_batch(split):
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == 'train':
            data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - args.block_size, (args.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+args.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+args.block_size]).astype(np.int64)) for i in ix])

        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)

        return x, y
    
    # model init
    model_args = dict(n_layer=args.n_layer, 
                      n_head=args.n_head, 
                      n_embd=args.n_embd, 
                      block_size=args.block_size,
                      bias=args.bias, 
                      vocab_size=args.vocab_size, 
                      dropout=args.dropout,
                      init_std=args.init_std,)
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    # muP parametrization
    if args.enable_mup:
        mup_shape_pth = os.path.join('mup_shapes', 
                                    f"gpt2_L{args.n_layer}_H{args.n_head}_V{args.vocab_size}.bsh")
        if not os.path.exists(mup_shape_pth) and master_process:
            base_args = copy.deepcopy(model_args)
            base_args['n_embd'] = args.n_head * 2
            base_model = GPT(GPTConfig(**base_args)).to('cpu')
            
            delta_args = copy.deepcopy(model_args)
            delta_args['n_embd'] = args.n_head * 4
            delta_model = GPT(GPTConfig(**delta_args)).to('cpu')

            make_base_shapes(base_model, delta_model, mup_shape_pth)
            del base_model, delta_model
        if ddp:
            torch.distributed.barrier()
        set_base_shapes(model, mup_shape_pth)
    
    model = model.to(device)

    # configure muP optimizer
    if args.enable_mup:
        optim_groups = get_optimizer_group(model, weight_decay=args.weight_decay)
        optimizer = MuAdamW(optim_groups,
                            lr=args.learning_rate,
                            betas=(0.9, 0.95),
                            fused=True,)
    else:
        optimizer = model.configure_optimizers(weight_decay=args.weight_decay, 
                                               learning_rate=args.learning_rate, 
                                               betas=(0.9, 0.95), 
                                               device_type='cuda',)

    # upscale model weights from a smaller model
    if args.up_scale_from is not None:
        ckpt_pth = os.path.join('out', args.up_scale_from, 'ckpt.pt')
        checkpoint = torch.load(ckpt_pth, map_location='cpu')

        source_model_args = checkpoint['model_args']
        check_config_equivalence(source_model_args, model_args)

        source_config = GPTConfig(**source_model_args)
        source_model = GPT(source_config)
        source_model.load_state_dict(checkpoint['model'], strict=True)
        set_base_shapes(source_model, mup_shape_pth, rescale_params=False)
        source_model = source_model.to(device)

        source_optim_groups = get_optimizer_group(source_model, 
                                                  weight_decay=checkpoint['config']['weight_decay'])
        source_optimizer = MuAdamW(source_optim_groups,
                                   lr=checkpoint['config']['learning_rate'],
                                   betas=(0.9, 0.95),)
        source_optimizer.load_state_dict(checkpoint['optimizer'])
        move_optimizer_state_to_device(source_optimizer, device)

        transfer_weights(source_model, model)
        transfer_optimizer(source_optimizer, optimizer, source_model, model)

        del source_model, source_optimizer, source_optim_groups
        torch.cuda.empty_cache()
        gc.collect()

    # lr scheduler compatible with muP
    if args.lr_scheduler_type == 'cosine_with_min_lr':
        warmup_steps = int(args.max_iters * args.warmup_ratio)
        warmup_steps = max(0, warmup_steps)

        warmup_start_factor = 1e-4
        min_lr_mult = 0.1
        total_steps = args.max_iters
        
        def lr_mult(step: int) -> float:
            if warmup_steps > 0 and step <= warmup_steps:
                return warmup_start_factor + (1.0 - warmup_start_factor) * (step / warmup_steps)
            
            denom = max(1, total_steps - warmup_steps)
            progress = (step - warmup_steps) / denom
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

            return min_lr_mult + (1.0 - min_lr_mult) * cosine

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_mult)
    elif args.lr_scheduler_type == 'constant_lr':
        lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    else:
        raise ValueError(f"Unknown lr_scheduler_type {args.lr_scheduler_type}")

    # wrap model for data parallel training
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # compile the model
    if args.enable_compile:
        model = torch.compile(model)

    raw_model = unwrap_model(model)

    # evaluation loop
    @torch.inference_mode()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['val']:
            losses = torch.zeros(args.eval_iters, device=device)
            for k in tqdm(range(args.eval_iters), disable=not args.eval_only):
                X, Y = get_batch(split)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(X, Y)
                losses[k] = loss.detach()
            out[split] = losses.mean().item()
        model.train()
        return out

    # training preparations
    iter_num = 0
    best_val_loss = 1e9
    train_loss_accum = torch.zeros((), device=device)

    X, Y = get_batch('train')

    progress_bar = tqdm(range(0, args.max_iters),
                        initial=iter_num,
                        desc='Training Steps',
                        disable=not master_process,)

    # training loop
    optimizer.zero_grad(set_to_none=True)
    while True:
        # evaluation
        if iter_num % args.eval_interval == 0 and master_process:
            losses = estimate_loss()

            logger.info(f"step {iter_num}: val loss {losses['val']:.4f}")
            wandb.log({'step': iter_num, 
                       'val/val_loss': losses['val'],})

            if args.save_checkpoint and iter_num > 0:
                best_val_loss = losses['val']
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': vars(args),
                }
                torch.save(checkpoint, os.path.join(args.save_dir, 'ckpt.pt'))

        if args.eval_only:
            break

        # forward backward pass
        for micro_step in range(args.gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == args.gradient_accumulation_steps - 1)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(X, Y)
                loss /= args.gradient_accumulation_steps

            X, Y = get_batch('train')
            loss.backward()
            train_loss_accum += loss.detach()

        # optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        # logging
        if (iter_num+1) % args.log_interval == 0 and master_process:
            train_loss_accum /= args.log_interval
            log_train_loss = train_loss_accum.item()
            train_loss_accum.zero_()

            logger.info(f"step {iter_num}: train loss {log_train_loss:.4f}")
            wandb.log({'step': iter_num, 
                       'train/train_loss': log_train_loss,
                       'train/learning_rate': lr_scheduler.get_last_lr()[0],})

        iter_num += 1
        progress_bar.update(1)

        if iter_num >= args.max_iters:
            break

    # cleanup
    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Basic arguments
    parser.add_argument('--dataset', type=str,
                        choices=['fineweb_edu'],
                        default='fineweb_edu')
    
    # Model parameters
    parser.add_argument('--vocab-size', type=int, default=8192)
    parser.add_argument('--n-layer', type=int, default=8)
    parser.add_argument('--n-head', type=int, default=8)
    parser.add_argument('--n-embd', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--init-std', type=float, default=0.02)

    # Training parameters
    parser.add_argument('--block-size', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    parser.add_argument('--max-iters', type=int, default=10000)
    parser.add_argument('--learning-rate-exponent', type=float, required=True)
    parser.add_argument('--weight-decay', type=float, default=1e-1)
    parser.add_argument('--warmup-ratio', type=float, default=0.05)
    parser.add_argument('--lr-scheduler-type', type=str, default='constant_lr')
    parser.add_argument('--enable-mup', action='store_true')
    parser.add_argument('--enable-compile', action='store_true')

    # Evaluation parameters
    parser.add_argument('--eval-interval', type=int, default=500)
    parser.add_argument('--eval-iters', type=int, default=400)
    parser.add_argument('--log-interval', type=int, default=25)
    parser.add_argument('--enable-log', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--save-checkpoint', action='store_true')
    parser.add_argument('--seed', type=int, default=1337)

    # For upscaling
    parser.add_argument('--up-scale-from', type=str, default=None)

    args = parser.parse_args()

    args.learning_rate = 2 ** (-args.learning_rate_exponent)
    args.dataset = f"{args.dataset}_{args.vocab_size}"

    if args.up_scale_from is not None:
        args.enable_mup = True

    args.prefix = 'upscale' if args.up_scale_from is not None else ('mup' if args.enable_mup else 'sp')

    args.run_name = f"{args.prefix}-L{args.n_layer}-H{args.n_head}-E{args.n_embd}-V{args.vocab_size}-lr2e{naming_conversion(args.learning_rate_exponent)}-InitStd{args.init_std:.0e}"
    args.save_dir = os.path.join('out', args.run_name)

    main(args)
