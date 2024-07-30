import glob
import os
import time
import uuid

import click
import numpy as np
import torch
import torch.nn as nn
import torch._inductor.config as config
import torch.distributed as dist
import wandb
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
torch.set_float32_matmul_precision('high')

from model import GPT, GPTConfig

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict
import html

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict
import torch
import numpy as np

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print(
            "---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README"
        )
        print(
            "---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try"
        )
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2]  # number of tokens (claimed)
    return ntok  # for now just return the number of tokens


def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]  # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens


class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert (
            len(self.files) > 0
        ), f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print0(
            f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files"
        )

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self):  # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()



def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


def linear_decay_lr(step, num_iterations, learning_rate):
    return learning_rate * (1 - step / num_iterations)


def warmup_cooldown_lr(
    step, num_iterations, learning_rate, warmup_iters, warmdown_iters
):
    if step < warmup_iters:
        return learning_rate * (step + 1) / warmup_iters
    elif step < num_iterations - warmdown_iters:
        return learning_rate
    else:
        decay_ratio = (num_iterations - step) / warmdown_iters
        return learning_rate * decay_ratio

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict
import torch
import numpy as np

def get_activation_stats(model):
    activation_stats = defaultdict(list)
    
    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            abs_output = output.abs().detach()
            stats = torch.stack([
                abs_output.mean(),
                abs_output.std(),
                abs_output.min(),
                abs_output.max()
            ]).cpu().numpy()
            activation_stats[module.name].append(stats)
        elif isinstance(output, tuple):
            for i, tensor in enumerate(output):
                if isinstance(tensor, torch.Tensor):
                    abs_output = tensor.abs().detach()
                    stats = torch.stack([
                        abs_output.mean(),
                        abs_output.std(),
                        abs_output.min(),
                        abs_output.max()
                    ]).cpu().numpy()
                    activation_stats[f"{module.name}_output{i}"].append(stats)

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.LayerNorm, nn.Embedding)):
            module.name = name  # Attach the name to the module
            hooks.append(module.register_forward_hook(hook_fn))

    return activation_stats, hooks

def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()

def log_plot(activation_stats, step):
    fig = go.Figure()
    
    sorted_modules = sorted(activation_stats.keys())
    
    for i, module_name in enumerate(sorted_modules):
        layer_activations = np.concatenate(activation_stats[module_name])
        
        fig.add_trace(go.Box(
            y=layer_activations,
            name=module_name,
            boxpoints='outliers',
            jitter=0.3,
            whiskerwidth=0.2,
            marker_size=2,
            line_width=1
        ))

    fig.update_layout(
        title=f'Per-Layer Activation Distribution (Step {step})',
        yaxis_title='Absolute Activation Value',
        xaxis_title='Layers',
        boxmode='group',
        showlegend=False,
        hovermode='closest'
    )

    # Update x-axis to show module names
    fig.update_xaxes(
        ticktext=sorted_modules,
        tickvals=list(range(len(sorted_modules))),
        tickangle=45
    )

    # Save the plot as HTML and log it to wandb
    html_file = f'/tmp/activation_stats_step_{step}.html'
    fig.write_html(html_file)
    
    with open(html_file, 'r') as f:
        html_content = f.read()
    
    wandb.log({"Activation Distribution": wandb.Html(html_content)}, step=step)

@click.command()
@click.option(
    "--input_bin",
    default="dev/data/tinyshakespeare/tiny_shakespeare_val.bin",
    help="Input .bin to train on",
)
@click.option(
    "--input_val_bin", default="", help="Input .bin to eval validation loss on"
)
@click.option(
    "--output_dir", default="", help="Output directory for logs and checkpoints"
)
@click.option(
    "--batch_size", default=4, help="Batch size, in units of #batch dimensions"
)
@click.option("--sequence_length", default=64, help="Sequence length")
@click.option("--num_iterations", default=10, help="Number of iterations to run")
@click.option("--learning_rate", default=1e-4, help="Learning rate")
@click.option("--warmup_iters", default=0, help="Learning rate warmup iterations")
@click.option("--warmdown_iters", default=0, help="Learning rate warmdown iterations")
@click.option("--weight_decay", default=0.0, help="Weight decay")
@click.option("--val_loss_every", default=0, help="Evaluate val loss every N steps")
@click.option("--val_max_steps", default=20, help="Number of batches of val to average")
@click.option("--save_every", default=5000, help="Save checkpoint every N steps")
@click.option("--wandb_project", default="gpt-training", help="W&B project name")
@click.option("--wandb_run_name", default=None, help="W&B run name")
@click.option("--vocab_size", default=50257, help="Vocabulary size")
@click.option("--n_layer", default=12, help="Number of layers")
@click.option("--n_head", default=12, help="Number of attention heads")
@click.option("--n_embd", default=768, help="Embedding dimension")
@click.option("--ffn_dim", default=None, help="FFN dimension (default is 4 * n_embd)")
@click.option(
    "--lr_scheduler",
    default="warmup_cooldown",
    type=click.Choice(["linear_decay", "warmup_cooldown"]),
    help="Learning rate scheduler type",
)
def main(
    input_bin,
    input_val_bin,
    output_dir,
    batch_size,
    sequence_length,
    num_iterations,
    learning_rate,
    warmup_iters,
    warmdown_iters,
    weight_decay,
    val_loss_every,
    val_max_steps,
    save_every,
    wandb_project,
    wandb_run_name,
    vocab_size,
    n_layer,
    n_head,
    n_embd,
    ffn_dim,
    lr_scheduler,
):

    # Set up DDP
    assert torch.cuda.is_available(), "CUDA is required for DDP"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    print(f"using device: {device}")

    # Initialize wandb
    if master_process:
        if wandb_run_name is None:
            wandb_run_name = f"bs{batch_size}_seq{sequence_length}_lr{learning_rate}_wd{weight_decay}"

        wandb.init(project=wandb_project, name=wandb_run_name, config={
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "num_iterations": num_iterations,
            "learning_rate": learning_rate,
            "warmup_iters": warmup_iters,
            "warmdown_iters": warmdown_iters,
            "weight_decay": weight_decay,
            "val_loss_every": val_loss_every,
            "val_max_steps": val_max_steps,
            "save_every": save_every,
            "vocab_size": vocab_size,
            "n_layer": n_layer,
            "n_head": n_head,
            "n_embd": n_embd,
            "ffn_dim": ffn_dim,
            "lr_scheduler": lr_scheduler,
        })
        

    # Error checking and convenience variables
    B, T = batch_size, sequence_length
    tokens_per_fwdbwd = B * T * ddp_world_size

    # Set up context manager
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    # Load tokens
    train_loader = DistributedDataLoader(input_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = None
    if input_val_bin:
        val_loader = DistributedDataLoader(
            input_val_bin, B, T, ddp_rank, ddp_world_size
        )
    x, y = train_loader.next_batch()

    # Initialize the model
    if ffn_dim is None:
        ffn_dim = 4 * n_embd
    model_config = GPTConfig(
        vocab_size=vocab_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd
    )
    model = GPT(model_config)
    model = model.train().cuda()
    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True
    print0("compiling the model...")
    #model = torch.compile(model)
    print0("done compiling the model")

    # Wrap model in DDP
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module

    # Initialize optimizer
    optimizer = raw_model.configure_optimizers(
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        betas=(0.9, 0.95),
        device_type=device,
    )

    # Learning rate scheduler
    if lr_scheduler == "linear_decay":
        get_lr = lambda step: linear_decay_lr(step, num_iterations, learning_rate)
    else:  # warmup_cooldown
        get_lr = lambda step: warmup_cooldown_lr(
            step, num_iterations, learning_rate, warmup_iters, warmdown_iters
        )

    run_id = str(uuid.uuid4())

    # Create logging directory
    logfile = None
    if master_process and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        logfile = os.path.join(output_dir, f"{run_id}.log")
        with open(logfile, "w") as f:
            pass

    # Main training loop
    timings = []
    for step in range(num_iterations + 1):
        t0 = time.time()
        last_step = step == num_iterations

        # Validation
        if val_loss_every > 0 and (step % val_loss_every == 0 or last_step) and val_loader is not None:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                activation_stats, hooks = get_activation_stats(model.module)  # Note: use model.module to access the base model in DDP
                for _ in range(val_max_steps):
                    x_val, y_val = val_loader.next_batch()
                    _, loss = model(x_val, y_val, return_logits=False)
                    val_loss += loss
                remove_hooks(hooks)
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                val_loss /= val_max_steps
            print0(f"val loss {val_loss}")
            if master_process:
                wandb.log({"val_loss": val_loss}, step=step)
                log_plot(activation_stats, step)
                if logfile is not None:
                    with open(logfile, "a") as f:
                        f.write(f"s:{step} tel:{val_loss}\n")
        # Training
        model.train()
        with ctx:
            _, loss = model(x, y, return_logits=False)
            train_loss = loss.detach()
        x, y = train_loader.next_batch()
        loss.backward()
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        t1 = time.time()
        tokens_per_second = ddp_world_size * B * T / (t1 - t0)
        dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
        lossf = train_loss.item()
        print0(
            f"step {step+1:4d}/{num_iterations} | train loss {lossf:.6f} | lr {lr:.2e} | ({(t1-t0)*1000:.2f} ms | {tokens_per_second:.0f} tok/s)"
        )
        if master_process:
            wandb.log(
                {
                    "train_loss": lossf,
                    "learning_rate": lr,
                    "tokens_per_second": tokens_per_second,
                },
                step=step,
            )
            if logfile is not None:
                with open(logfile, "a") as f:
                    f.write(f"s:{step} trl:{lossf}\n")

        if step > 0 and step > num_iterations - 20:
            timings.append(t1 - t0)

        if master_process and (step + 1) % save_every == 0:
            log = dict(model=raw_model.state_dict(), args=locals())
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            torch.save(log, f"logs/{run_id}/model_step{step:06d}.pt")

    # Final logging and cleanup
    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    print0(
        f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB"
    )

    if master_process:
        log = dict(model=raw_model.state_dict(), args=locals())
        os.makedirs(f"logs/{run_id}", exist_ok=True)
        torch.save(log, f"logs/{run_id}/final.pt")
        wandb.finish()

    destroy_process_group()


if __name__ == "__main__":
    main()
