import glob
import os
import re

import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torch.nn as nn

import wandb

torch.set_float32_matmul_precision("high")

from collections import defaultdict

import numpy as np
import plotly.graph_objects as go
import torch


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


def get_activation_stats(model):
    activation_stats = defaultdict(list)

    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            abs_output = output.abs().detach().view(-1)[:10].cpu().numpy()
            activation_stats[module.name].append(abs_output)
        else:
            raise ValueError("Output is not a tensor")

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
        print(f"Layer {module_name} has {len(layer_activations)} activations")
        print(layer_activations)
        fig.add_trace(
            go.Box(
                y=layer_activations,
                name=module_name,
                boxpoints="outliers",
                jitter=0.3,
                marker_size=2,
            )
        )

    fig.update_layout(
        title=f"Per-Layer Activation Distribution (Step {step})",
        yaxis_title="Absolute Activation Value",
        xaxis_title="Layers",
        boxmode="group",
        showlegend=False,
        hovermode="closest",
    )

    # Update x-axis to show module names
    fig.update_xaxes(
        ticktext=sorted_modules, tickvals=list(range(len(sorted_modules))), tickangle=45
    )

    # Save the plot as HTML and log it to wandb
    html_file = f"/tmp/activation_stats_step_{step}.html"
    fig.write_html(html_file)

    with open(html_file, "r") as f:
        html_content = f.read()

    wandb.log({"Activation Distribution": wandb.Html(html_content)})


def log_weight_plot(model_cur_sd, model_prev_sd, step):
    # Initialize a new run

    # Create lists to store data for the plot
    layer_names = []
    std_devs = []
    l1_norms = []
    param_counts = []
    colors = []
    markers = []

    # Iterate over the parameters and compute necessary metrics
    for name, param in model_cur_sd.items():
        if name in model_prev_sd:
            prev_param = model_prev_sd[name]
            std_dev = param.std().item()
            l1_norm = torch.abs(param - prev_param).mean().item()
            param_count = param.numel()

            # Determine color based on the criteria using regex
            layer_match = re.match(r".*\.h\.(\d+)(?:\..*)?$", name)

            if layer_match:
                layer_num = int(layer_match.group(1))
                colors.append(layer_num)
            else:
                colors.append(-1)

            # Determine marker type
            if param.ndim == 1:
                markers.append("x")
            else:
                markers.append("circle")

            layer_names.append(name)
            std_devs.append(std_dev)
            l1_norms.append(np.log1p(l1_norm))  # log(1 + x) transformation
            param_counts.append(np.log(param_count))

    # Create a DataFrame for the plot
    df = pd.DataFrame(
        {
            "Layer Name": layer_names,
            "Standard Deviation": std_devs,
            "L1 Norm of Changes (log scale)": l1_norms,
            "Parameter Count (log)": param_counts,
            "Color": colors,
            "Marker": markers,
        }
    )

    # Determine the number of layers
    max_layer_num = df[df["Color"] != -1]["Color"].max()

    # Create a color scale for the layers (yellow to red)
    color_scale = px.colors.sequential.YlOrRd
    color_discrete_map = {
        i: color_scale[int(i * (len(color_scale) - 1) / max_layer_num)]
        for i in range(int(max_layer_num) + 1)
    }
    color_discrete_map[-1] = "blue"  # Blue for non-layer parameters

    # Create Plotly figure
    fig = px.scatter(
        df,
        x="Standard Deviation",
        y="L1 Norm of Changes (log scale)",
        size="Parameter Count (log)",
        color="Color",
        hover_name="Layer Name",
        title=f"Model Weight Distribution and Changes, step {step}",
        symbol="Marker",
        color_discrete_map=color_discrete_map,
        opacity=0.7,
    )

    #
    path_to_plotly_html = "/tmp/weight_distribution_changes.html"
    fig.write_html(path_to_plotly_html, auto_play=False)
    wandb.log(
        {
            "Weight Distribution and Changes": wandb.Html(
                open(path_to_plotly_html).read()
            )
        }
    )
