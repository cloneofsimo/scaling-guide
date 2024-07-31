from utils import *


@click.command()
@click.option(
    "--batch_size", default=4, help="Batch size, in units of #batch dimensions"
)
@click.option("--ffn_dim", default=None, help="FFN dimension (default is 4 * n_embd)")
@click.option(
    "--gradient_accumulation_steps",
    default=1,
    help="Number of steps to accumulate gradients",
)
@click.option("--gradient_clip", default=1.0, help="Gradient clipping threshold")
@click.option(
    "--gpt_embed_init_std",
    default=0.02,
    help="GPT embedding layer initialization standard deviation",
)
@click.option(
    "--gpt_linear_init_std",
    default=0.02,
    help="GPT linear layer initialization standard deviation",
)
@click.option(
    "--input_bin",
    default="dev/data/tinyshakespeare/tiny_shakespeare_val.bin",
    help="Input .bin to train on",
)
@click.option(
    "--input_val_bin", default="", help="Input .bin to eval validation loss on"
)
@click.option("--learning_rate", default=1e-4, help="Learning rate")
@click.option(
    "--lr_scheduler",
    default="warmup_cooldown",
    type=click.Choice(["linear_decay", "warmup_cooldown"]),
    help="Learning rate scheduler type",
)
@click.option("--n_embd", default=768, help="Embedding dimension")
@click.option("--n_head", default=12, help="Number of attention heads")
@click.option("--n_layer", default=12, help="Number of layers")
@click.option("--num_iterations", default=10, help="Number of iterations to run")
@click.option(
    "--output_dir", default="", help="Output directory for logs and checkpoints"
)
@click.option("--save_every", default=5000, help="Save checkpoint every N steps")
@click.option("--sequence_length", default=64, help="Sequence length")
@click.option("--val_loss_every", default=0, help="Evaluate val loss every N steps")
@click.option("--val_max_steps", default=20, help="Number of batches of val to average")
@click.option("--vocab_size", default=50257, help="Vocabulary size")
@click.option("--wandb_project", default="gpt-training", help="W&B project name")
@click.option("--wandb_run_name", default=None, help="W&B run name")
@click.option("--warmdown_iters", default=0, help="Learning rate warmdown iterations")
@click.option("--warmup_iters", default=0, help="Learning rate warmup iterations")
@click.option("--weight_decay", default=0.0, help="Weight decay")
def main(
    batch_size,
    ffn_dim,
    gradient_accumulation_steps,
    gradient_clip,
    gpt_embed_init_std,
    gpt_linear_init_std,
    input_bin,
    input_val_bin,
    learning_rate,
    lr_scheduler,
    n_embd,
    n_head,
    n_layer,
    num_iterations,
    output_dir,
    save_every,
    sequence_length,
    val_loss_every,
    val_max_steps,
    vocab_size,
    wandb_project,
    wandb_run_name,
    warmdown_iters,
    warmup_iters,
    weight_decay,
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

        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            entity="simo",
            config={
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
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "gpt_embed_init_std": gpt_embed_init_std,
                "gpt_linear_init_std": gpt_linear_init_std,
            },
        )

    # Error checking and convenience variables
    B, T = batch_size, sequence_length
    tokens_per_fwdbwd = B * T * ddp_world_size * gradient_accumulation_steps

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
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        gpt_linear_init_std=gpt_linear_init_std,
        gpt_embed_init_std=gpt_embed_init_std,
    )
    model = GPT(model_config)
    model = model.train().cuda()
    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True
    print0("compiling the model...")
    # model = torch.compile(model)
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

    prv_sd = {}
    for name, param in model.named_parameters():
        prv_sd[name] = param.data.clone().cpu()

    # Main training loop
    timings = []
    for step in range(num_iterations + 1):
        t0 = time.time()
        last_step = step == num_iterations

        # Validation
        if (
            val_loss_every > 0
            and (step % val_loss_every == 0 or last_step)
            and val_loader is not None
        ):

            cur_sd = {}
            for name, param in model.named_parameters():
                cur_sd[name] = param.data.clone().cpu()

            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                activation_stats, hooks = get_activation_stats(
                    model.module
                )  # Note: use model.module to access the base model in DDP
                for _ in range(val_max_steps):
                    x_val, y_val = val_loader.next_batch()
                    _, loss = model(x_val, y_val, return_logits=False)
                    val_loss += loss
                remove_hooks(hooks)
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                val_loss /= val_max_steps
            print0(f"val loss {val_loss}")
            if master_process:
                wandb.log({"val_loss": val_loss})
                log_plot(activation_stats, step)
                log_weight_plot(cur_sd, prv_sd, step)
                for name, param in model.named_parameters():
                    prv_sd[name] = param.data.clone().cpu()

        # Training
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(gradient_accumulation_steps):
            with ctx:
                _, loss = model(x, y, return_logits=False)
                train_loss = loss.detach()
            x, y = train_loader.next_batch()
            loss = loss / gradient_accumulation_steps
            loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()

        torch.cuda.synchronize()
        t1 = time.time()
        tokens_per_second = tokens_per_fwdbwd / (t1 - t0)
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
                    "grad_norm": grad_norm,
                },
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
