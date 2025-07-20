import wandb

# Start a run
run = wandb.init(project="demo-disable-auto-step")

for iteration in range(3):
    loss = 1.0 / (iteration + 1)
    lr = 0.1 * (iteration + 1)

    # Log both metrics under the SAME step, no auto-increment
    # run.log({"loss": loss}, step=iteration, commit=False)
    # run.log({"lr": lr}, step=iteration, commit=False)

    run.log({
        "train/loss": loss,
        "train/lr": lr
        }, step=iteration
    )

print("Done")
run.finish()