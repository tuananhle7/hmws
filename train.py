import shutil
import util
import losses


def train(
    algorithm,
    generative_model,
    guide,
    memory,
    optimizer,
    num_particles,
    num_iterations,
    stats,
    run_args,
):
    for iteration in range(num_iterations):
        if "mws" in algorithm:
            if algorithm == "mws":
                loss_fn = losses.get_mws_loss
                # print(f"memory = {memory[0]}")
            elif algorithm == "cmws":
                loss_fn = losses.get_cmws_loss
            loss, memory = loss_fn(generative_model, guide, memory, num_particles)
            # print(f"memory = {memory}")
        else:
            if algorithm == "rws":
                loss_fn = losses.get_rws_loss
            elif algorithm == "elbo":
                loss_fn = losses.get_elbo_loss
            loss = loss_fn(generative_model, guide, num_particles)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        stats.losses.append(loss.detach().item())
        if iteration % run_args.log_interval == 0:
            print(f"Iteration {iteration} | Loss {stats.losses[-1]:.2f}")

        if iteration % run_args.save_interval == 0:
            util.save_checkpoint(
                util.get_checkpoint_path(run_args),
                generative_model,
                guide,
                optimizer,
                memory,
                stats,
                run_args=run_args,
            )
            shutil.copy(
                util.get_checkpoint_path(run_args), util.get_checkpoint_path(run_args, iteration),
            )

    util.save_checkpoint(
        util.get_checkpoint_path(run_args),
        generative_model,
        guide,
        optimizer,
        memory,
        stats,
        run_args=run_args,
    )
