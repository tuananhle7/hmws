import shutil
import util
import losses


def train(
    run_args, generative_model, guide, memory, optimizer, stats,
):
    for iteration in range(run_args.num_iterations):
        if "mws" in run_args.algorithm:
            if run_args.algorithm == "mws":
                loss, memory = losses.get_mws_loss(
                    generative_model, guide, memory, run_args.num_particles
                )
            elif run_args.algorithm == "cmws":
                if run_args.cmws_estimator == "is":
                    run_args.num_cmws_iterations = None
                elif run_args.cmws_estimator == "sgd":
                    run_args.num_particles = None
                elif run_args.cmws_estimator == "exact":
                    run_args.num_particles = None
                    run_args.num_cmws_iterations = None
                loss, memory = losses.get_cmws_loss(
                    generative_model,
                    guide,
                    memory,
                    run_args.num_cmws_mc_samples,
                    num_particles=run_args.num_particles,
                    num_iterations=run_args.num_cmws_iterations,
                )
            # print(f"memory = {memory}")
        else:
            if run_args.algorithm == "rws":
                loss_fn = losses.get_rws_loss
            elif run_args.algorithm == "elbo":
                loss_fn = losses.get_elbo_loss
            loss = loss_fn(generative_model, guide, run_args.num_particles)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        stats.locs_errors.append(
            util.get_locs_error(
                generative_model,
                guide,
                memory,
                num_particles=run_args.num_particles,
                num_iterations=run_args.num_cmws_iterations,
            )
        )
        if run_args.algorithm == "cmws":
            stats.cmws_memory_errors.append(
                util.get_cmws_memory_error(
                    generative_model,
                    guide,
                    memory,
                    num_particles=run_args.num_particles,
                    num_iterations=run_args.num_cmws_iterations,
                )
            )
            stats.inference_errors.append(stats.locs_errors[-1] + stats.cmws_memory_errors[-1])
        stats.losses.append(loss.detach().item())
        if iteration % run_args.log_interval == 0:
            util.logging.info(f"Iteration {iteration} | Loss {stats.losses[-1]:.2f}")

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
