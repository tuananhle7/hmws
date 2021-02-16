import util
import models
import pyro


def train(model, optimizer, stats, args):
    checkpoint_path = util.get_checkpoint_path(args)
    num_iterations_so_far = len(stats.losses)

    generative_model, guide = model

    # Initialize optimizer for pyro models
    svi = pyro.infer.SVI(
        model=generative_model,
        guide=guide,
        optim=optimizer,
        loss=pyro.infer.ReweightedWakeSleep(
            num_particles=args.num_particles,
            vectorize_particles=False,
            model_has_params=True,
            insomnia=args.insomnia,
        ),
    )
    util.logging.info(f"Using pyro version {pyro.__version__}")

    for iteration in range(num_iterations_so_far, args.num_iterations):
        # Generate data
        obs = models.generate_from_true_generative_model(
            args.batch_size, device=generative_model.device
        )

        # Step gradient
        loss = svi.step(obs)

        # Turn loss into a scalar
        if isinstance(loss, tuple):
            loss = sum(loss)

        # Record stats
        stats.losses.append(loss)

        # Log
        if iteration % args.log_interval == 0:
            util.logging.info(f"Iteration {iteration} | Loss = {stats.losses[-1]:.0f}")

        # Make a model tuple
        model = generative_model, guide

        # Save checkpoint
        if iteration % args.save_interval == 0 or iteration == args.num_iterations - 1:
            util.save_checkpoint(checkpoint_path, model, optimizer, stats, run_args=args)

        if iteration % args.checkpoint_interval == 0:
            util.save_checkpoint(
                util.get_checkpoint_path(args, checkpoint_iteration=iteration),
                model,
                optimizer,
                stats,
                run_args=args,
            )

        # End training based on `iteration`
        iteration += 1
        if iteration == args.num_iterations:
            break
