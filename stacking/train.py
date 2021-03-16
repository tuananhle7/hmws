import util
from models import stacking_pyro
import pyro
import torch
import losses


def train(model, optimizer, stats, args):
    checkpoint_path = util.get_checkpoint_path(args)
    num_iterations_so_far = len(stats.losses)

    generative_model, guide = model

    # Initialize optimizer for pyro models
    if "_pyro" in args.model_type:
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
        if "_pyro" in args.model_type:
            # Generate data
            obs = stacking_pyro.generate_from_true_generative_model(
                args.batch_size,
                num_primitives=args.data_num_primitives,
                device=generative_model.device,
            )

            # Step gradient
            loss = svi.step(obs)

            # Turn loss into a scalar
            if isinstance(loss, tuple):
                loss = sum(loss)

            # Record stats
            stats.losses.append(loss)
        else:
            # Generate data
            if args.model_type == "one_primitive":
                obs = stacking_pyro.generate_from_true_generative_model(
                    args.batch_size, num_primitives=1, device=generative_model.device,
                )
            elif args.model_type == "two_primitives":
                obs = stacking_pyro.generate_from_true_generative_model(
                    args.batch_size,
                    num_primitives=2,
                    device=generative_model.device,
                    fixed_num_blocks=True,
                )

            # Zero grad
            optimizer.zero_grad()

            # Evaluate loss
            if args.algorithm == "rws":
                loss = losses.get_rws_loss(generative_model, guide, obs, args.num_particles).mean()
            elif "elbo" in args.algorithm:
                loss = losses.get_elbo_loss(generative_model, guide, obs).mean()
            # elif "iwae" in args.algorithm:
            #     loss = losses.get_iwae_loss(
            #         generative_model, guide, obs, args.num_particles
            #     ).mean()

            # if "sleep" in args.algorithm:
            #     loss += losses.get_sleep_loss(
            #         generative_model, guide, args.batch_size * args.num_particles
            #     ).mean()

            # Compute gradient
            loss.backward()

            # Step gradient
            optimizer.step()

            # Record stats
            stats.losses.append(loss.item())

        # Check nan
        for name, param in generative_model.named_parameters():
            if torch.isnan(param).any():
                util.logging.info(f"Param {name} is nan: {param}")
                import pdb

                pdb.set_trace()
        for name, param in guide.named_parameters():
            if torch.isnan(param).any():
                util.logging.info(f"Param {name} is nan: {param}")
                import pdb

                pdb.set_trace()

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
