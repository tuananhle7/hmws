import pyro
import torch

import cmws
from cmws import losses, util


def train(model, optimizer, stats, args):
    # Extract
    checkpoint_path = util.get_checkpoint_path(args)
    num_iterations_so_far = len(stats.losses)
    num_sleep_pretraining_iterations_so_far = len(stats.sleep_pretraining_losses)
    generative_model, guide, memory = model["generative_model"], model["guide"], model["memory"]
    device = generative_model.device

    # Initialize optimizer for pyro models
    if "_pyro" in args.model_type:
        if args.algorithm == "rws":
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
        elif args.algorithm == "elbo":
            svi = pyro.infer.SVI(
                model=generative_model, guide=guide, optim=optimizer, loss=pyro.infer.Trace_ELBO()
            )
        else:
            raise ValueError(f"Algorithm {args.algorithm} not supported in pyro")
        util.logging.info(f"Using pyro version {pyro.__version__}")

    # Sleep pretraining
    for iteration in range(
        num_sleep_pretraining_iterations_so_far, args.num_sleep_pretraining_iterations
    ):
        if "_pyro" in args.model_type:
            raise NotImplementedError
        else:
            # Zero grad
            optimizer.zero_grad()

            # Evaluate loss
            loss = losses.get_sleep_loss(
                generative_model, guide, args.num_particles * args.batch_size
            ).mean()

            # Compute gradient
            loss.backward()

            # Step gradient
            optimizer.step()

            # Record stats
            stats.sleep_pretraining_losses.append(loss.item())

        # Log
        if iteration % args.log_interval == 0:
            util.logging.info(
                f"Sleep Pretraining Iteration {iteration} | "
                f"Loss = {stats.sleep_pretraining_losses[-1]:.0f} | "
                f"Max GPU memory allocated = {util.get_max_gpu_memory_allocated_MB(device):.0f} MB"
            )

        # Make a model tuple
        model = {"generative_model": generative_model, "guide": guide, "memory": memory}

        # Save checkpoint
        if (
            iteration % args.save_interval == 0
            or iteration == args.num_sleep_pretraining_iterations - 1
        ):
            if "cmws.examples.stacking.models" in str(type(generative_model)):
                cmws.examples.stacking.util.save_checkpoint(
                    checkpoint_path, model, optimizer, stats, run_args=args
                )
            if "cmws.examples.stacking_3d.models" in str(type(generative_model)):
                cmws.examples.stacking_3d.util.save_checkpoint(
                    checkpoint_path, model, optimizer, stats, run_args=args
                )
            if "cmws.examples.csg.models" in str(type(generative_model)):
                cmws.examples.csg.util.save_checkpoint(
                    checkpoint_path, model, optimizer, stats, run_args=args
                )

    # Data
    if "cmws.examples.stacking.models.stacking" in str(type(generative_model)):
        # Use a data loader
        train_data_iterator = util.cycle(
            cmws.examples.stacking.data.get_stacking_data_loader(
                device, args.batch_size, test=False
            )
        )
        test_data_loader = cmws.examples.stacking.data.get_stacking_data_loader(
            device, args.batch_size, test=True
        )
    else:
        # Generate test data
        # NOTE: super weird but when this is put before sleep pretraining, sleep pretraining doesn't
        # work
        if "cmws.examples.stacking.models" in str(type(generative_model)):
            test_obs = cmws.examples.stacking.data.generate_test_obs(args, device)
        elif "cmws.examples.stacking_3d.models" in str(type(generative_model)):
            test_obs = cmws.examples.stacking_3d.data.generate_test_obs(device)
        elif "cmws.examples.csg.models" in str(type(generative_model)):
            test_obs = cmws.examples.csg.data.generate_test_obs(args, device)

    # Normal training
    for iteration in range(num_iterations_so_far, args.num_iterations):
        if "cmws.examples.stacking.models.stacking" in str(type(generative_model)):
            obs, obs_id = next(train_data_iterator)
        else:
            # Generate data
            if "cmws.examples.stacking.models" in str(type(generative_model)):
                obs = cmws.examples.stacking.data.generate_obs(args, args.batch_size, device)
            elif "cmws.examples.stacking_3d.models" in str(type(generative_model)):
                obs = cmws.examples.stacking_3d.data.generate_obs(args.batch_size, device)
            elif "cmws.examples.csg.models" in str(type(generative_model)):
                obs = cmws.examples.csg.data.generate_obs(args, args.batch_size, device)

        if "_pyro" in args.model_type:
            # Step gradient
            loss = svi.step(obs)

            # Turn loss into a scalar
            if isinstance(loss, tuple):
                loss = sum(loss)

            # Record stats
            stats.losses.append(loss)
        else:
            # Zero grad
            optimizer.zero_grad()

            # Evaluate loss
            if args.algorithm == "rws":
                loss = losses.get_rws_loss(
                    generative_model, guide, obs, args.num_particles, insomnia=args.insomnia
                ).mean()
            elif "elbo" in args.algorithm:
                loss = losses.get_elbo_loss(generative_model, guide, obs).mean()
            elif args.algorithm == "vimco":
                loss = losses.get_vimco_loss(
                    generative_model, guide, obs, args.num_particles
                ).mean()
            elif args.algorithm == "cmws":
                loss = losses.get_cmws_loss(
                    generative_model,
                    guide,
                    memory,
                    obs,
                    obs_id,
                    args.num_particles,
                    args.num_proposals_mws,
                ).mean()

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

        # Compute log p and KL
        if iteration % args.test_interval == 0 or iteration == args.num_iterations - 1:
            if "_pyro" in args.model_type:
                stats.log_ps.append([iteration, float("nan")])
                stats.kls.append([iteration, float("nan")])
            else:
                util.logging.info("Computing logp and KL")
                if "cmws.examples.stacking.models.stacking" in str(type(generative_model)):
                    log_p, kl = [], []
                    for test_obs, test_obs_id in test_data_loader:
                        log_p_, kl_ = losses.get_log_p_and_kl(
                            generative_model, guide, test_obs, args.test_num_particles
                        )
                        log_p.append(log_p_)
                        kl.append(kl_)
                    log_p = torch.cat(log_p)
                    kl = torch.cat(kl)
                else:
                    log_p, kl = losses.get_log_p_and_kl(
                        generative_model, guide, test_obs, args.test_num_particles
                    )
                stats.log_ps.append([iteration, log_p.mean().item()])
                stats.kls.append([iteration, kl.mean().item()])

        # Log
        if iteration % args.log_interval == 0:
            util.logging.info(
                f"Iteration {iteration} | Loss = {stats.losses[-1]:.0f} | "
                f"Log p = {stats.log_ps[-1][1]:.0f} | KL = {stats.kls[-1][1]:.0f} | "
                f"Max GPU memory allocated = {util.get_max_gpu_memory_allocated_MB(device):.0f} MB"
            )

        # Make a model tuple
        model = {"generative_model": generative_model, "guide": guide, "memory": memory}

        # Save checkpoint
        if iteration % args.save_interval == 0 or iteration == args.num_iterations - 1:
            if "cmws.examples.stacking.models" in str(type(generative_model)):
                cmws.examples.stacking.util.save_checkpoint(
                    checkpoint_path, model, optimizer, stats, run_args=args
                )
            elif "cmws.examples.stacking_3d.models" in str(type(generative_model)):
                cmws.examples.stacking_3d.util.save_checkpoint(
                    checkpoint_path, model, optimizer, stats, run_args=args
                )
            elif "cmws.examples.csg.models" in str(type(generative_model)):
                cmws.examples.csg.util.save_checkpoint(
                    checkpoint_path, model, optimizer, stats, run_args=args
                )

        if iteration % args.checkpoint_interval == 0:
            if "cmws.examples.stacking.models" in str(type(generative_model)):
                cmws.examples.stacking.util.save_checkpoint(
                    util.get_checkpoint_path(args, checkpoint_iteration=iteration),
                    model,
                    optimizer,
                    stats,
                    run_args=args,
                )
            elif "cmws.examples.stacking_3d.models" in str(type(generative_model)):
                cmws.examples.stacking_3d.util.save_checkpoint(
                    util.get_checkpoint_path(args, checkpoint_iteration=iteration),
                    model,
                    optimizer,
                    stats,
                    run_args=args,
                )
            elif "cmws.examples.csg.models" in str(type(generative_model)):
                cmws.examples.csg.util.save_checkpoint(
                    util.get_checkpoint_path(args, checkpoint_iteration=iteration),
                    model,
                    optimizer,
                    stats,
                    run_args=args,
                )
