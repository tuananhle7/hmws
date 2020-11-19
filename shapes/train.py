import losses
import util
from models import hearts
from models import heartangles
from models import shape_program
from models import no_rectangle
from models import ldif_representation
import pyro


def train(model, optimizer, stats, args):
    checkpoint_path = util.get_checkpoint_path(args)
    num_iterations_so_far = len(stats.losses)

    generative_model, guide = model
    if args.model_type == "hearts" or args.model_type == "hearts_pyro":
        true_generative_model = hearts.TrueGenerativeModel().to(guide.device)
    elif args.model_type == "heartangles":
        true_generative_model = heartangles.TrueGenerativeModel().to(guide.device)
    elif args.model_type == "shape_program":
        true_generative_model = shape_program.TrueGenerativeModel().to(guide.device)
    elif args.model_type == "no_rectangle":
        true_generative_model = no_rectangle.TrueGenerativeModel().to(guide.device)
    elif args.model_type == "ldif_representation":
        true_generative_model = ldif_representation.TrueGenerativeModel().to(guide.device)

    if args.model_type == "hearts_pyro":
        svi = pyro.infer.SVI(
            model=generative_model,
            guide=guide,
            optim=optimizer,
            loss=pyro.infer.Trace_ELBO()
            # loss=pyro.infer.ReweightedWakeSleep(
            #     num_particles=args.num_particles,
            #     vectorize_particles=False,
            #     model_has_params=True,
            #     insomnia=1.0,
            # ),
        )

    for iteration in range(num_iterations_so_far, args.num_iterations):
        if args.model_type == "hearts_pyro":
            if "rws" in args.algorithm:
                _, obs = true_generative_model.sample((args.batch_size,))
                loss = svi.step(obs)

            # Record stats
            stats.losses.append(loss)
        else:
            # Zero grad
            optimizer.zero_grad()

            # Evaluate loss
            if args.model_type == "rectangles":
                loss = losses.get_sleep_loss(generative_model, guide, args.batch_size).mean()
            elif args.model_type == "hearts":
                _, obs = true_generative_model.sample((args.batch_size,))
                loss = losses.get_elbo_loss(generative_model, guide, obs).mean()
            elif (
                args.model_type == "heartangles"
                or args.model_type == "shape_program"
                or args.model_type == "no_rectangle"
                or args.model_type == "ldif_representation"
            ):
                if args.algorithm == "sleep":
                    loss = losses.get_sleep_loss(
                        true_generative_model, guide, args.batch_size * args.num_particles
                    ).mean()
                else:
                    _, obs = true_generative_model.sample((args.batch_size,))
                    if "rws" in args.algorithm:
                        loss = losses.get_rws_loss(
                            generative_model, guide, obs, args.num_particles
                        ).mean()
                    elif "vimco" in args.algorithm:
                        loss = losses.get_vimco_loss(
                            generative_model, guide, obs, args.num_particles
                        ).mean()
                    elif "iwae" in args.algorithm:
                        loss = losses.get_iwae_loss(
                            generative_model, guide, obs, args.num_particles
                        ).mean()

                    if "sleep" in args.algorithm:
                        loss += losses.get_sleep_loss(
                            generative_model, guide, args.batch_size * args.num_particles
                        ).mean()

            loss.backward()

            # Step gradient
            optimizer.step()

            # Record stats
            stats.losses.append(loss.item())

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
