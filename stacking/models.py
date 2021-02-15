import pyro
import render
import torch


def sample_stacking_program(num_primitives, device):
    # Init
    stacking_program = []
    num_sampled_primitives = 0
    available_primitive_ids = list(range(num_primitives))

    # Sample first primitive
    raw_primitive_id_logits = torch.ones((len(available_primitive_ids),), device=device)
    raw_primitive_id = pyro.sample(
        f"raw_primitive_id_{num_sampled_primitives}",
        pyro.distributions.Categorical(logits=raw_primitive_id_logits),
    )
    primitive_id = available_primitive_ids.pop(raw_primitive_id)
    num_sampled_primitives += 1
    stacking_program.append(primitive_id)

    # Sample the rest
    end_program = False
    while (not end_program) and (len(available_primitive_ids) > 0):
        # Sample an action for the next primitive
        # Action 0: put to the left of existing stack
        # Action 1: end program
        num_actions = 2
        action_id_logits = torch.ones((num_actions,), device=device)
        action_id = pyro.sample(
            f"action_id_{num_sampled_primitives}",
            pyro.distributions.Categorical(logits=action_id_logits),
        )

        if action_id == 1:
            # End program
            end_program = True
            break
        else:
            # Sample primitive
            raw_primitive_id_logits = torch.ones((len(available_primitive_ids),), device=device)
            raw_primitive_id = pyro.sample(
                f"raw_primitive_id_{num_sampled_primitives}",
                pyro.distributions.Categorical(logits=raw_primitive_id_logits),
            )
            primitive_id = available_primitive_ids.pop(raw_primitive_id)
            num_sampled_primitives += 1

            # Add to the stacking program based on previous action
            stacking_program.append(primitive_id)

    return torch.tensor(stacking_program, device=device)


def stacking_program_to_str(stacking_program, primitives):
    return [primitives[primitive_id].name for primitive_id in stacking_program]


def sample_raw_locations(stacking_program):
    device = stacking_program[0].device
    dist = pyro.distributions.Normal(torch.tensor(0.0, device=device), 1)
    raw_locations = []
    for primitive_id, stack in enumerate(stacking_program):
        raw_locations.append(pyro.sample(f"primitive_{primitive_id}_raw_loc", dist))
    return raw_locations


def generate_from_true_generative_model(device, num_channels=3, num_rows=256, num_cols=256):
    # Define params
    primitives = [
        render.Square(
            "A", torch.tensor([1.0, 0.0, 0.0], device=device), torch.tensor(0.1, device=device)
        ),
        render.Square(
            "B", torch.tensor([0.0, 1.0, 0.0], device=device), torch.tensor(0.2, device=device)
        ),
        render.Square(
            "C", torch.tensor([0.0, 0.0, 1.0], device=device), torch.tensor(0.3, device=device)
        ),
    ]
    num_primitives = len(primitives)

    # Sample
    stacking_program = sample_stacking_program(num_primitives, device)
    raw_locations = sample_raw_locations(stacking_program)

    # Render
    img = render.render(
        primitives, stacking_program, raw_locations, num_channels, num_rows, num_cols
    )

    return img
