import cmws
import cmws.examples.stacking.models.stacking
import torch
from cmws.losses import get_cmws_loss, get_log_marginal_joint, get_cmws_2_loss


def test_get_log_marginal_joint_dims():
    num_channels, im_size = 3, 32
    shape = [2, 3]
    discrete_shape = [7, 8]
    num_particles = 12

    generative_model = cmws.examples.stacking.models.stacking.GenerativeModel(im_size=im_size)
    guide = cmws.examples.stacking.models.stacking.Guide(im_size=im_size)

    obs = torch.rand(*[*shape, num_channels, im_size, im_size])
    num_blocks = torch.randint(2, size=discrete_shape + shape) + 1
    stacking_program = torch.randint(2, size=discrete_shape + shape + [guide.max_num_blocks])
    discrete_latent = (num_blocks, stacking_program)

    log_marginal_joint = get_log_marginal_joint(
        generative_model, guide, discrete_latent, obs, num_particles
    )

    assert list(log_marginal_joint.shape) == discrete_shape + shape


def test_get_cmws_loss_dims():
    num_channels, im_size = 3, 32
    batch_size = 7
    obs = torch.rand(*[batch_size, num_channels, im_size, im_size])
    obs_id = torch.arange(batch_size)
    num_particles = 8
    num_proposals = 9

    device = "cpu"
    args = cmws.examples.stacking.run.get_args_parser().parse_args([])
    args.algorithm = "cmws"
    model, optimizer, stats = cmws.examples.stacking.util.init(args, device)
    loss = get_cmws_loss(
        model["generative_model"],
        model["guide"],
        model["memory"],
        obs,
        obs_id,
        num_particles,
        num_proposals,
    )

    assert len(loss) == batch_size


def test_get_cmws_2_loss_dims():
    num_channels, im_size = 3, 32
    batch_size = 7
    obs = torch.rand(*[batch_size, num_channels, im_size, im_size])
    obs_id = torch.arange(batch_size)
    num_particles = 8
    num_proposals = 9

    device = "cpu"
    args = cmws.examples.stacking.run.get_args_parser().parse_args([])
    args.algorithm = "cmws"
    model, optimizer, stats = cmws.examples.stacking.util.init(args, device)
    loss = get_cmws_2_loss(
        model["generative_model"],
        model["guide"],
        model["memory"],
        obs,
        obs_id,
        num_particles,
        num_proposals,
    )

    assert len(loss) == batch_size
