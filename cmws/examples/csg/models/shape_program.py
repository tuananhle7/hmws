import torch
import torch.nn as nn
from cmws import util
from cmws.examples.csg import render
from cmws.examples.csg import util as csg_util

PROGRAMS = {
    0: "H",
    1: "R",
    2: "H + R",
    3: "H + H",
    4: "R + R",
    5: "H - H",
    6: "H - R",
    7: "R - H",
    8: "R - R",
}


def latent_to_str(latent, fixed_scale=False):
    """
    Args
        latent
            program_id []
            heart_poses
                raw_positions [max_num_shapes, 2]
                raw_scales [max_num_shapes]
            rectangle_poses [max_num_shapes, 4]

    Returns (str)
    """
    # Extract component strings
    if fixed_scale:
        program_id, raw_positions, rectangle_poses = latent

        heart_1_str = csg_util.heart_pose_to_str(raw_positions[0], fixed_scale=fixed_scale)
        heart_2_str = csg_util.heart_pose_to_str(raw_positions[1], fixed_scale=fixed_scale)
    else:
        program_id, (raw_positions, raw_scales), rectangle_poses = latent

        heart_1_str = csg_util.heart_pose_to_str(
            (raw_positions[0], raw_scales[0]), fixed_scale=fixed_scale
        )
        heart_2_str = csg_util.heart_pose_to_str(
            (raw_positions[1], raw_scales[1]), fixed_scale=fixed_scale
        )

    rectangle_1_str = csg_util.rectangle_pose_to_str(rectangle_poses[0])
    rectangle_2_str = csg_util.rectangle_pose_to_str(rectangle_poses[1])

    program = PROGRAMS[int(program_id.item())]
    if program == "H":
        return heart_1_str
    elif program == "R":
        return rectangle_1_str
    elif program == "H + R":
        return f"{heart_1_str}+\n{rectangle_1_str}"
    elif program == "H + H":
        return f"{heart_1_str}+\n{heart_2_str}"
    elif program == "R + R":
        return f"{rectangle_1_str}+\n{rectangle_2_str}"
    elif program == "H - H":
        return f"{heart_1_str}-\n{heart_2_str}"
    elif program == "H - R":
        return f"{heart_1_str}-\n{rectangle_1_str}"
    elif program == "R - H":
        return f"{rectangle_1_str}-\n{heart_1_str}"
    elif program == "R - R":
        return f"{rectangle_1_str}-\n{rectangle_2_str}"
    else:
        raise RuntimeError("program not found")


def evaluate_log_prob(program_id, heart_log_prob, rectangle_log_prob):
    """
    Args
        program_id [*shape]
        heart_log_prob [*shape, max_num_shapes]
        rectangle_log_prob [*shape, max_num_shapes]

    Returns [*shape]
    """
    shape = program_id.shape
    max_num_shapes = heart_log_prob.shape[-1]
    num_samples = int(torch.tensor(shape).prod().long().item())

    program_id_flattened = program_id.reshape(-1)
    heart_log_prob_flattened = heart_log_prob.reshape(-1, max_num_shapes)
    rectangle_log_prob_flattened = rectangle_log_prob.reshape(-1, max_num_shapes)

    log_probs = []
    for sample_id in range(num_samples):
        program = PROGRAMS[int(program_id_flattened[sample_id].item())]
        heart_1_log_prob, heart_2_log_prob = (
            heart_log_prob_flattened[sample_id, 0],
            heart_log_prob_flattened[sample_id, 1],
        )
        rectangle_1_log_prob, rectangle_2_log_prob = (
            rectangle_log_prob_flattened[sample_id, 0],
            rectangle_log_prob_flattened[sample_id, 1],
        )

        if program == "H":
            log_probs.append(heart_1_log_prob)
        elif program == "R":
            log_probs.append(rectangle_1_log_prob)
        elif program == "H + R":
            log_probs.append(heart_1_log_prob + rectangle_1_log_prob)
        elif program == "H + H":
            log_probs.append(heart_1_log_prob + heart_2_log_prob)
        elif program == "R + R":
            log_probs.append(rectangle_1_log_prob + rectangle_2_log_prob)
        elif program == "H - H":
            log_probs.append(heart_1_log_prob + heart_2_log_prob)
        elif program == "H - R":
            log_probs.append(heart_1_log_prob + rectangle_1_log_prob)
        elif program == "R - H":
            log_probs.append(rectangle_1_log_prob + heart_1_log_prob)
        elif program == "R - R":
            log_probs.append(rectangle_1_log_prob + rectangle_2_log_prob)
        else:
            raise RuntimeError("program not found")
    return torch.stack(log_probs).view(shape)


def evaluate_obs(program_id, hearts_obs, rectangles_obs):
    """
    Args
        program_id [*shape]
        hearts_obs [*shape, max_num_shapes, num_rows, num_cols]
        rectangles_obs [*shape, max_num_shapes, num_rows, num_cols]

    Returns [*shape, num_rows, num_cols]
    """
    shape = program_id.shape
    max_num_shapes, num_rows, num_cols = hearts_obs.shape[-3:]
    num_samples = int(torch.tensor(shape).prod().long().item())

    program_id_flattened = program_id.view(-1)
    hearts_obs_flattened = hearts_obs.view(-1, max_num_shapes, num_rows, num_cols)
    rectangles_obs_flattened = rectangles_obs.view(-1, max_num_shapes, num_rows, num_cols)

    obss = []
    for sample_id in range(num_samples):
        program = PROGRAMS[int(program_id_flattened[sample_id].item())]
        heart_1, heart_2 = (
            hearts_obs_flattened[sample_id, 0],
            hearts_obs_flattened[sample_id, 1],
        )
        rectangle_1, rectangle_2 = (
            rectangles_obs_flattened[sample_id, 0],
            rectangles_obs_flattened[sample_id, 1],
        )

        if program == "H":
            obs = heart_1
        elif program == "R":
            obs = rectangle_1
        elif program == "H + R":
            obs = heart_1 + rectangle_1
        elif program == "H + H":
            obs = heart_1 + heart_2
        elif program == "R + R":
            obs = rectangle_1 + rectangle_2
        elif program == "H - H":
            obs = heart_1 - heart_2
        elif program == "H - R":
            obs = heart_1 - rectangle_1
        elif program == "R - H":
            obs = rectangle_1 - heart_1
        elif program == "R - R":
            obs = rectangle_1 - rectangle_2
        else:
            raise RuntimeError("program not found")

        obss.append(torch.clamp(obs, 0, 1))
    return torch.stack(obss).view(*[*shape, num_rows, num_cols])


class TrueGenerativeModel(nn.Module):
    def __init__(self, im_size=64):
        super().__init__()
        self.im_size = im_size
        self.register_buffer("blank_canvas", torch.zeros((self.im_size, self.im_size)))
        self.max_num_shapes = 2
        self.num_programs = 9

    @property
    def device(self):
        return self.blank_canvas.device

    def get_heart_obs_dist(self, heart_pose, num_rows=None, num_cols=None):
        """p_H(obs | heart_pose)

        Args
            heart_pose
                raw_position [*shape, 2]
                raw_scale [*shape]
            num_rows (int)
            num_cols (int)

        Returns: distribution with batch_shape [*shape] and event_shape
            [num_rows, num_cols]
        """
        if num_rows is None:
            num_rows = self.im_size
        if num_cols is None:
            num_cols = self.im_size

        # Extract
        raw_position, raw_scale = heart_pose
        position = raw_position.sigmoid() - 0.5
        scale = raw_scale.sigmoid() * 0.8 + 0.1

        # Create blank canvas
        shape = scale.shape
        blank_canvas = torch.zeros((*shape, num_rows, num_cols), device=self.device)

        return torch.distributions.Independent(
            torch.distributions.Bernoulli(
                probs=render.render_heart((position, scale), blank_canvas).clamp(1e-6, 1 - 1e-6)
            ),
            reinterpreted_batch_ndims=2,
        )

    def get_rectangle_obs_dist(self, rectangle_pose, num_rows=None, num_cols=None):
        """p_H(obs | rectangle_pose)

        Args
            rectangle_pose [*shape, 4]
            num_rows (int)
            num_cols (int)

        Returns: distribution with batch_shape [*shape] and event_shape
            [num_rows, num_cols]
        """
        if num_rows is None:
            num_rows = self.im_size
        if num_cols is None:
            num_cols = self.im_size

        # Create blank canvas
        shape = rectangle_pose.shape[:-1]
        blank_canvas = torch.zeros((*shape, num_rows, num_cols), device=self.device)

        return torch.distributions.Independent(
            torch.distributions.Bernoulli(
                probs=render.render_rectangle(rectangle_pose, blank_canvas).clamp(1e-6, 1 - 1e-6)
            ),
            reinterpreted_batch_ndims=2,
        )

    @property
    def rectangle_pose_dist(self):
        """p_R(z_R)

        Returns distribution with batch_shape [] and event_shape [4]
        """
        return csg_util.RectanglePoseDistribution(self.device)

    @property
    def heart_pose_dist(self):
        """p_H(z_H)

        Returns distribution with batch_shape [] and event_shape ([2], [])
        """
        # Position distribution
        raw_position_dist = torch.distributions.Independent(
            torch.distributions.Normal(
                torch.zeros((2,), device=self.device), torch.ones((2,), device=self.device)
            ),
            reinterpreted_batch_ndims=1,
        )

        # Scale distribution
        raw_scale_dist = torch.distributions.Normal(
            torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device)
        )

        return util.JointDistribution([raw_position_dist, raw_scale_dist])

    @property
    def program_id_dist(self):
        """p_I(I)"""
        return torch.distributions.Categorical(
            logits=torch.ones((self.num_programs,), device=self.device)
        )

    def sample(self, sample_shape=[]):
        """
        Args:
            sample_shape: list-like object (default [])

        Returns:
            latent
                program_id [*sample_shape]
                heart_poses
                    raw_positions [*sample_shape, max_num_shapes, 2]
                    raw_scales [*sample_shape, max_num_shapes]
                rectangle_poses [*sample_shape, max_num_shapes, 4]
            obs [*sample_shape, im_size, im_size]
        """
        # Sample LATENT
        program_id = self.program_id_dist.sample(sample_shape)
        heart_poses = self.heart_pose_dist.sample([*sample_shape, self.max_num_shapes])
        rectangle_poses = self.rectangle_pose_dist.sample([*sample_shape, self.max_num_shapes])

        # Sample OBS
        # [*sample_shape, max_num_shapes, im_size, im_size]
        hearts_obs = self.get_heart_obs_dist(heart_poses, self.im_size, self.im_size).sample()
        rectangles_obs = self.get_rectangle_obs_dist(
            rectangle_poses, self.im_size, self.im_size
        ).sample()

        # Select OBS
        # [*sample_shape, im_size, im_size]
        obs = evaluate_obs(program_id, hearts_obs, rectangles_obs)

        return (program_id, heart_poses, rectangle_poses), obs

    def get_obs_probs(self, latent):
        """
        Args:
            latent
                program_id [*shape]
                heart_poses
                    raw_positions [*shape, max_num_shapes, 2]
                    raw_scales [*shape, max_num_shapes]
                rectangle_poses [*shape, max_num_shapes, 4]

        Returns:
            obs_probs [*shape, im_size, im_size]
        """
        program_id, heart_poses, rectangle_poses = latent

        # Sample OBS
        # [*shape, max_num_shapes, im_size, im_size]
        hearts_obs_probs = self.get_heart_obs_dist(
            heart_poses, self.im_size, self.im_size
        ).base_dist.probs
        rectangles_obs_probs = self.get_rectangle_obs_dist(
            rectangle_poses, self.im_size, self.im_size
        ).base_dist.probs

        # Select OBS
        # [*shape, im_size, im_size]
        obs_probs = evaluate_obs(program_id, hearts_obs_probs, rectangles_obs_probs)

        return obs_probs


class GenerativeModel(nn.Module):
    def __init__(self, im_size=64):
        super().__init__()
        self.im_size = im_size
        self.mlp = util.init_mlp(2, 1, 100, 3, non_linearity=nn.ReLU())
        self.max_num_shapes = 2
        self.num_programs = 9

    @property
    def device(self):
        return next(self.mlp.parameters()).device

    def get_heart_obs_dist(self, heart_pose, num_rows=None, num_cols=None):
        """p_H(obs | heart_pose)

        Args
            heart_pose
                raw_position [*shape, 2]
                raw_scale [*shape]
            num_rows (int)
            num_cols (int)

        Returns: distribution with batch_shape [*shape] and event_shape
            [num_rows, num_cols]
        """
        if num_rows is None:
            num_rows = self.im_size
        if num_cols is None:
            num_cols = self.im_size
        # Extract
        raw_position, raw_scale = heart_pose
        position = raw_position.sigmoid() - 0.5
        scale = raw_scale.sigmoid() * 0.8 + 0.1

        # util.logging.info(f"position = {position} | scale = {scale}")
        shape = scale.shape
        # num_samples = int(torch.tensor(shape).prod().item())
        position_x, position_y = position[..., 0], position[..., 1]  # [*shape]
        # [num_rows, num_cols]
        canvas_x, canvas_y = render.get_canvas_xy(num_rows, num_cols, self.device)

        # Shift and scale
        # [num_samples, num_rows, num_cols]
        canvas_x = (canvas_x[None] - position_x.view(-1, 1, 1)) / scale.view(-1, 1, 1)
        canvas_y = (canvas_y[None] - position_y.view(-1, 1, 1)) / scale.view(-1, 1, 1)

        # Build input
        # [num_samples * num_rows * num_cols, 2]
        mlp_input = torch.stack([canvas_x, canvas_y], dim=-1).view(-1, 2)

        # Run MLP
        logits = self.mlp(mlp_input).view(*[*shape, num_rows, num_cols])

        if torch.isnan(logits).any():
            raise RuntimeError("nan")

        return torch.distributions.Independent(
            torch.distributions.Bernoulli(logits=logits), reinterpreted_batch_ndims=2
        )

    def get_rectangle_obs_dist(self, rectangle_pose, num_rows=None, num_cols=None):
        """p_H(obs | rectangle_pose)

        Args
            rectangle_pose [*shape, 4]
            num_rows (int)
            num_cols (int)

        Returns: distribution with batch_shape [*shape] and event_shape
            [num_rows, num_cols]
        """
        if num_rows is None:
            num_rows = self.im_size
        if num_cols is None:
            num_cols = self.im_size

        # Create blank canvas
        shape = rectangle_pose.shape[:-1]
        blank_canvas = torch.zeros((*shape, num_rows, num_cols), device=self.device)

        return torch.distributions.Independent(
            torch.distributions.Bernoulli(
                probs=render.render_rectangle(rectangle_pose, blank_canvas).clamp(1e-6, 1 - 1e-6)
            ),
            reinterpreted_batch_ndims=2,
        )

    @property
    def rectangle_pose_dist(self):
        """p_R(z_R)

        Returns distribution with batch_shape [] and event_shape [4]
        """
        return csg_util.RectanglePoseDistribution(self.device)

    @property
    def heart_pose_dist(self):
        """p_H(z_H)

        Returns distribution with batch_shape [] and event_shape ([2], [])
        """
        # Position distribution
        raw_position_dist = torch.distributions.Independent(
            torch.distributions.Normal(
                torch.zeros((2,), device=self.device), torch.ones((2,), device=self.device)
            ),
            reinterpreted_batch_ndims=1,
        )

        # Scale distribution
        raw_scale_dist = torch.distributions.Normal(
            torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device)
        )

        return util.JointDistribution([raw_position_dist, raw_scale_dist])

    @property
    def program_id_dist(self):
        """p_I(I)"""
        return torch.distributions.Categorical(
            logits=torch.ones((self.num_programs,), device=self.device)
        )

    def log_prob(self, latent, obs):
        """
        Args:
            latent
                program_id [*shape]
                heart_poses
                    raw_positions [*shape, max_num_shapes, 2]
                    raw_scales [*shape, max_num_shapes]
                rectangle_poses [*shape, max_num_shapes, 4]
            obs [*shape, num_rows, num_cols]

        Returns: [*shape]
        """
        # Deconstruct latent
        program_id, heart_poses, rectangle_poses = latent

        # PRIOR
        # Evaluate individual log probs
        # [*shape]
        program_id_log_prob = self.program_id_dist.log_prob(program_id)
        # [*shape, max_num_shapes]
        heart_poses_log_prob = self.heart_pose_dist.log_prob(heart_poses)
        # [*shape, max_num_shapes]
        rectangle_poses_log_prob = self.rectangle_pose_dist.log_prob(rectangle_poses)

        # Combine log probs
        # [*shape]
        poses_log_prob = evaluate_log_prob(
            program_id, heart_poses_log_prob, rectangle_poses_log_prob
        )

        latent_log_prob = program_id_log_prob + poses_log_prob

        # LIKELIHOOD
        # Expand obs
        shape = obs.shape[:-2]
        num_rows, num_cols = obs.shape[-2:]
        # [*shape, max_num_shapes, num_rows, num_cols]
        obs_expanded = obs[..., None, :, :].expand(
            *[*shape, self.max_num_shapes, num_rows, num_cols]
        )

        # Evaluate individual log probs
        # [*shape, max_num_shapes]
        hearts_obs_log_prob = self.get_heart_obs_dist(
            heart_poses, self.im_size, self.im_size
        ).log_prob(obs_expanded)
        # [*shape, max_num_shapes]
        rectangles_obs_log_prob = self.get_rectangle_obs_dist(
            rectangle_poses, self.im_size, self.im_size
        ).log_prob(obs_expanded)

        # Combine log probs
        # [*shape]
        obs_log_prob = evaluate_log_prob(program_id, hearts_obs_log_prob, rectangles_obs_log_prob)

        result = latent_log_prob + obs_log_prob

        if torch.isnan(result).any():
            raise RuntimeError("nan")

        if torch.isinf(result).any():
            raise RuntimeError("nan")

        return result

    def sample(self, sample_shape=[]):
        """
        Args:
            sample_shape: list-like object (default [])

        Returns:
            latent
                program_id [*sample_shape]
                heart_poses
                    raw_positions [*sample_shape, max_num_shapes, 2]
                    raw_scales [*sample_shape, max_num_shapes]
                rectangle_poses [*sample_shape, max_num_shapes, 4]
            obs [*sample_shape, im_size, im_size]
        """
        # Sample LATENT
        program_id = self.program_id_dist.sample(sample_shape)
        heart_poses = self.heart_pose_dist.sample([*sample_shape, self.max_num_shapes])
        rectangle_poses = self.rectangle_pose_dist.sample([*sample_shape, self.max_num_shapes])

        # Sample OBS
        # [*sample_shape, max_num_shapes, im_size, im_size]
        hearts_obs = self.get_heart_obs_dist(heart_poses, self.im_size, self.im_size).sample()
        rectangles_obs = self.get_rectangle_obs_dist(
            rectangle_poses, self.im_size, self.im_size
        ).sample()

        # Select OBS
        # [*sample_shape, im_size, im_size]
        obs = evaluate_obs(program_id, hearts_obs, rectangles_obs)

        return (program_id, heart_poses, rectangle_poses), obs

    def get_obs_probs(self, latent):
        """
        Args:
            latent
                program_id [*shape]
                heart_poses
                    raw_positions [*shape, max_num_shapes, 2]
                    raw_scales [*shape, max_num_shapes]
                rectangle_poses [*shape, max_num_shapes, 4]

        Returns:
            obs_probs [*shape, im_size, im_size]
        """
        program_id, heart_poses, rectangle_poses = latent

        # Sample OBS
        # [*shape, max_num_shapes, im_size, im_size]
        hearts_obs_probs = self.get_heart_obs_dist(
            heart_poses, self.im_size, self.im_size
        ).base_dist.probs
        rectangles_obs_probs = self.get_rectangle_obs_dist(
            rectangle_poses, self.im_size, self.im_size
        ).base_dist.probs

        # Select OBS
        # [*shape, im_size, im_size]
        obs_probs = evaluate_obs(program_id, hearts_obs_probs, rectangles_obs_probs)

        return obs_probs


class Guide(nn.Module):
    def __init__(self, im_size=64):
        super().__init__()
        self.max_num_shapes = 2
        self.num_programs = 9
        self.im_size = im_size
        self.cnn = util.init_cnn(output_dim=16)
        self.cnn_features_dim = 400  # computed manually

        # Object id MLP
        self.program_id_mlp = util.init_mlp(
            self.cnn_features_dim, self.num_programs, hidden_dim=100, num_layers=3
        )

        # Heart poses MLP
        self.raw_position_mlp = util.init_mlp(
            self.cnn_features_dim, self.max_num_shapes * 2 * 2, hidden_dim=100, num_layers=3
        )
        self.raw_scale_mlp = util.init_mlp(
            self.cnn_features_dim, self.max_num_shapes * 2, hidden_dim=100, num_layers=3
        )

        # Rectangle poses MLP
        self.rectangle_mlp = util.init_mlp(
            self.cnn_features_dim, self.max_num_shapes * 4 * 2, hidden_dim=100, num_layers=3
        )

    @property
    def device(self):
        return next(self.program_id_mlp.parameters()).device

    def get_cnn_features(self, obs, cnn_type):
        """
        Args:
            obs: [batch_size, im_size, im_size]
            cnn_type

        Returns: [batch_size, cnn_features_dim]
        """
        batch_size = obs.shape[0]
        return self.cnn(obs[:, None]).view(batch_size, -1)

    def get_program_id_dist(self, obs):
        """q_I(z_I | x)

        Args:
            obs: [*shape, im_size, im_size]

        Returns: dist with batch_shape [*shape] and event_shape []
        """
        shape = obs.shape[:-2]
        num_samples = int(torch.tensor(shape).prod().item())

        # [num_samples, cnn_features_dim]
        cnn_features = self.get_cnn_features(
            obs.reshape(num_samples, self.im_size, self.im_size), "program_id"
        )

        logits = self.program_id_mlp(cnn_features).view(*shape, self.num_programs)

        if torch.isnan(logits).any():
            raise RuntimeError("nan")

        return torch.distributions.Categorical(logits=logits)

    def get_heart_pose_dist(self, obs):
        """q_H(z_H | x)

        Args:
            obs: [*shape, im_size, im_size]

        Returns: dist with batch_shape [*shape, max_num_shapes] and event_shape ([2], [])
        """
        shape = obs.shape[:-2]
        num_samples = int(torch.tensor(shape).prod().item())

        # [num_samples, cnn_features_dim]
        cnn_features = self.get_cnn_features(
            obs.reshape(num_samples, self.im_size, self.im_size), "heart"
        )

        # Position dist
        position_raw_loc, position_raw_scale = self.raw_position_mlp(cnn_features).chunk(2, dim=-1)
        position_loc, position_scale = (
            position_raw_loc.view(*[*shape, self.max_num_shapes, 2]),
            position_raw_scale.exp().view(*[*shape, self.max_num_shapes, 2]),
        )
        raw_position_dist = torch.distributions.Independent(
            torch.distributions.Normal(position_loc, position_scale), reinterpreted_batch_ndims=1
        )

        # Scale dist
        scale_raw_loc, scale_raw_scale = self.raw_scale_mlp(cnn_features).chunk(2, dim=-1)
        scale_loc, scale_scale = (
            scale_raw_loc.view(*[*shape, self.max_num_shapes]),
            scale_raw_scale.view(*[*shape, self.max_num_shapes]).exp(),
        )
        raw_scale_dist = torch.distributions.Normal(scale_loc, scale_scale)

        return util.JointDistribution([raw_position_dist, raw_scale_dist])

    def get_rectangle_pose_dist(self, obs):
        """q_R(z_R | x)

        Args:
            obs: [*shape, im_size, im_size]

        Returns: dist with batch_shape [*shape, max_num_shapes] and event_shape [4]
        """
        shape = obs.shape[:-2]
        num_samples = int(torch.tensor(shape).prod().item())

        # [num_samples, cnn_features_dim]
        cnn_features = self.get_cnn_features(
            obs.reshape(num_samples, self.im_size, self.im_size), "rectangle"
        )

        raw_loc, raw_scale = self.rectangle_mlp(cnn_features).chunk(2, dim=-1)
        loc, scale = (
            raw_loc.view(*[*shape, self.max_num_shapes, 4]),
            raw_scale.exp().view(*[*shape, self.max_num_shapes, 4]),
        )
        return torch.distributions.Independent(
            torch.distributions.Normal(loc, scale), reinterpreted_batch_ndims=1
        )

    def log_prob(self, obs, latent):
        """
        Args:
            obs [*shape, im_size, im_size]
            latent
                program_id [*shape]
                heart_poses
                    raw_positions [*shape, max_num_shapes, 2]
                    raw_scales [*shape, max_num_shapes]
                rectangle_poses [*shape, max_num_shapes, 4]

        Returns: [*shape]
        """
        # Deconstruct latent
        program_id, heart_poses, rectangle_poses = latent

        # Compute individual log probs
        # [*shape]
        program_id_log_prob = self.get_program_id_dist(obs).log_prob(program_id)
        # [*shape, max_num_shapes]
        heart_poses_log_prob = self.get_heart_pose_dist(obs).log_prob(heart_poses)
        # [*shape, max_num_shapes]
        rectangle_poses_log_prob = self.get_rectangle_pose_dist(obs).log_prob(rectangle_poses)

        # Combine log probs
        result = program_id_log_prob + evaluate_log_prob(
            program_id, heart_poses_log_prob, rectangle_poses_log_prob
        )

        if torch.isnan(result).any():
            raise RuntimeError("nan")

        if torch.isinf(result).any():
            raise RuntimeError("nan")

        return result

    def sample(self, obs, sample_shape=[]):
        """
        Args:
            obs [*shape, im_size, im_size]

        Returns:
            latent
                program_id [*sample_shape, *shape]
                heart_poses
                    raw_position [*sample_shape, *shape, max_num_shapes, 2]
                    raw_scale [*sample_shape, max_num_shapes, *shape]
                rectangle_poses [*sample_shape, *shape, max_num_shapes, 4]

        """

        program_id = self.get_program_id_dist(obs).sample(sample_shape)
        heart_poses = self.get_heart_pose_dist(obs).sample(sample_shape)
        rectangle_poses = self.get_rectangle_pose_dist(obs).sample(sample_shape)

        return program_id, heart_poses, rectangle_poses


class TrueGenerativeModelFixedScale(nn.Module):
    def __init__(self, im_size=64):
        super().__init__()
        self.im_size = im_size
        self.register_buffer("blank_canvas", torch.zeros((self.im_size, self.im_size)))
        self.max_num_shapes = 2
        self.num_programs = 9

    @property
    def device(self):
        return self.blank_canvas.device

    def get_heart_obs_dist(self, heart_pose, num_rows=None, num_cols=None):
        """p_H(obs | heart_pose)

        Args
            heart_pose
                raw_position [*shape, 2]
                raw_scale [*shape]
            num_rows (int)
            num_cols (int)

        Returns: distribution with batch_shape [*shape] and event_shape
            [num_rows, num_cols]
        """
        if num_rows is None:
            num_rows = self.im_size
        if num_cols is None:
            num_cols = self.im_size

        # Extract
        raw_position = heart_pose
        position = raw_position.sigmoid() - 0.5
        shape = position.shape[:-1]
        device = position.device
        scale = torch.ones(shape, device=device)

        # Create blank canvas
        shape = scale.shape
        blank_canvas = torch.zeros((*shape, num_rows, num_cols), device=self.device)

        return torch.distributions.Independent(
            torch.distributions.Bernoulli(
                probs=render.render_heart((position, scale), blank_canvas).clamp(1e-6, 1 - 1e-6)
            ),
            reinterpreted_batch_ndims=2,
        )

    def get_rectangle_obs_dist(self, rectangle_pose, num_rows=None, num_cols=None):
        """p_H(obs | rectangle_pose)

        Args
            rectangle_pose [*shape, 4]
            num_rows (int)
            num_cols (int)

        Returns: distribution with batch_shape [*shape] and event_shape
            [num_rows, num_cols]
        """
        if num_rows is None:
            num_rows = self.im_size
        if num_cols is None:
            num_cols = self.im_size

        # Create blank canvas
        shape = rectangle_pose.shape[:-1]
        blank_canvas = torch.zeros((*shape, num_rows, num_cols), device=self.device)

        return torch.distributions.Independent(
            torch.distributions.Bernoulli(
                probs=render.render_rectangle(rectangle_pose, blank_canvas).clamp(1e-6, 1 - 1e-6)
            ),
            reinterpreted_batch_ndims=2,
        )

    @property
    def rectangle_pose_dist(self):
        """p_R(z_R)

        Returns distribution with batch_shape [] and event_shape [4]
        """
        return csg_util.SquarePoseDistribution(False, self.device)

    @property
    def heart_pose_dist(self):
        """p_H(z_H)

        Returns distribution with batch_shape [] and event_shape ([2], [])
        """
        # Position distribution
        raw_position_dist = torch.distributions.Independent(
            torch.distributions.Normal(
                torch.zeros((2,), device=self.device), torch.ones((2,), device=self.device)
            ),
            reinterpreted_batch_ndims=1,
        )

        return raw_position_dist

    @property
    def program_id_dist(self):
        """p_I(I)"""
        return torch.distributions.Categorical(
            logits=torch.ones((self.num_programs,), device=self.device)
        )

    def sample(self, sample_shape=[]):
        """
        Args:
            sample_shape: list-like object (default [])

        Returns:
            latent
                program_id [*sample_shape]
                heart_poses
                    raw_positions [*sample_shape, max_num_shapes, 2]
                    raw_scales [*sample_shape, max_num_shapes]
                rectangle_poses [*sample_shape, max_num_shapes, 4]
            obs [*sample_shape, im_size, im_size]
        """
        # Sample LATENT
        program_id = self.program_id_dist.sample(sample_shape)
        heart_poses = self.heart_pose_dist.sample([*sample_shape, self.max_num_shapes])
        rectangle_poses = self.rectangle_pose_dist.sample([*sample_shape, self.max_num_shapes])

        # Sample OBS
        # [*sample_shape, max_num_shapes, im_size, im_size]
        hearts_obs = self.get_heart_obs_dist(heart_poses, self.im_size, self.im_size).sample()
        rectangles_obs = self.get_rectangle_obs_dist(
            rectangle_poses, self.im_size, self.im_size
        ).sample()

        # Select OBS
        # [*sample_shape, im_size, im_size]
        obs = evaluate_obs(program_id, hearts_obs, rectangles_obs)

        return (program_id, heart_poses, rectangle_poses), obs

    def get_obs_probs(self, latent):
        """
        Args:
            latent
                program_id [*shape]
                heart_poses
                    raw_positions [*shape, max_num_shapes, 2]
                    raw_scales [*shape, max_num_shapes]
                rectangle_poses [*shape, max_num_shapes, 4]

        Returns:
            obs_probs [*shape, im_size, im_size]
        """
        program_id, heart_poses, rectangle_poses = latent

        # Sample OBS
        # [*shape, max_num_shapes, im_size, im_size]
        hearts_obs_probs = self.get_heart_obs_dist(
            heart_poses, self.im_size, self.im_size
        ).base_dist.probs
        rectangles_obs_probs = self.get_rectangle_obs_dist(
            rectangle_poses, self.im_size, self.im_size
        ).base_dist.probs

        # Select OBS
        # [*shape, im_size, im_size]
        obs_probs = evaluate_obs(program_id, hearts_obs_probs, rectangles_obs_probs)

        return obs_probs
