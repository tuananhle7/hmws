import torch
import cmws.util


def test_condition_mvn_shape():
    device = cmws.util.get_device()
    dim, dim_y = 10, 3
    dim_x = dim - dim_y

    for shape in [[], [2], [2, 3]]:
        num_elements = cmws.util.get_num_elements(shape)

        # Init p(x, y)
        loc = torch.randn(*[*shape, dim], device=device)
        cov = (
            torch.eye(dim, device=device)[None]
            .expand(num_elements, dim, dim)
            .view(*[*shape, dim, dim])
        )
        multivariate_normal_dist = torch.distributions.MultivariateNormal(loc, cov)

        # Init y
        y = torch.randn(*[*shape, dim_y], device=device)

        # Compute p(x | y)
        conditioned_multivariate_normal_dist = cmws.util.condition_mvn(multivariate_normal_dist, y)

        # Extract params of p(x | y)
        loc_new = conditioned_multivariate_normal_dist.mean
        cov_new = conditioned_multivariate_normal_dist.covariance_matrix

        assert list(loc_new.shape) == shape + [dim_x]
        assert list(cov_new.shape) == shape + [dim_x, dim_x]
