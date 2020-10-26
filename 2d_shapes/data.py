# import models
# import sys
# import torch


# class SyntheticDataset(torch.utils.data.Dataset):
#     """Synthetic dataset based on the generative model

#     Args;
#         generative_model
#         return_latent (bool; optional): Return `latent` in addition to `marbless` (default `False`)
#     """

#     def __init__(
#         self, generative_model, return_latent: bool = False,
#     ):
#         self.generative_model = generative_model
#         self.return_latent = return_latent

#     def __getitem__(self, index: int):
#         """
#         Args:
#             index (int)

#         Returns:
#             latent (only if return_latent is true):
#             obs
#         """
#         latent, obs = self.generative_model.sample(sample_shape=())
#         if self.return_latent:
#             return latent, obs
#         else:
#             return obs

#     def __len__(self):
#         return sys.maxsize


# def get_data_loader(
#     num_bags_range,
#     num_marbles_range,
#     random_num_marbles_per_bag,
#     batch_size,
#     device="cpu",
#     generative_model=None,
#     **data_loader_kwargs,
# ):
#     if generative_model is None:
#         generative_model = models.GenerativeModel().to(device)

#     dataset = SyntheticDataset(generative_model)
#     return torch.utils.data.DataLoader(
#         dataset, batch_size=batch_size, collate_fn=lambda x: x, **data_loader_kwargs
#     )
