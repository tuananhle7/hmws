def get_sleep_loss(generative_model, guide, num_samples):
    """
    Args:
        generative_model
        guide
        num_samples (int)

    Returns: [num_samples]
    """
    latent, obs = generative_model.sample(sample_shape=(num_samples,))
    return -guide.log_prob(obs, latent)
