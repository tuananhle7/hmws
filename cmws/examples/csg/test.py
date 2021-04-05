from cmws import util
import models.hearts

if __name__ == "__main__":
    # Sample
    device = "cuda"
    batch_size = 13

    true_generative_model = models.hearts.TrueGenerativeModel().to(device)
    latent, obs = true_generative_model.sample((batch_size,))

    # Plot
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, batch_size, figsize=(2 * batch_size, 2), sharex=True, sharey=True)
    for batch_id in range(batch_size):
        axs[batch_id].imshow(obs[batch_id].cpu(), cmap="Greys", vmin=0, vmax=1)
    util.save_fig(fig, "test.png")

    # Log probs
    guide = models.hearts.Guide().to(device)
    log_prob = guide.log_prob(obs, latent)
    util.logging.info(f"guide log prob = {log_prob}")

    generative_model = models.hearts.GenerativeModel().to(device)
    log_prob = generative_model.log_prob(latent, obs)
    util.logging.info(f"generative model log prob = {log_prob}")
