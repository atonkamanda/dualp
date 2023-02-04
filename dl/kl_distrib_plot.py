import torch
import matplotlib.pyplot as plt

def plot_kl_divergence(mean1, std1, mean2, std2):
    x = torch.linspace(-10, 10, 100)
    p = torch.distributions.Normal(mean1, std1)
    q = torch.distributions.Normal(mean2, std2)
    p_pdf = p.log_prob(x)
    q_pdf = q.log_prob(x)
    # Compute KL divergence using PyTorch
    #kl = torch.distributions.kl_divergence(p, q)

    plt.fill_between(x.numpy(), p.log_prob(x).exp().numpy(), color='blue', alpha=0.3, label='p')
    plt.fill_between(x.numpy(), q.log_prob(x).exp().numpy(), color='orange', alpha=0.3, label='q')
    plt.legend()
    plt.title(f'KL Divergence: {kl:.2f}')
    plt.show()
plot_kl_divergence(0, 1, 2, 1)

