import numpy as np
import matplotlib.pyplot as plt

real = np.load("data/processed/images.npy")

fake = np.load("data/processed/images.npy")

plt.hist(real.flatten(), bins=50, alpha=0.5, label="real")
plt.hist(fake.flatten(), bins=50, alpha=0.5, label="fake")

plt.legend()

plt.savefig("figures/hist_real_fake.png")

plt.show()