import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare

# Simulate a dice roll experiment
num_rolls = 1000  # number of dice rolls

# Scenario 1: Uniform probabilities
# Sample probabilities from a uniform distribution
uniform_probs = np.ones(6) / 6
uniform_rolls = np.random.choice(6, size=num_rolls, p=uniform_probs)

# Scenario 2: Biased probabilities
# Sample probabilities from a Dirichlet distribution
alpha = [1, 2, 3, 4, 5, 6]  # concentration parameters for Dirichlet distribution
biased_probs = np.random.dirichlet(alpha, size=1).flatten()
biased_rolls = np.random.choice(6, size=num_rolls, p=biased_probs)

# Plot the results
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].hist(uniform_rolls, bins=6, density=True, alpha=0.5, label='Uniform')
ax[0].hist(biased_rolls, bins=6, density=True, alpha=0.5, label='Biased')
ax[0].legend()
ax[0].set_title('Dice Rolls')

ax[1].bar(np.arange(6), uniform_probs, alpha=0.5, label='Uniform')
ax[1].bar(np.arange(6), biased_probs, alpha=0.5, label='Biased')
ax[1].legend()
ax[1].set_title('Categorical Probabilities')

plt.show()

# Scenario 1: Uniform probabilities
uniform_probs = np.array([1/6]*6)  # all dice rolls have equal probability
num_rolls = 1000  # number of dice rolls
uniform_rolls = np.random.choice(6, size=num_rolls, p=uniform_probs)

# Scenario 2: Biased probabilities
biased_probs = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])  # higher probability for rolling 6
biased_rolls = np.random.choice(6, size=num_rolls, p=biased_probs)

# Perform chi-squared test to compare scenarios
observed_freqs1, _ = np.histogram(uniform_rolls, bins=6)
observed_freqs2, _ = np.histogram(biased_rolls, bins=6)
expected_freqs1 = uniform_probs * num_rolls
expected_freqs2 = biased_probs * num_rolls

_, p_value = chisquare(observed_freqs1, expected_freqs1)
print(f"p-value for Scenario 1: {p_value:.3f}")

_, p_value = chisquare(observed_freqs2, expected_freqs2)
print(f"p-value for Scenario 2: {p_value:.3f}")
