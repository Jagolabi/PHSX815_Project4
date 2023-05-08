import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare

# Set up the simulation parameters
g = 9.81  # acceleration due to gravity (m/s^2)
dt = 0.01  # time step (s)
num_steps = 1000  # number of time steps to simulate

# Define a function to simulate the motion of a pendulum
def simulate_pendulum(l):
    # Define the initial conditions
    theta0 = np.deg2rad(45)  # initial angle (radians)
    omega0 = 0  # initial angular velocity (radians/s)

    # Initialize arrays to store the position and velocity at each time step
    theta = np.zeros(num_steps)
    omega = np.zeros(num_steps)

    # Set the initial conditions
    theta[0] = theta0
    omega[0] = omega0

    # Simulate the motion of the pendulum
    for i in range(1, num_steps):
        # Update the angular velocity
        omega[i] = omega[i-1] - (g / l) * np.sin(theta[i-1]) * dt

        # Update the angle
        theta[i] = theta[i-1] + omega[i] * dt

    return theta, omega

# Set up the initial conditions
l1 = 1  # length of pendulum 1 (m)
l2 = 2  # length of pendulum 2 (m)

# Simulate the motion of the pendulum for each length
theta1, omega1 = simulate_pendulum(l1)
theta2, omega2 = simulate_pendulum(l2)

# Plot the results
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(theta1, omega1, label=f'l = {l1:.1f} m')
ax[0].plot(theta2, omega2, label=f'l = {l2:.1f} m')
ax[0].set_xlabel('Angle (radians)')
ax[0].set_ylabel('Angular velocity (radians/s)')
ax[0].set_title('Pendulum Motion')
ax[0].legend()

ax[1].hist(theta1, bins=20, density=True, alpha=0.5, label=f'l = {l1:.1f} m')
ax[1].hist(theta2, bins=20, density=True, alpha=0.5, label=f'l = {l2:.1f} m')
ax[1].set_xlabel('Angle (radians)')
ax[1].set_ylabel('Probability density')
ax[1].set_title('Angle Distribution')
ax[1].legend()

plt.show()

# Divide the range of angles into bins
num_bins = 20
bins = np.linspace(-np.pi, np.pi, num_bins + 1)

# Count the number of observed angles in each bin
observed1, _ = np.histogram(theta1, bins=bins)
observed2, _ = np.histogram(theta2, bins=bins)

# Compute the expected frequencies assuming a uniform distribution
expected = np.ones(num_bins) * len(theta1) / num_bins

# Perform the chi-squared test
statistic, p_value = chisquare(observed1, expected)
print(f'Chi-squared test for l = {l1:.1f} m: statistic={statistic:.3f}, p={p_value:.3f}')

statistic, p_value = chisquare(observed2, expected)
print(f'Chi-squared test for l = {l2:.1f} m: statistic={statistic:.3f}, p={p_value:.3f}')
