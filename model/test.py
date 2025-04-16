import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

config = {
    "font.family":"serif",
    "font.size":14,
    "mathtext.fontset":'stix',
    "font.serif":['SimSun']
}
rcParams.update(config)
# # # Defining constellation points
# bpsk_points = np.array([-1, 1])  # BPSK symbols on the real line
# qpsk_points = np.array([0.707+0.707j, -0.707+0.707j, -0.707-0.707j, 0.707-0.707j])  # QPSK symbols on the unit circle

# # Create figure and subplots
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # One row, two columns

# # Helper function to draw a circle
# def draw_circle(ax):
#     circle = plt.Circle((0, 0), 1, color='blue', fill=False, linestyle='dashed', linewidth=1.5)
#     ax.add_artist(circle)

# # Plotting BPSK
# ax1.scatter(np.real(bpsk_points), np.imag(bpsk_points), color='red', s=100, marker='x')
# ax1.set_xlabel('In-phase',size=14)
# ax1.set_ylabel('Quadrature',size=14)
# ax1.axhline(0, color='black', linewidth=0.5)
# ax1.axvline(0, color='black', linewidth=0.5)
# ax1.grid(True)
# ax1.axis('equal')
# ax1.set_xlim(-1.5, 1.5)
# ax1.set_ylim(-1.5, 1.5)
# draw_circle(ax1)
# ax1.text(0.5, -0.2, '(a) BPSK', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)

# # Plotting QPSK
# ax2.scatter(np.real(qpsk_points), np.imag(qpsk_points), color='red', s=100, marker='x')
# ax2.set_xlabel('In-phase')
# ax2.set_ylabel('Quadrature')
# ax2.axhline(0, color='black', linewidth=0.5)
# ax2.axvline(0, color='black', linewidth=0.5)
# ax2.grid(True)
# ax2.axis('equal')
# ax2.set_xlim(-1.5, 1.5)
# ax2.set_ylim(-1.5, 1.5)
# draw_circle(ax2)
# ax2.text(0.5, -0.2, '(b) QPSK', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)

# plt.tight_layout()
# plt.savefig('BQ_new.png',dpi=600, bbox_inches='tight')
# plt.show()



# # 星座图影响
# import numpy as np
# import matplotlib.pyplot as plt

# Define the theoretical QPSK points
qpsk_points = np.array([0.707+0.707j, -0.707+0.707j, -0.707-0.707j, 0.707-0.707j])
qpsk_points_gain = np.array([0.5+1j, -0.5+1j, -0.5-1j, 0.5-1j])
num_samples = 50  # Number of samples per impairment

# Generate impairments
np.random.seed(0)
noise_std = 0.1  # Standard deviation for white noise
phase_noise_std = np.pi / 16  # Phase noise standard deviation
gain_imbalance = 0.1  # Gain difference in dB
phase_imbalance = np.pi / 12  # Phase imbalance in radians

# Create the plots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs = axs.flatten()

# Function to replicate points
def replicate_points(points, num_replicas):
    return np.tile(points, (num_replicas, 1)).flatten()

# (a) White Noise
noisy_points = qpsk_points[:, None] + noise_std * (np.random.randn(len(qpsk_points), num_samples) + 1j * np.random.randn(len(qpsk_points), num_samples))
noisy_points = noisy_points.flatten()
axs[0].scatter(np.real(noisy_points), np.imag(noisy_points), color='blue', marker='.', alpha=0.5)
axs[0].scatter(np.real(qpsk_points), np.imag(qpsk_points), color='red', marker='x')
axs[0].set_title('(a) 白噪声', fontproperties="SimHei")
axs[0].grid(True)

# (b) Phase Noise
phase_noise = np.random.normal(0, phase_noise_std, (len(qpsk_points), num_samples))
phase_noisy_points = np.abs(qpsk_points[:, None]) * np.exp(1j * (np.angle(qpsk_points[:, None]) + phase_noise))
phase_noisy_points = phase_noisy_points.flatten()
axs[1].scatter(np.real(phase_noisy_points), np.imag(phase_noisy_points), color='blue', marker='.', alpha=0.5)
axs[1].scatter(np.real(qpsk_points), np.imag(qpsk_points), color='red', marker='x')
axs[1].set_title('(b) 相位噪声', fontproperties="SimHei")
axs[1].grid(True)
# Setup for noise and other impairments

# Setup for noise and other impairments
np.random.seed(0)
gain_factors = (1.2, 0.8)  # Gain factors for I (stretch) and Q (compress) respectively

# (c) IQ Gain Imbalance
# Apply gain imbalance to generate noisy points around the original location
iq_gain_noisy = qpsk_points_gain[:, None] + noise_std * (np.random.randn(len(qpsk_points), num_samples) + 1j * np.random.randn(len(qpsk_points), num_samples))
iq_gain_noisy = iq_gain_noisy.flatten()
axs[2].scatter(np.real(iq_gain_noisy), np.imag(iq_gain_noisy), color='blue', marker='.', alpha=0.5)
axs[2].scatter(np.real(qpsk_points), np.imag(qpsk_points), color='red', marker='x')
axs[2].set_title('(c) IQ增益不平衡', fontproperties="SimHei")
axs[2].grid(True)


# (d) IQ Phase Imbalance
iq_phase_points = np.real(qpsk_points) * np.exp(1j * phase_imbalance) + 1j * np.imag(qpsk_points) * np.exp(-1j * phase_imbalance)
iq_phase_points = iq_phase_points[:, None] + noise_std * (np.random.randn(len(qpsk_points), num_samples) + 1j * np.random.randn(len(qpsk_points), num_samples))
iq_phase_points = iq_phase_points.flatten()
axs[3].scatter(np.real(iq_phase_points), np.imag(iq_phase_points), color='blue', marker='.', alpha=0.5)
axs[3].scatter(np.real(qpsk_points), np.imag(qpsk_points), color='red', marker='x')
axs[3].set_title('(d) IQ相位不平衡', fontproperties="SimHei")
axs[3].grid(True)

# Adjust plot settings
for ax in axs:
    ax.axis('equal')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('In-Phase')
    ax.set_ylabel('Quadrature')

# Show the plot
plt.tight_layout()
plt.savefig('./img/influence_new.png',dpi=600, bbox_inches='tight')
plt.show()


