import numpy as np
import cv2
import matplotlib.pyplot as plt

ksize = 41
sigma = 6.0
theta = np.deg2rad(90)   # change ça
lambd = 12.0
gamma = 0.5
psi = 0

k = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
k = k - k.mean()

plt.imshow(k, cmap="seismic")
plt.colorbar()
plt.title("Gabor kernel")
plt.savefig("gabor_kernel.png")