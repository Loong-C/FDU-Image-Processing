import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, maximum_filter
img = cv2.imread('img/input.png', cv2.IMREAD_GRAYSCALE)
F = np.fft.fftshift(np.fft.fft2(img))
S = np.log1p(np.abs(F))
contrast = S - gaussian_filter(S, 6)
h, w = img.shape
cy, cx = h // 2, w // 2
y, x = np.ogrid[:h, :w]

contrast[(y-cy)**2 + (x-cx)**2 < 25**2] = -np.inf
contrast[(abs(y-cy) < 3) | (abs(x-cx) < 3)] = -np.inf

peaks = np.argwhere(
    (contrast == maximum_filter(contrast, size=9))
    & (contrast > 2.6)
)

points = [(int(i-cy), int(j-cx))
          for i, j in peaks if i > cy or (i == cy and j > cx)]

print(points)