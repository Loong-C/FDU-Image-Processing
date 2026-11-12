import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def add_periodic_noise(image):
    height, width = image.shape
    y, x = np.indices(image.shape)

    frequency_x = round(width / 14)
    frequency_y = round(height / 20)

    amplitude = 30

    phase = 2 * np.pi * (
        frequency_x * x / width +
        frequency_y * y / height
    )

    noise = amplitude * np.sin(phase)
    noisy_image = image.astype(float) + noise

    return noisy_image, frequency_y, frequency_x

def notch_pass_filter(shape, fx, fy, radius=1):
    height, width = shape
    y, x = np.indices(shape)
    center_y = height // 2
    center_x = width // 2
    x1 = center_x + fx
    y1 = center_y + fy
    x2 = center_x - fx
    y2 = center_y - fy
    distance1 = (x - x1)**2 + (y - y1)**2
    distance2 = (x - x2)**2 + (y - y2)**2
    mask = (distance1 <= radius**2) | (distance2 <= radius**2)
    return mask.astype(float)

def optimal_weights(noisy_image, estimated_noise, window_size=61):
    height, width = noisy_image.shape
    radius = window_size // 2
    g = np.pad(noisy_image, radius, mode="reflect")
    n = np.pad(estimated_noise, radius, mode="reflect")
    weights = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            g_patch = g[y:y + window_size, x:x + window_size]
            n_patch = n[y:y + window_size, x:x + window_size]
            g_mean = np.mean(g_patch)
            n_mean = np.mean(n_patch)
            covariance = np.mean(
                (g_patch - g_mean) * (n_patch - n_mean)
            )
            variance = np.mean((n_patch - n_mean) ** 2)
            weights[y, x] = covariance / variance
    return weights

def main():
    img = Image.open('img/brain.png').convert('L')
    img_array = np.array(img)
    noisy_image, fy, fx = add_periodic_noise(img_array)
    spectrum = np.fft.fftshift(np.fft.fft2(noisy_image))
    H_np = notch_pass_filter(noisy_image.shape, fx, fy)
    noise_spectrum = H_np * spectrum
    estimated_noise = np.fft.ifft2(
        np.fft.ifftshift(noise_spectrum)
    ).real
    weights = optimal_weights(noisy_image, estimated_noise)
    restored_image = noisy_image - weights * estimated_noise
    restored_image = np.clip(restored_image, 0, 255)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_array, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original')
    axes[1].imshow(noisy_image, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('Noisy')
    axes[2].imshow(restored_image, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title('Restored')
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()



