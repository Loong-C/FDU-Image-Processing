import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def unif_white_noise(img, A):
    h, w = img.shape
    noise = np.random.uniform(-A, A, (h, w))
    noisy_img = img + noise
    noisy_img = np.clip(noisy_img, 0, 255)
    return noisy_img.astype(np.uint8)

def salt_pepper_noise(img, p):
    noisy_img = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            r = np.random.rand()
            if r < p / 2:
                noisy_img[i, j] = 0
            elif r < p:
                noisy_img[i, j] = 255
    return noisy_img

def main():
    heart_img = Image.open('img/heart.png').convert('L')
    brain_img = Image.open('img/brain.png').convert('L')
    heart_array = np.array(heart_img)
    brain_array = np.array(brain_img)
    white_noise_heart = unif_white_noise(heart_array, 50)
    white_noise_brain = unif_white_noise(brain_array, 50)
    salt_pepper_heart = salt_pepper_noise(heart_array, 0.05)
    salt_pepper_brain = salt_pepper_noise(brain_array, 0.05)
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    axes = axes.ravel()
    items = [
        (heart_array,          'Original Heart Image'),
        (brain_array,          'Original Brain Image'),
        (white_noise_heart,    'Heart with Uniform White Noise'),
        (white_noise_brain,    'Brain with Uniform White Noise'),
        (salt_pepper_heart,    'Heart with Salt & Pepper Noise'),
        (salt_pepper_brain,    'Brain with Salt & Pepper Noise'),
    ]
    for ax, (data, title) in zip(axes, items):
        ax.imshow(data, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()