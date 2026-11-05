import numpy as np
import cv2
import matplotlib.pyplot as plt
def fourier_transform(img):
    return np.fft.fftshift(np.fft.fft2(img))
def inverse_fourier_transform(img):
    return np.fft.ifft2(np.fft.ifftshift(img)).real
def low_frequency_filter(img_freq, threshold):
    h, w = img_freq.shape
    img_filtered = np.zeros_like(img_freq)
    for i in range(h):
        for j in range(w):
            if np.sqrt((i - h//2)**2 + (j - w//2)**2) < threshold:
                img_filtered[i, j] = img_freq[i, j]
    return img_filtered

def sharpen_filter(img_freq, threshold=40, k=1.5):
    h, w = img_freq.shape
    img_filtered = np.zeros_like(img_freq)
    for i in range(h):
        for j in range(w):
            distance = np.sqrt(
                (i - h//2)**2 + (j - w//2)**2
            )
            if distance < threshold:
                img_filtered[i, j] = img_freq[i, j]
            else:
                img_filtered[i, j] = (1 + k) * img_freq[i, j]

    return img_filtered

def low_frequency_smoothing(img, threshold=40):
    h, w = img.shape
    padded = np.zeros((2 * h, 2 * w), dtype=np.float64)
    padded[:h, :w] = img
    img_freq = fourier_transform(padded)
    filtered_freq = low_frequency_filter(img_freq, threshold)
    filtered_padded = inverse_fourier_transform(filtered_freq)
    result = filtered_padded[:h, :w]
    return result, img_freq, filtered_freq

def shapening(img):
    h, w = img.shape
    padded = np.zeros((2 * h, 2 * w), dtype=np.float64)
    padded[:h, :w] = img
    img_freq = fourier_transform(padded)
    filtered_freq = sharpen_filter(img_freq, 40)
    filtered_padded = inverse_fourier_transform(filtered_freq)
    result = filtered_padded[:h, :w]
    return result, img_freq, filtered_freq

def main():
    img = cv2.imread('img/input.png', cv2.IMREAD_GRAYSCALE)
    smoothed, img_freq, filtered_freq = low_frequency_smoothing(img, threshold=40)
    sharpened, img_freq_sharp, filtered_freq_sharp = shapening(img)
    cv2.imwrite(
        'img/lena_smoothed.bmp',
        np.clip(smoothed, 0, 255).astype(np.uint8)
    )

    cv2.imwrite(
        'img/lena_sharpened.bmp',
        np.clip(sharpened, 0, 255).astype(np.uint8)
    )
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    results = [
        ("Smoothing", smoothed, img_freq, filtered_freq),
        ("Sharpening", sharpened, img_freq_sharp, filtered_freq_sharp)
    ]

    for row, (name, result, F, G) in enumerate(results):
        axes[row, 0].imshow(img, cmap="gray")
        axes[row, 0].set_title("Original Image")

        axes[row, 1].imshow(np.log1p(np.abs(F)), cmap="gray")
        axes[row, 1].set_title("Original Spectrum")

        axes[row, 2].imshow(np.log1p(np.abs(G)), cmap="gray")
        axes[row, 2].set_title("Filtered Spectrum")

        axes[row, 3].imshow(np.clip(result, 0, 255), cmap="gray")
        axes[row, 3].set_title(name)

        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()