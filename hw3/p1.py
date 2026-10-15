import matplotlib.pyplot as plt
import numpy as np
import cv2


def part_convolution(patch, kernel):
    result = 0.0
    h, w = patch.shape
    for i in range(h):
        for j in range(w):
            result += patch[i, j] * kernel[i, j]
    return result


def padding(img, kernel_shape):
    kh, kw = kernel_shape
    return np.pad(img, ((kh // 2, kh // 2), (kw // 2, kw // 2)), mode="reflect")


def img_convolution(img, kernel):
    img = np.asarray(img, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    kh, kw = kernel.shape
    padded = padding(img, kernel.shape)
    rotated_kernel = kernel[::-1, ::-1]  # convolution rather than correlation
    h, w = img.shape
    result = np.zeros((h, w), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            result += rotated_kernel[i, j] * padded[i:i + h, j:j + w]
    return result


def avg_smoothing(img, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float64)
    kernel /= kernel.sum()
    return img_convolution(img, kernel)


def gaussian_smoothing(img, kernel_size, sigma):
    radius = kernel_size // 2
    axis = np.arange(-radius, radius + 1, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return img_convolution(img, kernel)


def laplacian_filter(img):
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float64)
    return img_convolution(img, kernel)


def laplacian_sharpen(img, alpha=0.5):
    img = np.asarray(img, dtype=np.float64)
    return img - alpha * laplacian_filter(img)


def unsharp_masking(img, kernel_size=5, sigma=1.2, alpha=1.0):
    img = np.asarray(img, dtype=np.float64)
    blurred = gaussian_smoothing(img, kernel_size, sigma)
    mask = img - blurred
    return img + alpha * mask

def to_uint8(img):
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)

def main():
    img = cv2.imread("hw3/images/input.jpg", cv2.IMREAD_GRAYSCALE)
    img = np.asarray(img, dtype=np.float64)
    avg_smoothed = avg_smoothing(img, 15)
    gaussian_smoothed = gaussian_smoothing(img, 31, 5.0)
    laplacian_sharpened = laplacian_sharpen(img, 1.0)
    unsharp_masked = unsharp_masking(img, 31, 5.0, 2.0)
    cv2.imwrite("hw3/images/input_greyscale.png", to_uint8(img))
    cv2.imwrite("hw3/images/output1_avg_smoothed.png", to_uint8(avg_smoothed))
    cv2.imwrite("hw3/images/output1_gaussian_smoothed.png", to_uint8(gaussian_smoothed))
    cv2.imwrite("hw3/images/output1_laplacian_sharpened.png", to_uint8(laplacian_sharpened))
    cv2.imwrite("hw3/images/output1_unsharp_masked.png", to_uint8(unsharp_masked))

if __name__ == "__main__":
    main()