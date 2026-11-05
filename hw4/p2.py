from p1 import fourier_transform, inverse_fourier_transform
import ast
import numpy as np
import cv2
with open('points.txt', 'r') as f:
    points = ast.literal_eval(f.read())

def notch_filter(img_freq, points, sigma=10):
    h, w = img_freq.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    mask = np.ones((h, w))
    for dy, dx in points:
        for sign in [-1, 1]:
            py = cy + sign * dy
            px = cx + sign * dx
            distance_y = np.abs(y - py)
            distance_x = np.abs(x - px)
            distance_y = np.minimum(distance_y, h - distance_y)
            distance_x = np.minimum(distance_x, w - distance_x)
            D2 = distance_y**2 + distance_x**2
            mask *= 1 - np.exp(-D2 / (2 * sigma**2))
    return img_freq * mask, mask

def selective_filter(img, points, sigma=10):
    img_freq = fourier_transform(img)
    filtered_freq, mask = notch_filter(img_freq, points, sigma)
    result = inverse_fourier_transform(filtered_freq)
    return result, img_freq, filtered_freq, mask

def main():
    img = cv2.imread('img/input.png', cv2.IMREAD_GRAYSCALE)
    result, F, G, mask = selective_filter(img, points, sigma=10)
    cv2.imwrite(
        'img/denoised.png',
        np.clip(result, 0, 255).astype(np.uint8)
    )

if __name__ == '__main__':
    main()