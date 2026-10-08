import cv2
import numpy as np

def bilinear_resize(img, N):
    h, w = img.shape[:2]
    new_h, new_w = h * N, w * N
    output = np.zeros(
        (new_h, new_w) + img.shape[2:],
        dtype=np.uint8
    )

    for y in range(new_h):
        for x in range(new_w):
            center_x = (x + 0.5) / N - 0.5
            center_y = (y + 0.5) / N - 0.5
            y0 = int(np.floor(center_y))
            x0 = int(np.floor(center_x))
            y1 = min(y0 + 1, h - 1)
            x1 = min(x0 + 1, w - 1)
            u = center_x - x0
            v = center_y - y0
            intensity = (1-u)*(1-v)*img[y0, x0] + \
                        u*(1-v)*img[y0, x1] + \
                        (1-u)*v*img[y1, x0] + \
                        u*v*img[y1, x1]
            output[y, x] = np.rint(intensity).astype(np.uint8)

    return output


if __name__ == "__main__":
    img = cv2.imread("hw2/img/input.png")

    N = 3
    result = bilinear_resize(img, N)

    cv2.imwrite("hw2/img/enlarged.png", result)

