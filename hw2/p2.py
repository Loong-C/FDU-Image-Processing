import cv2
import numpy as np
LEVELS = np.arange(256, dtype=np.float64)
def otsu_from_hist(hist):
    """从 256 维灰度直方图中求 Otsu 阈值。"""
    hist = hist.astype(np.float64)
    n = hist.sum()
    counts = np.cumsum(hist)[:-1]       # 灰度 <= t 的像素数
    sums = np.cumsum(hist * LEVELS)[:-1]  # 灰度 <= t 的灰度总和
    total = np.sum(hist * LEVELS)

    valid = (counts > 0) & (counts < n)
    if not np.any(valid):
        return 0

    scores = np.full(255, -1.0)
    scores[valid] = (
        (n * sums[valid] - total * counts[valid]) ** 2
        / (counts[valid] * (n - counts[valid]))
    )
    return int(np.argmax(scores) + 1)


def local_otsu(img, block_size=31):
    """对每个像素的局部窗口求 Otsu 阈值，滑动更新窗口直方图。"""
    height, width = img.shape
    radius = block_size // 2
    padded = np.pad(img, radius, mode="reflect")
    output = np.empty_like(img)

    # 当前行的第一个窗口的直方图
    row_hist = np.bincount(
        padded[:block_size, :block_size].ravel(), minlength=256
    ).astype(np.int32)

    for y in range(height):
        if y > 0:
            # 窗口向下移动一格，删掉顶部一行，加入底部一行
            np.add.at(row_hist, padded[y - 1, :block_size], -1)
            np.add.at(row_hist, padded[y + block_size - 1, :block_size], 1)

        hist = row_hist.copy()
        for x in range(width):
            if x > 0:
                # 窗口向右移动一格，删掉左侧一列，加入右侧一列
                np.add.at(hist, padded[y:y + block_size, x - 1], -1)
                np.add.at(hist, padded[y:y + block_size, x + block_size - 1], 1)

            t = otsu_from_hist(hist)
            output[y, x] = 255 if img[y, x] >= t else 0

    return output


if __name__ == "__main__":

    img = cv2.imread("hw2/img/input.png", cv2.IMREAD_GRAYSCALE)
    hist = np.bincount(img.ravel(), minlength=256)
    local_result = local_otsu(img, block_size=31)
    cv2.imwrite("hw2/img/local_otsu_result.png", local_result)

