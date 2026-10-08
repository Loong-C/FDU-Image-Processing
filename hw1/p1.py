import cv2

# ---------------------------------------------------------------------------
# 任务 1：分段线性变换（对比度拉伸）
# ---------------------------------------------------------------------------
def pixel_process(in_, L=256, r1=32, s1=16, r2=64, s2=128):
    """把单像素灰度 in_ 按折线 (0,0)-(r1,s1)-(r2,s2)-(L-1,L-1) 映射为新灰度。
    """
    in_ = int(in_)
    out_ = 0
    if 0 <= in_ < r1:                                   # 第 1 段：暗区，斜率 s1/r1 = 0.5
        out_ = in_ * s1 / r1
    elif r1 <= in_ < r2:                                # 第 2 段：中间灰度，斜率 (s2-s1)/(r2-r1) = 3.5（拉伸最强）
        out_ = s1 + (in_ - r1) * ((s2 - s1) / (r2 - r1))
    elif r2 <= in_ <= L - 1:                            # 第 3 段：亮区，斜率 (L-1-s2)/(L-1-r2) ≈ 0.665
        out_ = s2 + (in_ - r2) * ((L - 1 - s2) / (L - 1 - r2))
    return int(out_)                                    # int() 截断取整，输出仍落在 0~255


def read_intensity(in_path):
    """读入图像并转为单通道灰度图（uint8，取值 0~255）。"""
    img_gray = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
    return img_gray


def img_process(img_gray):
    """逐像素调用 pixel_process，完成整幅图像的灰度映射。"""
    h, w = img_gray.shape
    for i in range(h):
        for j in range(w):
            img_gray[i, j] = pixel_process(img_gray[i, j])
    return img_gray


def write_image(out_path, img_processed):
    """保存结果图像。"""
    cv2.imwrite(out_path, img_processed)


def main():
    in_path = 'img/resource.jpg'
    out_path = 'img/output_1.jpg'
    img_gray = read_intensity(in_path)
    img_processed = img_process(img_gray)           # 对比度拉伸
    write_image(out_path, img_processed)


if __name__ == '__main__':
    main()