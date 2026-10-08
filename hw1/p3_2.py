import numpy as np
import p3_1
import p2_2
import p1

# ---------------------------------------------------------------------------
# 任务 3(2)：局部直方图均衡化
# ---------------------------------------------------------------------------
def local_hist_equalization(img, r):
    """对每个像素，用其 (2r-1)x(2r-1) 邻域的直方图做均衡化，
    但只把映射结果写回该邻域的中心像素（邻域不重叠地“移动”处理）。
    """
    output = img.copy()
    # eff_local_hist 以增量方式依次给出每个中心点的局部直方图
    for i, j, H_local in p2_2.eff_local_hist(img, r):
        t = p3_1.histogram_equalization(H_local)    # 由局部直方图得到灰度映射表
        old_value = int(img[i][j])
        output[i][j] = t[old_value]                 # 只更新中心像素
    return output


def main():
    in_path = 'img/resource.jpg'
    out_path = 'img/output_3.jpg'
    img = p1.read_intensity(in_path)
    img_processed = local_hist_equalization(img, 9)     # 邻域半径 r = 9（17x17 窗口）
    p1.write_image(out_path, img_processed)


if __name__ == '__main__':
    main()