import math
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 任务 2(1)：n 维联合直方图
# ---------------------------------------------------------------------------
def n_dim_histogram(data, B):
    # Pi_B[i] = B[0]*...*B[i]，即前 i+1 维的 bin 总数；Pi_B[i-1] 是第 i 维的跨度
    Pi_B = [B[0]] * len(B)
    for i in range(1, len(B)):
        Pi_B[i] = B[i] * Pi_B[i - 1]

    l = len(data[0])                    # 数据维数 n
    min_x = [255] * l                   # 每一维的取值下界
    max_x = [0] * l                     # 每一维的取值上界
    for x in data:
        for i in range(l):
            if x[i] > max_x[i]:
                max_x[i] = x[i]
            if x[i] < min_x[i]:
                min_x[i] = x[i]
    width = [0] * l                     # 每一维单个 bin 的宽度
    H = [0] * Pi_B[-1]                  # 展平后的直方图，共 prod(B) 个 bin
    for i in range(l):
        width[i] = (max_x[i] - min_x[i]) / B[i]

    for x in data:
        index = 0                       # 把 n 维 bin 坐标压成一维下标
        for i in range(l):
            # 第 i 维的 bin 号，截断到 [0, B[i]-1]，防止落在区间外的样本越界
            k = min(
                max(
                    math.floor((x[i] - min_x[i]) / width[i]),
                    0
                ),
                B[i] - 1
            )
            if i == 0:
                index += k
            else:
                index += k * Pi_B[i - 1]        # 第 0 维作为最低位
        H[index] += 1                           # 该 n 维 bin 计数 +1
    return H


# ---------------------------------------------------------------------------
# 测试：构造二维数据 (x, y)，两维高度相关，联合直方图应呈对角线“脊”
# ---------------------------------------------------------------------------
np.random.seed(0)
N = 2000
x = np.random.uniform(0, 255, N)            # 第 1 维：0~255 均匀分布
noise = np.random.normal(0, 20, N)
y = x + noise                               # 第 2 维：x 叠加高斯噪声
y = np.clip(y, 0, 255)
data = np.column_stack((x, y)).tolist()     # 整理成 N x 2 的样本列表
B = [16, 16]                                # 每一维 16 个 bin
H = n_dim_histogram(data, B)

# 图 2-1：原始二维数据散点图
plt.figure(figsize=(7, 6))
plt.scatter(x, y, s=8, alpha=0.4)
plt.xlabel("Dimension 1: x")
plt.ylabel("Dimension 2: y")
plt.title("Original Two-Dimensional Data")
plt.xlim(0, 255)
plt.ylim(0, 255)
plt.grid()
plt.show()

# 图 2-2：二维联合直方图（第 0 维为列、第 1 维为行，与展平顺序一致）
H_2d = np.array(H).reshape(B[1], B[0])
plt.figure(figsize=(7, 6))
plt.imshow(
    H_2d,
    origin="lower",
    aspect="auto",
    extent=[
        min(x),
        max(x),
        min(y),
        max(y)
    ]
)
plt.xlabel("Dimension 1: x")
plt.ylabel("Dimension 2: y")
plt.title("2D Joint Histogram)")
plt.colorbar(label="Number of Samples")
plt.show()