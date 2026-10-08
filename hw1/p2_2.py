# ---------------------------------------------------------------------------
# 任务 2(2)：局部直方图的高效（增量）更新
# ---------------------------------------------------------------------------
def eff_local_hist(img, r):
    """以 (i, j) 为中心、边长 2r-1 的方形邻域，逐个中心点生成局部直方图。

    生成器依次 yield (i, j, H_local)，i/j 遍历所有“窗口完整落在图内”的中心点。
    相邻邻域只差一行或一列，因此只需在旧直方图上做增减即可，避免重复统计。
    """
    h, w = img.shape
    i_start = r - 1          # 合法中心点范围：保证 2r-1 大小的窗口不越界
    i_end = h - r
    j_start = r - 1
    j_end = w - r
    i = i_start
    j = j_start
    H_local = [0] * 256
    # 统计第一个邻域的直方图（唯一的 O((2r-1)^2) 初始化）
    for x in range(i-r+1, i+r):
        for y in range(j-r+1, j+r):
            H_local[img[x][y]] += 1
    yield i, j, H_local.copy()

    # 蛇形（boustrophedon）扫描：行方向奇偶交替，
    # 这样换行时无需重新统计，只需“移出上一行、移入下一行”
    for i in range(i_start, i_end + 1):
        if i != i_start:
            # 垂直移动一格：窗口上边移出、下边移入
            out_row = i - r
            in_row = i + r - 1
            for y in range(j-r+1, j+r):
                H_local[img[out_row][y]] -= 1
                H_local[img[in_row][y]] += 1
            yield i, j, H_local.copy()
        if (i - i_start) % 2 == 0:
            # 偶数行：自左向右，水平移动一格 = 移出左列、移入右列
            for new_j in range(j + 1, j_end + 1):
                out_col = new_j - r
                in_col = new_j + r - 1
                for x in range(i-r+1, i+r):
                    H_local[img[x][out_col]] -= 1
                    H_local[img[x][in_col]] += 1
                j = new_j
                yield i, j, H_local.copy()
        else:
            # 奇数行：自右向左，方向相反
            for new_j in range(j - 1, j_start - 1, -1):
                out_col = new_j + r
                in_col = new_j - r + 1
                for x in range(i-r+1, i+r):
                    H_local[img[x][out_col]] -= 1
                    H_local[img[x][in_col]] += 1
                j = new_j
                yield i, j, H_local.copy()