import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import shape, MultiPolygon
from data.trajectoryData import *


def plotMap(plt, map):
    for areaItem in map:
        areaItem = np.array(areaItem)
        plt.plot(areaItem[:, 0], areaItem[:, 1], 'k-', linewidth='1')
def plot_geo_map(plt, geo_json, building_color, aisle_color):
    feature = geo_json['features'][0]
    geometry = feature['geometry']
    # 将坐标转换为 Shapely 的 Polygon 对象
    polygon = shape(geometry)
    polygon = polygon.buffer(0)
    for geom in polygon.geoms if isinstance(polygon, MultiPolygon) else [polygon]:
        # 绘制外环（外部边界）
        x, y = polygon.exterior.xy
        plt.fill(x, y, alpha=0.5, fc=aisle_color, edgecolor='black')

        # 绘制内环（如果有）
        for interior in polygon.interiors:
            x, y = interior.xy
            plt.fill(x, y, alpha=0.5, fc=building_color, edgecolor='none')

def plotTrajectory(plt, trajectories):
    for tra in trajectories:
        tra = np.array(tra)
        plt.plot(tra[:, 0], tra[:, 1], '-', linewidth='2', color='violet', marker='o')

def getAllPoints(trajectories):
    return np.vstack(trajectories)

def get_range(aisle):
    min_lat = 90
    max_lat = -90
    min_lon = 180
    max_lon =-180
    for line in aisle:
        for point in line:
            min_lat, min_lon, max_lat, max_lon = min(point[1], min_lat),\
                                                 min(point[0], min_lon),\
                                                 max(point[1], max_lat),\
                                                 max(point[0], max_lon)
    return (min_lat, max_lat, min_lon, max_lon)

def plot_heatmap(plt, grid_width, grid_height, color_map, trajectories, range):
    """
    绘制网格热力图。

    参数：
    - grid_width: 横向网格数
    - grid_height: 纵向网格数
    - color_map: 颜色选择
    - trajectories: 轨迹列表，每个轨迹为一个点列表，形式：[[x1, y1], [x2, y2], ...]
    """
    # 创建一个零矩阵来记录每个网格经过的次数
    grid_counts = np.zeros((grid_height, grid_width))

    x_min, x_max = range[2], range[3]
    y_min, y_max = range[0], range[1]

    # 计算每个轨迹点属于哪个网格
    for traj in trajectories:
        for point in traj:
            x, y = point

            # 计算点在网格中的位置
            col = int((x - x_min) / (x_max - x_min) * grid_width)
            row = int((y - y_min) / (y_max - y_min) * grid_height)

            # 确保列和行不超出边界
            col = min(max(col, 0), grid_width - 1)
            row = min(max(row, 0), grid_height - 1)

            # 增加对应网格的计数
            grid_counts[row, col] += 1

    # 绘制热力图
    plt.imshow(grid_counts, cmap=color_map, origin='lower', interpolation='nearest')
    plt.tick_params(axis='both', labelsize=18)
    cbar = plt.colorbar(label='Grid Count', fraction=0.05)
    # 设置 colorbar 标签字体大小
    cbar.set_label('Grid Count', fontsize=18)
    # 设置 colorbar 刻度标签字体大小
    cbar.ax.tick_params(labelsize=18)
    # plt.xlabel('Grid X')
    # plt.ylabel('Grid Y')
    return grid_counts

def plot_sim_heapmap(plt, real_grid_counts, sim_grid_counts, cmap):
    # 计算两者的相似性（例如，计算每个网格的差异）
    similarity = np.abs(real_grid_counts - sim_grid_counts)
    print(similarity)
    # 相似性热力图
    plt.imshow(similarity, cmap='YlGnBu', origin='lower', interpolation='nearest')
    # plt.set_title('Trajectory Similarity Heatmap')
    plt.colorbar(label='Difference', fraction=0.05)

# plot_heatmap(plt, 25, 25, color_map='viridis', trajectories=trajectory, range = get_range(area))
# plt.show()

