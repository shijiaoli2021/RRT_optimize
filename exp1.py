import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.interpolate import interp1d
from data.trajectoryData import *
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing
from shapely.affinity import scale
from scipy.spatial import ConvexHull
import alphashape
import model.rrt as rrt
import model.rrt_point as rrt_point
from scipy.integrate import simps
from shapely.geometry import shape, MultiPolygon


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

def plotAlphaShape(plt, trajectories):
    # 计算 Alpha 形状
    all_points = getAllPoints(trajectories)
    alpha_shape = alphashape.alphashape(all_points, alpha=0.5)
    for trajectory in trajectories:
        trajectory = np.array(trajectory)
        plt.plot(trajectory[:, 0], trajectory[:, 1], label="轨迹")
    boundary_points = np.array(alpha_shape.exterior.coords)

    def smooth_boundary(points, num_points=200):
        x, y = points[:, 0], points[:, 1]
        t = np.linspace(0, 1, len(x))
        t_new = np.linspace(0, 1, num_points)
        x_smooth = make_interp_spline(t, x)(t_new)
        y_smooth = make_interp_spline(t, y)(t_new)
        return np.column_stack([x_smooth, y_smooth])

    smoothed_boundary = smooth_boundary(boundary_points)
    # Alpha 形状
    plt.plot(boundary_points[:, 0], boundary_points[:, 1], label="Alpha 包络", color="blue")

    # 平滑包络
    plt.plot(smoothed_boundary[:, 0], smoothed_boundary[:, 1], label="平滑包络", color="red")
    # plt.fill(*zip(*alpha_shape.exterior.coords), alpha=0.3, label="Alpha 包络")

def plotTuBao(plt, trajectories):
    # 收集所有点
    all_points = getAllPoints(trajectories)

    # 计算凸包
    hull = ConvexHull(all_points)
    hull_points = all_points[hull.vertices]

    # 创建 Shapely 多边形并扩展
    polygon = Polygon(hull_points)
    expanded_polygon = scale(polygon, xfact=1.1, yfact=1.1)  # 比例扩展
    # 绘制原始轨迹
    for traj in trajectories:
        traj = np.array(traj)
        plt.plot(traj[:, 0], traj[:, 1], label="轨迹")
    # 绘制凸包
    plt.fill(*zip(*hull_points), alpha=0.3, label="凸包")

def plotEnvelope(plt, trajectories):

    all_points = np.array(np.vstack(trajectory))
    # 插值点数和横坐标范围
    x_min, x_max = min(all_points[:, 0]), max(all_points[:, 0])
    trajectories = [np.array(tra) for tra in trajectories]
    print("x_min:{}, x_max:{}".format(x_min, x_max))
    num_points = 30
    x_interp = np.linspace(x_min, x_max, num_points)  # 插值横坐标

    # 存储所有轨迹在插值点的纵坐标
    interpolated_ys = []

    for traj in trajectories:
        # 对轨迹点进行线性插值
        f = interp1d(traj[:, 0], traj[:, 1], kind='linear', bounds_error=False)
        interpolated_ys.append(f(x_interp))  # 计算每个插值点的纵坐标

    # 转换为数组，方便计算
    interpolated_ys = np.array(interpolated_ys)

    # 计算每个插值点的上下界
    y_min = np.min(interpolated_ys, axis=0)
    y_max = np.max(interpolated_ys, axis=0)

    # for traj in trajectories:
    #     plt.plot(traj[:, 0], traj[:, 1], label="Trajectory")  # 绘制轨迹

    plt.fill_between(x_interp, y_min, y_max, color='gray', alpha=0.5, label="Envelope")  # 绘制包络线
    plt.plot(x_interp, y_min, 'r--', label="Lower Bound")  # 下界
    plt.plot(x_interp, y_max, 'b--', label="Upper Bound")  # 上界
    # # 绘制扩展包络
    # x, y = expanded_polygon.exterior.xy
    # plt.plot(x, y, color="red", label="扩展包络")


def plotEnvelope_forY(plt, trajectories):

    all_points = np.array(np.vstack(trajectory))
    # 插值点数和横坐标范围
    y_min, y_max = min(all_points[:, 1]), max(all_points[:, 1])
    trajectories = [np.array(tra) for tra in trajectories]
    print("x_min:{}, x_max:{}".format(y_min, y_max))
    num_points = 300
    y_interp = np.linspace(y_min, y_max, num_points)  # 插值横坐标

    # 存储所有轨迹在插值点的纵坐标
    interpolated_xs = []

    for traj in trajectories:
        # 对轨迹点进行线性插值
        f = interp1d(traj[:, 1], traj[:, 0], kind='linear', bounds_error=False)
        interpolated_xs.append(f(y_interp))  # 计算每个插值点的纵坐标

    # 转换为数组，方便计算
    interpolated_xs = np.array(interpolated_xs)

    # 计算每个插值点的上下界
    x_min = np.min(interpolated_xs, axis=0)
    x_max = np.max(interpolated_xs, axis=0)

    # for traj in trajectories:
    #     plt.plot(traj[:, 0], traj[:, 1], label="Trajectory")  # 绘制轨迹

    plt.fill_betweenx(y_interp, x_min, x_max, color='gray', alpha=0.5, label="Envelope")  # 绘制包络线
    plt.plot(x_min, y_interp, 'r--', label="Lower Bound")  # 下界
    plt.plot(x_max, y_interp, 'b--', label="Upper Bound")

def plotEnvelope2(plt, trajectories, fill_color, zorder):

    all_points = np.array(np.vstack(trajectory))
    # 插值点数和横坐标范围
    x_min, x_max = min(all_points[:, 0]), max(all_points[:, 0])
    trajectories = [np.array(tra) for tra in trajectories]
    print("x_min:{}, x_max:{}".format(x_min, x_max))
    num_points = 30
    x_interp = np.linspace(x_min, x_max, num_points)  # 插值横坐标

    # 存储所有轨迹在插值点的纵坐标
    interpolated_ys = [[100] * len(x_interp), [-100] * len(x_interp)]
    for traj in trajectories:
        for i in range(len(traj)-1):
            f = interp1d(traj[i:i+2, 0], traj[i:i+2, 1], kind='linear', bounds_error=False)
            # 适配y
            for i in range(len(x_interp)):
                cash_y = f(x_interp[i])
                if cash_y is not None:
                    interpolated_ys[0][i] = min(interpolated_ys[0][i], cash_y)
                    interpolated_ys[1][i] = max(interpolated_ys[1][i], cash_y)

    # 转换为数组，方便计算
    interpolated_ys = np.array(interpolated_ys)
    # 过滤无效点
    mask = np.isin(interpolated_ys, [-100, 100])

    interpolated_ys = interpolated_ys[:, ~mask.any(axis=0)]
    x_interp = x_interp[~mask.any(axis=0)]
    print(interpolated_ys)

    for traj in trajectories:
        plt.plot(traj[:, 0], traj[:, 1], label="Trajectory", color=fill_color)  # 绘制轨迹

    plt.fill_between(x_interp, interpolated_ys[0], interpolated_ys[1], color=fill_color, alpha=0.5, label="Envelope", zorder= zorder)  # 绘制包络线
    # plt.plot(x_interp, interpolated_ys[0], 'r--', label="Lower Bound")  # 下界
    # plt.plot(x_interp, interpolated_ys[1], 'b--', label="Upper Bound")  # 上界
    return x_interp, interpolated_ys
    # # 绘制扩展包络
    # x, y = expanded_polygon.exterior.xy
    # plt.plot(x, y, color="red", label="扩展包络")
def cal_intersection_area(x1, y_min1, y_max1, x2, y_min2, y_max2):
    x_min, x_max = min(x1[0], x2[0]), max(x1[-1], x2[-1])
    f1_min = interp1d(x1, y_min1, kind='linear', bounds_error=False)
    f1_max = interp1d(x1, y_max1, kind='linear', bounds_error=False)
    f2_min = interp1d(x2, y_min2, kind='linear', bounds_error=False)
    f2_max = interp1d(x2, y_max2, kind='linear', bounds_error=False)
    x_interp = np.linspace(x_min, x_max, num=200)
    y_common_min = np.maximum(f1_min(x_interp), f2_min(x_interp))
    y_common_max = np.minimum(f1_max(x_interp), f2_max(x_interp))
    return cal_area(x_interp, y_common_min, y_common_max)


def cal_area(x, y_min, y_max):
    return np.abs(simps(y_max - y_min, x))

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
    return min_lat, max_lat, min_lon, max_lon

def generate_rrt_routes(startPositions, endPositions, aisle):
    latitude_min, latitude_max, longitude_min, longitude_max = get_range(aisle)
    rrt_args = {
        "startPosition": startPositions[0],
        "endPosition": endPositions[0],
        "terminalDis": 6,
        "aisle": area,
        "aisle_json": geo_json,
        "lonRange": [longitude_min, longitude_max],
        "latRange": [latitude_min, latitude_max],
        "step": 6,
        "is_multi_polygon": True
    }
    rrt_routes = []
    rrt_model = rrt.RRT(**rrt_args)
    # rrt_model.plotMap_multi_polygon()
    rrt_model.exploreRoute()
    rrt_model.buildRoute()
    rrt_routes.append([[point.lon, point.lat] for point in rrt_model.routes])
    for i in range(1, len(startPoints)):
        print("已生成第{}条轨迹，该条轨迹共产生随机点数：{}".format(i, len(rrt_model.tree)))
        rrt_args["startPosition"] = startPositions[i]
        rrt_args["endPosition"] = endPositions[i]
        rrt_model.init_args(**rrt_args)
        rrt_model.exploreRoute()
        rrt_model.buildRoute()
        rrt_routes.append([[point.lon, point.lat] for point in rrt_model.routes])
    return rrt_routes

if __name__ == '__main__':
    startPoints = [trac[0] for trac in trajectory]
    endPoints = [trac[-1] for trac in trajectory]
    rrt_routes = generate_rrt_routes(startPoints, endPoints, area)
    fig = plt.figure()
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.subplot(1, 3, 1)
    # plotMap(plt, area)
    # plotEnvelope(plt, trajectory)
    # plt.subplot(1, 3, 2)
    # plotMap(plt, area)
    # plotTuBao(plt, trajectory)
    plt.subplot(111)
    # print(rrt_routes)
    plot_geo_map(plt, geo_json, 'lightgreen', 'lightblue')
    rrt_routes = np.array(rrt_routes)
    # plt.plot(rrt_routes[0][:, 0], rrt_routes[0][:, 1])
    x_true, interpolated_ys_true = plotEnvelope2(plt, trajectory, '#862adc', 1)
    x_sim, interpolated_ys_sim = plotEnvelope2(plt, rrt_routes, '#f59a23', 2)
    area_true = cal_area(x_true, interpolated_ys_true[0], interpolated_ys_true[1])
    area_sim = cal_area(x_sim, interpolated_ys_sim[0], interpolated_ys_sim[1])
    common_area = cal_intersection_area(x_true, interpolated_ys_true[0], interpolated_ys_true[1],
                                        x_sim, interpolated_ys_sim[0], interpolated_ys_sim[1])
    print("真实轨迹带：{}，模拟轨迹带：{}， 交集：{}， 交集/真实：{}，交集/模拟:{}"
          .format(area_true, area_sim, common_area, common_area / area_true, common_area / area_sim))
    plt.xticks([])
    plt.yticks([])
    plt.axis("off")
    plt.savefig("./figures/track_sim.svg", bbox_inches='tight')
    plt.savefig("./figures/track_sim.png", bbox_inches='tight')
    plt.show()