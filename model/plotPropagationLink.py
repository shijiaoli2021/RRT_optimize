import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from matplotlib.ticker import MultipleLocator
from itertools import cycle

COLOR = ["#9602d2", "#fb0104", "#d7691b", "#dca522",
         "#dbb98b", "#f8e9d6", "#ffd504", "#6a8e24",
         "#c1e7b5", "#8dbd8b", "#99fa98", "#83cdfc",
         "#aed8e6", "#00c0bf", "#f2feff"]
SCATTER_COLOR = []

TIME_DELTA = 1

RADIUS_STEP = 1

def plotFunc(r, startAngle, endAngle, ax, color):
    theta = np.linspace(np.radians(startAngle), np.radians(endAngle), 100)
    ax.plot(theta, np.full_like(theta, r), linewidth=3, color=color)

def scatterOnR(r, startAngle, endAngle, ):
    pass

def displayChild(point):
    if point is None:
        return
    print("idx:{},level:{}".format(point.idx, point.level))
    if point.next is None or len(point.next) == 0:
        return
    for level in point.next.keys():
        list = point.next[level]
        for cashPoint in list:
            displayChild(cashPoint)


class Point:
    def __init__(self, idx, level):
        self.idx = idx
        self.level = level
        self.pre = None
        self.next = {}
    def setPre(self, pre):
        self.pre = pre
    def setNext(self, next):
        if next.level not in self.next.keys():
            self.next[next.level] = []
        self.next[next.level].append(next)

def plot(point, startTheta):

    '''
    从根节点开始画， 标记根节点后，开始画其子节点
    深度优先画点
    '''
    angleMap = {}
    angleMap[point.level] = startTheta
    scatterData = []
    lineData = []
    pointMap = {}
    plotDeepFirst(point, angleMap, scatterData, lineData, pointMap, startTheta)
    ax = plt.subplot(111, polar=True)
    scatterData = np.array(scatterData)
    print(scatterData)
    ax.scatter(np.radians(scatterData[:, 1]), scatterData[:, 0])
    for line in lineData:
        line = np.array(line)
        ax.plot(np.radians(line[:, 1]), line[:, 0], color=COLOR[int(line[0, 0])])
    # plt.show()

def plot2(point):
    points = []
    edges = []
    buildNodeAndEdges(point, points, edges)
    # 创建一个有向图
    G = nx.DiGraph()
    G.add_nodes_from(points)
    G.add_edges_from(edges)
    # 定义节点位置
    pos = nx.spring_layout(G, seed=42, weight='weight')  # 使用 spring 布局算法
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue')
    # 绘制边
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='gray')
    # 绘制标签
    nx.draw_networkx_labels(G, pos, font_size=5, font_color='black')
    # 添加标题
    plt.title("Propagation Chain", fontsize=14)
    # 显示图形
    plt.show()


def plot_propagation_tree(roots, node_colors, angle_start=np.pi / 6, angle_range=2 * np.pi, radius_step=1, min_angle_gap=0.4,
                          start_date='2022-05-12', date_interval=5):
    """
    绘制传播树的极坐标图，调整点的分布以避免重叠，将r轴刻度映射为日期，并设置日期间隔。

    参数:
    - root: 头节点 (Node 对象)。
    - angle_start: 起始角度 (弧度)。
    - angle_range: 总角度范围 (弧度)。
    - radius_step: 每层之间的半径差。
    - min_angle_gap: 子节点之间的最小角度间隔 (弧度)。
    - start_date: 起始日期 (字符串格式)。
    - date_interval: 日期间隔，单位为天。
    """

    def parse_node(node, angle_start, angle_range):
        """递归解析树节点，生成极坐标和连接关系。"""
        num_children = sum(len(node.next.get(l, [])) for l in node.next)
        # 调整角度范围以考虑最小间隔
        effective_angle_range = angle_range - num_children * min_angle_gap
        angle_step = effective_angle_range / max(num_children, 1)
        current_angle = angle_start

        # 当前节点
        r = (node.level+1) * radius_step  # r 由 level 决定
        coords.append((r, current_angle, node.idx, node.level))
        if node.pre is not None:
            connections.append((node.pre, node))  # 存储父子节点对象关系

        # 处理子节点
        for l in node.next:
            for child in node.next[l]:
                # 递归调用时直接使用子节点
                parse_node(child, current_angle, angle_step)
                current_angle += angle_step + min_angle_gap  # 添加最小角度间隔
    # 创建图形和极坐标
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    interval = angle_range / len(tree)
    for i in range(len(roots)):
        root = roots[i]
        coords = []
        connections = []
        # 解析根节点
        parse_node(root, angle_start + i * interval, interval)

        # 获取最大层级和分配颜色
        max_level = max(level for r, theta, idx, level in coords)  # 获取最大层级
        colormap = plt.cm.viridis  # 使用viridis colormap
        level_colors = {
            level: colormap(1 - level / max_level)  # 反转颜色映射
            for level in {level for r, theta, idx, level in coords}  # 提取层级集合
        }
        # 绘制节点  # 所有节点的颜色
        for r, theta, idx, level in coords:
            ax.plot(theta, r, 'o', color=node_colors[i], markersize=8, label=f'Node {idx}')
            # ax.text(theta, r, f'{idx}', fontsize=5, ha='center', va='center')

        # 绘制边并根据起始层级决定颜色
        for parent, child in connections:
            parent_coord = next((r, theta) for r, theta, idx, level in coords if idx == parent.idx)
            child_coord = next((r, theta) for r, theta, idx, level in coords if idx == child.idx)
            edge_color = level_colors[parent.level]  # 边的颜色取决于父节点的层级
            ax.plot([parent_coord[1], child_coord[1]], [parent_coord[0], child_coord[0]], color=edge_color)
    # 设置标题和样式
    # ax.set_title("Propagation Tree in Polar Coordinates", va='bottom')
    ax.grid(True)
    # 将r值映射为日期并设置刻度
    start_date = datetime.strptime(start_date, '%Y-%m-%d')

    def r_to_date(r):
        """将r值（天数）转换为日期"""
        return start_date + timedelta(days=int(r))

    # 设置r轴刻度为日期
    r_values = [r for r, _, _, _ in coords]
    ax.set_yticks(r_values)  # 设置r轴刻度
    ax.set_yticklabels([r_to_date(r).strftime('%Y-%m-%d') for r in r_values])  # 设置日期标签
    ax.set_xticklabels([])

    # 设置r轴的日期间隔
    ax.yaxis.set_major_locator(MultipleLocator(date_interval))  # 设置r轴刻度的日期间隔

    # 添加图例
    legend_elements = [
        Line2D([0], [0], color=colormap(1 - 0 / max_level), lw=2, label=f'Level 0'),
        Line2D([0], [0], color=colormap(1 - 0.25), lw=2, label=f'Level 1'),
        Line2D([0], [0], color=colormap(1 - 0.5), lw=2, label=f'Level 2'),
        Line2D([0], [0], color=colormap(1 - 0.75), lw=2, label=f'Level 3'),
        Line2D([0], [0], color=colormap(1 - 1), lw=2, label=f'Level {max_level}')
    ]
    # ax.legend(handles=legend_elements, loc='upper right')


def plot_propagation_tree_with_main_arcs(root, angle_start=0, angle_range=2 * np.pi, radius_step=1, min_angle_gap=0.3,
                                         node_size=4):
    """
    绘制传播树的极坐标图，使用总弧线表示父节点与子节点之间的连接，并在总弧线末端分流到子节点。

    参数:
    - root: 头节点 (Node 对象)。
    - angle_start: 起始角度 (弧度)。
    - angle_range: 总角度范围 (弧度)。
    - radius_step: 每层之间的半径差。
    - min_angle_gap: 子节点之间的最小角度间隔 (弧度)。
    - node_size: 节点大小，默认值是100。
    """
    coords = []
    connections = []

    def parse_node(node, angle_start, angle_range):
        """递归解析树节点，生成极坐标和连接关系。"""
        num_children = sum(len(node.next.get(l, [])) for l in node.next)
        # 调整角度范围以考虑最小间隔
        effective_angle_range = angle_range - num_children * min_angle_gap
        angle_step = effective_angle_range / max(num_children, 1)
        current_angle = angle_start

        # 当前节点
        r = node.level * radius_step  # r 由 level 决定
        coords.append((r, current_angle, node.idx, node.level, node))  # 存储节点信息

        # 处理子节点
        for l in node.next:
            child_nodes = node.next[l]
            if child_nodes:
                child_start_angle = current_angle
                child_end_angle = current_angle + angle_step * len(child_nodes)
                child_angle_range = child_end_angle - child_start_angle

                # 为父节点添加总弧线连接到子节点范围
                connections.append(("arc", node, child_nodes, child_start_angle, child_end_angle, r))

                # 递归处理每个子节点
                for child in child_nodes:
                    parse_node(child, current_angle, angle_step)
                    current_angle += angle_step + min_angle_gap  # 添加最小角度间隔

    # 解析根节点
    parse_node(root, angle_start, angle_range)

    # 创建图形和极坐标
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

    # 绘制节点
    node_color = 'blue'  # 所有节点的颜色
    for r, theta, idx, level, node in coords:
        ax.plot(theta, r, 'o', color=node_color, markersize=node_size, label=f'Node {idx}')
        ax.text(theta, r, f'{idx}', fontsize=8, ha='center', va='center')

    # 绘制边
    for connection in connections:
        if connection[0] == "arc":
            _, parent, children, start_angle, end_angle, r = connection
            child_radius = r + radius_step  # 子节点所在的半径

            # 绘制总弧线
            angles = np.linspace(start_angle, end_angle, 100)
            radii = np.full_like(angles, r)
            ax.plot(angles, radii, color='gray', lw=1.5, linestyle='--')

            # 绘制分流线到每个子节点
            child_angle_step = (end_angle - start_angle) / len(children)
            for i, child in enumerate(children):
                child_angle = start_angle + i * child_angle_step + child_angle_step / 2
                ax.plot([child_angle, child_angle], [r, child_radius], color='gray', lw=1)

    # 设置标题和样式
    ax.set_title("Propagation Tree with Main Arcs", va='bottom')
    ax.grid(True)



def plot_propagation_tree_with_colors(root, angle_start=0, angle_range=2 * np.pi, radius_step=1, min_angle_gap=0.3,
                                      node_size=5):
    """
    绘制传播树的极坐标图，每条链路的节点和边用不同颜色标记，父节点和子节点直接相连。

    参数:
    - root: 头节点 (Node 对象)。
    - angle_start: 起始角度 (弧度)。
    - angle_range: 总角度范围 (弧度)。
    - radius_step: 每层之间的半径差。
    - min_angle_gap: 子节点之间的最小角度间隔 (弧度)。
    - node_size: 节点大小，默认值是100。
    """
    coords = []
    connections = []

    # 创建颜色循环器
    colors = cycle(plt.cm.tab10.colors)  # 使用 Matplotlib 内置的 10 种颜色
    node_colors = {}  # 用于存储每条链路的颜色

    def parse_node(node, angle_start, angle_range, color=None):
        """递归解析树节点，生成极坐标和连接关系，并分配颜色。"""
        num_children = sum(len(node.next.get(l, [])) for l in node.next)
        # 调整角度范围以考虑最小间隔
        effective_angle_range = angle_range - num_children * min_angle_gap
        angle_step = effective_angle_range / max(num_children, 1)
        current_angle = angle_start

        # 当前节点
        r = node.level * radius_step  # r 由 level 决定
        coords.append((r, current_angle, node.idx, node.level, node, color))  # 存储节点信息

        # 如果是根节点的直接子节点，为该链路分配新颜色
        if node.level == 0:
            color = next(colors)

        # 处理子节点
        for l in node.next:
            child_nodes = node.next[l]
            for child in child_nodes:
                child_angle = current_angle + angle_step / 2  # 子节点分配的角度中点
                child_r = (node.level + 1) * radius_step  # 子节点的半径

                # 添加连接
                connections.append(((current_angle + angle_step / 2, r), (child_angle, child_r), color))

                # 递归处理子节点
                parse_node(child, current_angle, angle_step, color)
                current_angle += angle_step + min_angle_gap  # 添加最小角度间隔

    # 解析根节点
    parse_node(root, angle_start, angle_range)

    # 创建图形和极坐标
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

    # 绘制节点
    for r, theta, idx, level, node, color in coords:
        ax.plot(theta, r, 'o', color=color, markersize=node_size, label=f'Node {idx}')
        ax.text(theta, r, f'{idx}', fontsize=8, ha='center', va='center')

    # 绘制边
    for ((theta1, r1), (theta2, r2), color) in connections:
        ax.plot([theta1, theta2], [r1, r2], color=color, lw=1)

    # 设置标题和样式
    ax.set_title("Propagation Tree with Direct Connections", va='bottom')
    ax.grid(True)


def buildNodeAndEdges(point, points, edges):
    if point is None:
        return
    points.append(("point" + str(point.idx)))
    if point.pre is not None:
        edges.append(("point"+str(point.pre.idx), ("point"+str(point.idx), point.level - point.pre.level)))
    if point.next is not None:
        for level in point.next:
            list = point.next[level]
            for cashPoint in list:
                buildNodeAndEdges(cashPoint, points, edges)

def plotDeepFirst(point:Point, angleMap, scatterData, lineData, pointMap, min_const):
    if point.level not in angleMap.keys():
        angleMap[point.level] = min_const
    if point.pre is not None:
        minTheta = max(angleMap[point.level], angleMap[point.pre.level])
    else:
        minTheta = angleMap[point.level]
    # minTheta = angleMap[point.level]
    scatterData.append([point.level, minTheta])
    pointMap[point] = minTheta
    if point.pre is not None:
        lineData.append([[point.pre.level, pointMap[point.pre]], [point.level, minTheta]])
    angleMap[point.level] = minTheta + 60 / (point.level + 2)
    if point.next is None or len(point.next) == 0:
        return
    for level in point.next.keys():
        list = point.next[level]
        for cashPoint in list:
            plotDeepFirst(cashPoint, angleMap, scatterData, lineData, pointMap, min_const)

data = np.array(pd.read_excel("../data/ex8.xlsx").values)
print(data)
baseTime = np.datetime64('1970-01-01')
dataLabel = data[:, 1]
# data[:, 1] = [int((np.datetime64(t) - baseTime) / np.timedelta64(1, 'D')) for t in data[:, 1]]
r_to_label = {data[i, 1] * RADIUS_STEP:dataLabel[i] for i in range(len(data))}
timeIdx = {}
minD = min(data[:, 1])
maxD = max(data[:, 1])
for d in range(minD, maxD+1):
    timeIdx[d] = 1 + (d - minD) / TIME_DELTA
map = {}
for i in range(len(data)):
    item = data[i]
    point = Point(item[0], int(timeIdx[item[1]]))
    map[item[0]] = point

tree = []
treeIdx = set()
# 构建树关系
for i in range(len(data)):
    item = data[i]
    point = map[item[0]]
    if point is None:
        continue
    if item[2] not in map.keys():
        prePoint = Point(item[2], 0)
        point.setPre(prePoint)
        prePoint.setNext(point)
        if item[2] not in treeIdx:
            tree.append(prePoint)
            treeIdx.add(item[2])
        map[item[2]] = prePoint
    else:
        prePoint = map[item[2]]
        point.setPre(prePoint)
        prePoint.setNext(point)
print("树中根节点数量：{}".format(len(tree)))
point = tree[0]
    # displayChild(point)
plot_propagation_tree(tree, COLOR)
plt.savefig("../figures/polar.svg", bbox_inches= 'tight', transparent=True)
plt.savefig("../figures/polar.png", bbox_inches= 'tight', transparent=True)
plt.show()

