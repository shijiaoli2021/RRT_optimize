import matplotlib.pyplot as plt
import numpy as np

# 树结构示例
tree = {
    "idx": 0,
    "level": 0,
    "pre": None,
    "next": {
        1: [{"idx": 1, "level": 1, "pre": 0, "next": {}},
            {"idx": 2, "level": 1, "pre": 0, "next": {
                2: [{"idx": 3, "level": 2, "pre": 2, "next": {}}],
                3: [{"idx": 4, "level": 3, "pre": 3, "next": {}}]
            }}]
    }
}


# 解析树为极坐标数据
def parse_tree(node, angle_start=0, angle_range=2 * np.pi, radius_step=1):
    """解析树结构，生成极坐标"""
    coords = []
    connections = []

    def _parse_node(node, level, angle_start, angle_range, radius):
        num_children = sum(len(node['next'].get(l, [])) for l in node['next'])
        angle_step = angle_range / max(num_children, 1)
        current_angle = angle_start

        # 当前节点
        coords.append((radius, current_angle, node['idx']))
        if node['pre'] is not None:
            connections.append((node['pre'], node['idx']))

        # 处理子节点
        for l in node['next']:
            for child in node['next'][l]:
                _parse_node(child, level + 1, current_angle, angle_step, radius + radius_step)
                current_angle += angle_step

    _parse_node(node, 0, angle_start, angle_range, 0)
    return coords, connections


# 获取极坐标数据
coords, connections = parse_tree(tree)

# 绘制传播树的极坐标图
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

# 绘制节点
for r, theta, idx in coords:
    ax.plot(theta, r, 'o', label=f'Node {idx}')
    ax.text(theta, r, f'{idx}', fontsize=10, ha='center', va='center')

# 绘制边
for parent, child in connections:
    parent_coord = next((r, theta) for r, theta, idx in coords if idx == parent)
    child_coord = next((r, theta) for r, theta, idx in coords if idx == child)
    ax.plot([parent_coord[1], child_coord[1]], [parent_coord[0], child_coord[0]], 'k-')

# 设置标题和样式
ax.set_title("Propagation Tree in Polar Coordinates", va='bottom')
ax.grid(True)

# 显示图形
plt.show()
