import networkx as nx
import matplotlib.pyplot as plt

# 创建一个有向图
G = nx.DiGraph()

# 添加节点（传播者、感染者）
nodes = ['Patient 0', 'Patient 1', 'Patient 2', 'Patient 3', 'Patient 4']
G.add_nodes_from(nodes)

# 添加传播路径（有向边）
edges = [
    ('Patient 0', 'Patient 1'),
    ('Patient 0', 'Patient 2'),
    ('Patient 1', 'Patient 3'),
    ('Patient 2', 'Patient 4')
]
G.add_edges_from(edges)

# 绘制传播链路图
plt.figure(figsize=(10, 6))

# 定义节点位置
pos = nx.spring_layout(G, seed=42)  # 使用 spring 布局算法

# 绘制节点
nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')

# 绘制边
nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='gray')

# 绘制标签
nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')

# 添加标题
plt.title("Propagation Chain", fontsize=14)

# 显示图形
plt.show()
