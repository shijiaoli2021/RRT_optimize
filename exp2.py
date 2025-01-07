'''
生成轨迹，与热力图展示
'''
import util.generate_function as gen
import util.plot_function as plot_fun
from data.trajectoryData import *
import matplotlib.pyplot as plt
import pickle
import numpy as np

# pointList = gen.get_entrance(doorLine)
range = plot_fun.get_range(area)
# args = {
#         "startPositions": [],
#         "endPositions": [],
#         "terminalDis": 6,
#         "aisle": area,
#         "aisle_json": geo_json,
#         "is_multi_polygon": True,
#         "lonRange": [range[2], range[3]],
#         "latRange": [range[0], range[1]],
#         "step": 3
#     }
# trueRoutes, simRoutes = gen.generate_trac(pointList, args)
# with open('./data/true_trac.pkl', 'wb') as true_file:
#         pickle.dump(trueRoutes, true_file)
#         true_file.close()
#         # trueRoutes = pickle.load(file, encoding='bytes')
# with open('./data/sim_trac.pkl', 'wb') as sim_file:
#         pickle.dump(simRoutes, sim_file)
#         sim_file.close()
plt.rcParams['font.family'] = 'SimHei'  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题
with open('./data/true_trac.pkl', 'rb') as true_file:
        trueRoutes = pickle.load(true_file, encoding='bytes')
with open('./data/sim_trac.pkl', 'rb') as sim_file:
        simRoutes = pickle.load(sim_file, encoding='bytes')
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
true_counts = plot_fun.plot_heatmap(plt, 128, 128, 'Blues', trueRoutes, range)
plt.title("（a）真实轨迹热力图", y=-0.17, fontsize = 20)
plt.subplot(1, 2, 2)
sim_counts = plot_fun.plot_heatmap(plt, 128, 128, 'Reds', simRoutes, range)
plt.title("（b）生成轨迹热力图", y=-0.17, fontsize = 20)
plt.subplots_adjust(wspace=0.32)
plt.savefig('./figures/heatmap.svg', bbox_inches='tight')
plt.savefig('./figures/heatmap.png', bbox_inches='tight')
# plt.subplot(1, 3, 3)
# plot_fun.plot_sim_heapmap(plt, true_counts, sim_counts, 'YlGnBu')
plt.show()