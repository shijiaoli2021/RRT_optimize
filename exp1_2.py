import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'

plt.figure(figsize=(8, 7))
data = [7.40, 17.45, 27.37, 34.57, 39.82, 33.80, 45.98, 48.82, 62.19]
plt.plot(range(1, len(data)+1), data, marker='o', linewidth=2.5, markersize=8)
plt.xlabel("目标终点个数", fontsize = 18)
plt.ylabel('有效点比例/%', fontsize = 18)
plt.tick_params(axis='both', labelsize=16)
plt.savefig('./figures/exp_zhexian.png', bbox_inches='tight')
plt.savefig('./figures/exp_zhexian.svg', bbox_inches='tight')
plt.savefig('./figures/exp_zhexian.eps', bbox_inches='tight')



plt.show()