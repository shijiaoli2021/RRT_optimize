import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.rcParams['font.family'] = 'SimHei'

titles = ["（a）病例-主体6传播过程", "（b）病例-主体13传播过程", "（c）病例-主体1传播过程", "（d）病例-主体51传播过程"]
image_files = ["figures/(a) 病例-主体6传播过程1.png",
               "figures/(b) 病例-主体13传播过程1.png",
               "figures/(c) 病例-主体1传播过程1.png",
               "figures/(d) 病例-主体51传播过程1.png"]
fig, axes = plt.subplots(2, 2, figsize=(16, 10))  # 创建一个2x2的网格，调整画布大小

# 展示每张图片，并为每张图片添加标题
for i, ax in enumerate(axes.flat):
    img = mpimg.imread(image_files[i])  # 读取图片
    ax.imshow(img)  # 显示图片
    ax.set_title(titles[i], y=-0.13, fontsize = 20)  # 设置标题
    ax.axis('off')  # 关闭坐标轴显示

# 调整间距
plt.subplots_adjust(wspace=0.03)  # 调整水平和垂直间距
plt.savefig("./figures/bingli.png", bbox_inches='tight')
plt.show()