import json
import matplotlib.pyplot as plt
from shapely.geometry import shape
from shapely.geometry import Polygon, MultiPolygon, Point
from data.trajectoryData import *

# 提供的 GeoJSON 数据
geojson_data = geo_json

# 解析 GeoJSON 数据中的几何部分
feature = geojson_data['features'][0]
geometry = feature['geometry']

# 将坐标转换为 Shapely 的 Polygon 对象
polygon = shape(geometry)

polygon = polygon.buffer(0)

if Point(116.33279717654608, 39.97783580410584).within(polygon):
    print("在内部")

# 创建一个绘图对象
fig, ax = plt.subplots()


# 绘制外环（外部边界）
for geom in polygon.geoms if isinstance(polygon, MultiPolygon) else [polygon]:
    x, y = geom.exterior.xy
    ax.fill(x, y, alpha=0.5, fc='lightblue', edgecolor='blue')

    # 绘制内环（如果有）
    for interior in geom.interiors:
        x, y = interior.xy
        ax.fill(x, y, alpha=0.5, fc='lightgreen', edgecolor='green')


# 设置标题
ax.set_title("Polygon from GeoJSON")

# 显示图形
plt.show()
