import model.rrt_point as rrtPoint
import random
from shapely.geometry import Point, Polygon, MultiPolygon, shape, LineString
import matplotlib.pyplot as plt

'''
RRT 算法
'''
class RRT:
    def __init__(self, **kwargs):
        # 初始化路径树
        print(kwargs)
        self.init_args(**kwargs)

    def init_args(self, **kwargs):
        self.startPoint = rrtPoint.Point(kwargs['startPosition'][1], kwargs['startPosition'][0])
        self.endPoint = rrtPoint.Point(kwargs['endPosition'][1], kwargs['endPosition'][0])
        self.teminalDis = kwargs['terminalDis']
        self.initRouteTree(self.startPoint)
        self.original_aisle = kwargs['aisle']
        self.lonRange = kwargs['lonRange']
        self.latRange = kwargs['latRange']
        self.step = kwargs['step']
        self.routes = []
        aisle = kwargs['aisle']
        if kwargs['aisle_json'] is not None:
            # 解析 GeoJSON 数据中的几何部分
            geojson_data = kwargs['aisle_json']
            feature = geojson_data['features'][0]
            geometry = feature['geometry']
            # 将坐标转换为 Shapely 的 Polygon 对象
            polygon = shape(geometry)
            polygon = polygon.buffer(0)
            self.aisle = polygon
            return
        if not kwargs['is_multi_polygon']:
            self.aisle = Polygon(aisle)
        else:
            polygons = []
            for area_cash in aisle:
                polygons.append(Polygon(area_cash))
            self.aisle = MultiPolygon(polygons)
            self.plotMap_multi_polygon()
            if not self.aisle.is_valid:
                print("-----------buffer----------")
                self.aisle = self.aisle.buffer(0)

    # def initBarrierKb(self):
    #     for line in self.barriers:
    #         line.append((line[1][0] - line[1][0]) / line[1][1] - line[1][0])
    #         line.append(line[1][0] - line[2]*line[1][1])

    def initRouteTree(self, startPoint):
        self.tree = [startPoint]
        self.treePositionList = [(startPoint.lon, startPoint.lat)]

    def exploreRoute(self):
        while True:
            point = self.productRandomPoint()
            prePoint = self.findNear(point)
            newPoint = self.productNewPoint(point, prePoint)
            while (not LineString([(newPoint.lon, newPoint.lat), (prePoint.lon, newPoint.lat)]).within(self.aisle)):
                point = self.productRandomPoint()
                prePoint = self.findNear(point)
                newPoint = self.productNewPoint(point, prePoint)
            newPoint.pre = prePoint
            self.tree.append(newPoint)
            self.treePositionList.append((newPoint.lat, newPoint.lon))
            dis = rrtPoint.cal_distance(newPoint, self.endPoint, 'LL')
            # print("dis:{}, point_num:{}".format(dis, len(self.tree)))
            if dis < self.teminalDis:
                self.endPoint.pre = newPoint
                self.tree.append(self.endPoint)
                break

    def productRandomPoint(self):
        point = rrtPoint.Point(random.uniform(self.latRange[0], self.latRange[1]),
                               random.uniform(self.lonRange[0], self.lonRange[1]))
        while not self.checkPoint(point):
            point = rrtPoint.Point(random.uniform(self.latRange[0], self.latRange[1]),
                                   random.uniform(self.lonRange[0], self.lonRange[1]))
        return point

    def productNewPoint(self, point, prePoint):
        dis = rrtPoint.cal_distance(point, prePoint, 'LL')
        return rrtPoint.Point(prePoint.lat + (self.step / dis) * (point.lat - prePoint.lat),
                              prePoint.lon + (self.step / dis) * (point.lon - prePoint.lon))


    def checkPoint(self, point):
        if (point.lon, point.lat) in self.treePositionList:
            return False
        else:
            return True

    def plotMap_multi_polygon(self):
        multi_polygon = self.aisle
        # 绘制 MultiPolygon 对象
        fig, ax = plt.subplots()

        for geom in multi_polygon.geoms if isinstance(multi_polygon, MultiPolygon) else [multi_polygon]:
            # 绘制外环（外部边界）
            x, y = multi_polygon.exterior.xy
            ax.fill(x, y, alpha=0.5, fc='lightblue', edgecolor='blue')

            # 绘制内环（如果有）
            for interior in multi_polygon.interiors:
                x, y = interior.xy
                ax.fill(x, y, alpha=0.5, fc='lightgreen', edgecolor='green')

        # 设置标题
        ax.set_title("MultiPolygon")

        # 显示图形
        plt.show()

    def checkExploreTerminal(self, newPoint):
        return rrtPoint.cal_distance(newPoint, self.endPoint, 'LL') < self.teminalDis

    # def checkBarier(self, point1, point2):
    #     k = (point1.lat - point2.lat) / (point1.lon - point2.lon)
    #     b = point1.lat - k * point1.lon
    #     for line in self.barriers:
    #         if k == line[2]:
    #             if b != line[3]:
    #                 return False
    #             if min(line[0][0], line[0][1]) > max(point1[0], point2[0])


    def findNear(self, point):
        nearIdx = self.startPoint
        dis = rrtPoint.cal_distance(point, self.startPoint, 'LL')
        for cashPoint in self.tree:
            cash_dis = rrtPoint.cal_distance(point, cashPoint, 'LL')
            if cash_dis < dis:
                nearIdx = cashPoint
                dis = cash_dis
        return nearIdx


    def buildRoute(self):
        point = self.endPoint
        self.routes.insert(0, point)
        while point.pre is not None:
            self.routes.insert(0, point.pre)
            point = point.pre

    def clearTree(self):
        return

    def plotMap(self):
        plt.figure()
        plt.xlim((self.lonRange[0], self.lonRange[1]))
        plt.ylim((self.latRange[0], self.latRange[1]))
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.xticks(np.arange(x_width))
        # plt.yticks(np.arange(y_width))
        plt.grid()
        plt.plot(self.startPoint.lat, self.startPoint.lon, 'ro')
        plt.plot(self.endPoint.lat, self.endPoint.lon, marker='o', color='yellow')
        import numpy as np
        aisle_array = np.array(self.original_aisle)
        plt.plot(aisle_array[:, 0], aisle_array[:, 1], 'k-', linewidth='1')

        print('len(routine_list)', len(self.routes))
        routesY = [point.lat for point in self.routes]
        routesX = [point.lon for point in self.routes]
        treeNodeX = [point.lon for point in self.tree]
        treeNodeY = [point.lat for point in self.tree]
        plt.scatter(treeNodeX, treeNodeY, color='red', marker='o')
        plt.plot(routesX, routesY, '-', linewidth='2', color='violet', marker='o')
        plt.show()





