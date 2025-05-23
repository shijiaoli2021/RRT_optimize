import model.rrt_point as rrtPoint
import random
from shapely.geometry import Point, Polygon, LineString, shape, MultiPolygon
import matplotlib.pyplot as plt

'''
RRT 算法
'''
class MLRRT:
    def __init__(self, **kwargs):
        # 初始化路径树
        print(kwargs)
        self.init_args(**kwargs)

    def init_args(self, **kwargs):
        self.buildStartAndEndPoints(kwargs['startPositions'], kwargs['endPositions'])
        self.teminalDis = kwargs['terminalDis']
        self.initRouteTree()
        self.finishNum = 0
        self.routes = []
        self.lonRange = kwargs['lonRange']
        self.latRange = kwargs['latRange']
        self.step = kwargs['step']
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
            self.original_aisle = aisle
        else:
            polygons = []
            for area_cash in aisle:
                polygons.append(Polygon(area_cash))
            self.aisle = MultiPolygon(polygons)
            if not self.aisle.is_valid:
                print("-----------buffer----------")
                self.aisle = self.aisle.buffer(0)


    def buildPoints(self, positionsList):
        return [rrtPoint.Point(position[1], position[0]) for position in positionsList]

    def buildPointsDict(self, positionsList):
        return {rrtPoint.Point(position[1], position[0]): 0 for position in positionsList}

    def buildStartAndEndPoints(self, startPositions, endPositions):
        self.startPoints = self.buildPointsDict(startPositions)
        self.endPoints = self.buildPointsDict(endPositions)

    # def initBarrierKb(self):
    #     for line in self.barriers:
    #         line.append((line[1][0] - line[1][0]) / line[1][1] - line[1][0])
    #         line.append(line[1][0] - line[2]*line[1][1])

    def initRouteTree(self):
        self.tree = []
        self.treePositionList = []

    def exploreRoute(self):
        while True:
            if len(self.tree) % 60 == 0:
                print("当前树中点位数量共{}个，完成路径探索条数：{}".format(len(self.tree), self.finishNum))
            if self.checkExploreTerminal():
                break
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
            # 检查新点是否完成新的一条路径探索，并处理
            self.handleNewPoint(newPoint)

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

    def handleNewPoint(self, newPoint):
        unFinishPoints = [key for (key, value) in self.endPoints.items() if value == 0]
        if len(unFinishPoints) == 0:
            return
        for endPoint in unFinishPoints:
            if self.checkEnding(endPoint, newPoint):
                return

    def checkEnding(self, endPoint, newPoint):
        dis = rrtPoint.cal_distance(newPoint, endPoint, 'LL')
        if dis < self.teminalDis:
            endPoint.pre = newPoint
            self.tree.append(endPoint)
            self.endPoints[endPoint] = 1
            self.finishNum += 1
            return True
        return False


    def checkPoint(self, point):
        if (point.lon, point.lat) in self.treePositionList:
            return False
        else:
            return True

    def checkExploreTerminal(self):
        finish = len([key for (key, value) in self.endPoints.items() if value == 1])
        return finish == len(self.endPoints)

    def findNear(self, point):
        # 找到距离最近的起始点进行开始匹配
        nearStart = self.findNearPoint(point, [key for (key, value) in self.startPoints.items() if value == 0])
        if nearStart is not None:
            self.tree.append(nearStart)
            self.startPoints[nearStart] = 1
        return self.findNearPoint(point, self.tree)

    def findNearPoint(self, point, points):
        if len(points) == 0:
            return None
        nearPoint = points[0]
        dis = rrtPoint.cal_distance(point, points[0], 'LL')
        for cashPoint in points:
            cash_dis = rrtPoint.cal_distance(point, cashPoint, 'LL')
            if cash_dis < dis:
                nearPoint = cashPoint
                dis = cash_dis
        return nearPoint

    def prepareRoute(self):
        for endPoint in self.endPoints:
            if self.endPoints[endPoint] == 0:
                continue
            self.buildRoute(endPoint)

    def buildRoute(self, endPoint):
        route = []
        idx = endPoint
        route.insert(0, idx)
        while idx.pre is not None:
            route.insert(0, idx.pre)
            idx = idx.pre
        self.routes.append(route)

    def clearTree(self):
        return

    def plotMap(self):
        plt.figure(figsize=(12.5, 5))
        plt.xlim((self.lonRange[0], self.lonRange[1]))
        plt.ylim((self.latRange[0], self.latRange[1]))
        plt.xticks([])
        plt.yticks([])
        plt.axis("off")
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.xticks(np.arange(x_width))
        # plt.yticks(np.arange(y_width))
        # plt.grid()
        # plt.plot(self.startPoint.lat, self.startPoint.lon, 'ro')
        # plt.plot(self.endPoint.lat, self.endPoint.lon, marker='o', color='yellow')
        import numpy as np
        aisle_array = np.array(self.original_aisle)
        # plt.plot(aisle_array[:, 0], aisle_array[:, 1], '-', color = 'white', linewidth='1')

        print('len(routine_list)', len(self.routes))
        routeSet = set()
        for route in self.routes:
            routeSet.update(route)
        inValidNode = [node for node in self.tree if node not in routeSet]
        inValidNode_X = [point.lon for point in inValidNode]
        inValidNode_Y = [point.lat for point in inValidNode]
        validNode_X = [point.lon for point in routeSet]
        validNode_Y = [point.lat for point in routeSet]
        # 绘制route
        for route in self.routes:
            routeY = [point.lat for point in route]
            routeX = [point.lon for point in route]
            plt.plot(routeX, routeY, '-', linewidth='2', color='#53A8FF', zorder=1)
        plt.scatter(inValidNode_X, inValidNode_Y, color='#D7D7D7', marker='o', zorder=2)
        plt.scatter(validNode_X, validNode_Y, color='#5C5CFF', marker='o', zorder=2)
        x, y = self.aisle.exterior.xy
        plt.fill(x, y, fc='white', edgecolor='white', label='原始多边形')
        plt.savefig("./figures/mlrrt.svg", bbox_inches= 'tight', transparent=True)
        plt.savefig("./figures/mlrrt.png", bbox_inches= 'tight', transparent=True)
        plt.show()