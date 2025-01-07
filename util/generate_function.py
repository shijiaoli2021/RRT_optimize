import model.rrt_point as point
import random
import model.multy_line_rrt as ml_rrt
import pickle

def get_entrance(entrance_list):
    # f = open("./vegetable2_area_exit_entrance_channel_2m.pkl", 'rb')
    # inOutAreaData = pickle.load(f)
    inOutAreaData = entrance_list
    res = []
    for i in range(len(inOutAreaData)):
        point1 = point.Point(inOutAreaData[i][0][1], inOutAreaData[i][0][0])
        point2 = point.Point(inOutAreaData[i][1][1], inOutAreaData[i][1][0])
        res.append(generate_position(point1, point2))
    return res

def generate_position(point1, point2):
    k = (point2.lon - point1.lon) / (point2.lat - point1.lat)
    b = point2.lon - k * point2.lat
    random_lat = random.uniform(point1.lat, point2.lat)
    return [k * random_lat + b, random_lat]

def generate_trac(pointList, args):
    # 生成轨迹
    print("----开始生成轨迹----")
    mlrrt_model = ml_rrt.MLRRT(**args)
    true_res = []
    sim_res = []
    for i in range(len(pointList)):
        # 随机选择5-len(point)轨迹
        end_num = random.randint(4, 10)

        # 随机选择结束点
        end_point_idx = random.sample(list(range(0, i)) + list(range(i+1, len(pointList))), end_num)

        # 生成轨迹
        print("第{}个点预计生成{}条轨迹，结束点位为：{}".format(i, end_num, end_point_idx))
        # 1）配置参数
        args["startPositions"] = [pointList[i]]
        args["endPositions"] = [pointList[idx] for idx in end_point_idx]

        # 2）初始化参数
        mlrrt_model.init_args(**args)
        # 3) 生成轨迹
        mlrrt_model.exploreRoute()
        mlrrt_model.prepareRoute()
        true_res += mlrrt_model.routes

        mlrrt_model.init_args(**args)
        # 3) 生成轨迹
        mlrrt_model.exploreRoute()
        mlrrt_model.prepareRoute()
        sim_res += mlrrt_model.routes
    print("生成轨迹完毕，共生成{}条轨迹".format(len(true_res)))
    true_res = [[[point.lon, point.lat] for point in trac] for trac in true_res]
    sim_res = [[[point.lon, point.lat] for point in trac] for trac in sim_res]
    return true_res, sim_res




