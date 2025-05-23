import model.rrt as rrt
import model.rrt_point as point
import numpy as np
import pickle
import random
import model.multy_line_rrt as mlrrt

ENTRANCE_PATH = "./vegetable2_area_exit_entrance_channel_2m.pkl"
ENTRANCE_LIST = [[116.276397908911, 39.8777897962623], [116.276470402447, 39.8777936233885]], [[116.27661422027, 39.8778012159132], [116.276686713806, 39.8778050430395]], [[116.276830531628, 39.8778126355641], [116.276903025165, 39.8778164626904]], [[116.277046842987, 39.8778240552151], [116.277119336524, 39.8778278823413]], [[116.277263154346, 39.877835474866], [116.277335647882, 39.8778393019922]], [[116.277479465705, 39.8778468945169], [116.277551959241, 39.8778507216432]], [[116.277173338355, 39.8772256091773], [116.277103115606, 39.8771964563507]], [[116.276961173538, 39.8771679438638], [116.27689313786, 39.8771143990033]], [[116.27675115655, 39.8770863241726], [116.276683120872, 39.877032779312]], [[116.276541218046, 39.8770038291691], [116.276471196737, 39.8769724297078]]
def get_entrance(path):
    # f = open("./vegetable2_area_exit_entrance_channel_2m.pkl", 'rb')
    # inOutAreaData = pickle.load(f)
    inOutAreaData = ENTRANCE_LIST
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

aisle = [[116.276363311022, 39.8773698765688], [116.276435219933, 39.8773736728311], [116.276397908911, 39.8777897962623], [116.276470402447, 39.8777936233885], [116.276507980313, 39.8773745238958], [116.276651798136, 39.8773821164205], [116.27661422027, 39.8778012159132], [116.276686713806, 39.8778050430395], [116.276724291672, 39.8773859435468], [116.276868109494, 39.8773935360714], [116.276830531628, 39.8778126355641], [116.276903025165, 39.8778164626904], [116.276940603031, 39.8773973631977], [116.277084420853, 39.8774049557224], [116.277046842987, 39.8778240552151], [116.277119336524, 39.8778278823413], [116.27715691439, 39.8774087828486], [116.277300732212, 39.8774163753733], [116.277263154346, 39.877835474866], [116.277335647882, 39.8778393019922], [116.277373115872, 39.8774214279366], [116.277445024783, 39.877425224199], [116.277442866489, 39.8774492952851], [116.277514689068, 39.8774540543909], [116.277479465705, 39.8778468945169], [116.277551959241, 39.8778507216432], [116.277586515495, 39.8774653216711], [116.277658424407, 39.8774691179334], [116.27767, 39.877326], [116.277541976429, 39.8773176930695], [116.277539965291, 39.8773401229453], [116.277164958941, 39.8773190633457], [116.277173338355, 39.8772256091773], [116.277103115606, 39.8771964563507], [116.277092465405, 39.8773152362194], [116.276948647582, 39.8773076436948], [116.276961173538, 39.8771679438638], [116.27689313786, 39.8771143990033], [116.276876154046, 39.8773038165685], [116.276732336224, 39.8772962240438], [116.27675115655, 39.8770863241726], [116.276683120872, 39.877032779312], [116.276659842687, 39.8772923969176], [116.276516024865, 39.8772848043929], [116.276541218046, 39.8770038291691], [116.276471196737, 39.8769724297078], [116.276443264485, 39.8772839533282], [116.276371355574, 39.8772801570659], [116.276363311022, 39.8773698765688]]
aisle_array = np.array(aisle)
longitude_max = max(aisle_array[:, 0])
longitude_min = min(aisle_array[:, 0])
latitude_max = max(aisle_array[:, 1])
latitude_min = min(aisle_array[:, 1])
all_entrance = get_entrance(ENTRANCE_PATH)
# args = {
#     "startPosition": all_entrance[0],
#     "endPosition": all_entrance[5],
#     "terminalDis": 3,
#     "aisle": aisle,
#     "lonRange": [longitude_min, longitude_max],
#     "latRange": [latitude_min, latitude_max],
#     "step": 3
# }
# rrt = rrt.RRT(**args)
# rrt.exploreRoute()
# rrt.buildRoute()
# rrt.plotMap()

startPositions = get_entrance(ENTRANCE_PATH)
endPositions = get_entrance(ENTRANCE_PATH)
print(len(ENTRANCE_LIST))
args2 = {
    "startPositions": [all_entrance[0]],
    "endPositions": all_entrance,
    "terminalDis": 3,
    "aisle": aisle,
    "aisle_json": None,
    "lonRange": [longitude_min, longitude_max],
    "latRange": [latitude_min, latitude_max],
    "step": 3,
    "is_multi_polygon": False
}
mlrrt = mlrrt.MLRRT(**args2)
res = []
for i in range(1, len(all_entrance)):
    args2['endPositions'] = endPositions[1:(i+1)]
    mlrrt.init_args(**args2)
    mlrrt.exploreRoute()
    mlrrt.prepareRoute()
    sum_num = sum([len(route) for route in mlrrt.routes])
    res.append(sum_num / len(mlrrt.tree))
print(res)
# mlrrt.plotMap()
