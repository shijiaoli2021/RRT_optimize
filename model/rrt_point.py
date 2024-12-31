import math

class Point:
    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon
        self.pre = None

def cal_distance(point1, point2, type):
    if type == 'Euclidean':
        return math.sqrt((point1.lat - point2.lat) ** 2 + (point1.lon - point2.lon) ** 2)
    if type == 'Manhattan':
        return math.sqrt(abs(point1.lat - point2.lat) + abs(point1.lon - point2.lon))
    if type == 'LL':
        lon1, lat1, lon2, lat2 = map(math.radians, [point1.lon, point1.lat, point2.lon, point2.lat])
        dlgt = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlgt / 2) ** 2
        distance = 2 * math.asin(math.sqrt(a)) * 6371 * 1000
        distance = round(distance, 3)
        return distance