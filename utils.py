# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from math import radians, atan, tan, sin, acos, cos

def getDistance(latA, lonA, latB, lonB):

    ra = 6378140  # 赤道半径: 米
    rb = 6356755  # 极线半径: 米
    flatten = (ra - rb) / ra  # Partial rate of the earth
    # change angle to radians
    radLatA = radians(latA)#弧度
    radLonA = radians(lonA)
    radLatB = radians(latB)
    radLonB = radians(lonB)

    try:
        pA = atan(rb / ra * tan(radLatA))
        pB = atan(rb / ra * tan(radLatB))
        x = acos(sin(pA) * sin(pB) + cos(pA) * cos(pB) * cos(radLonA - radLonB))
        c1 = (sin(x) - x) * (sin(pA) + sin(pB)) ** 2 / cos(x / 2) ** 2
        c2 = (sin(x) + x) * (sin(pA) - sin(pB)) ** 2 / sin(x / 2) ** 2
        dr = flatten / 8 * (c1 - c2)
        distance = ra * (x + dr)
        return distance  # meter
    except:
        return 0.0000001
        
def f(d):
    return 1 / (1 + np.exp(-(d-1000)/250))
    
def getDistanceFromDF(data):
    tmp = data[['end_lat','end_lon','predict_end_lat','predict_end_lon']].astype(float)
    #从数据中取出'end_lat','end_lon','predict_end_lat','predict_end_lon'4列，转为浮点型
    error = []#设置一个空列表error
    for i in tmp.values:
        t = getDistance(i[0],i[1],i[2],i[3])#逐条计算误差
        error.append(t)#将误差加入列表error
    print (np.sum(f(np.array(error))) / tmp.shape[0])#打印

def dateConvert(data,isTrain):

    print ('convert string to datetime')
    data['start_time'] = pd.to_datetime(data['start_time'])#转化开始时间
    if isTrain:#如果是训练集
        data['end_time'] = pd.to_datetime(data['end_time'])#转化结束时间
    data['weekday'] = data['start_time'].dt.weekday + 1#生成新的一列，为星期几，weekday对应0到6，所以这里加1
    data['hour']= data['start_time'].dt.hour
    return data
    
def latitude_longitude_to_go(data,isTrain):

    tmp = data[['start_lat','start_lon']]#取出出发地经纬度
    start_geohash = []#定义一个空列表
    for t in tmp.values:#逐行遍历出发地经纬度
        start_geohash.append(str(round(t[0],5)) + '_' + str(round(t[1],5)))#保留小数点后5位，将经纬度合并
    data['startGo'] = start_geohash#生成新的一列，值为合并后的经纬度
    if isTrain:#如果是训练集，对目的地经纬度做如上处理
        tmp = data[['end_lat','end_lon']]
        end_geohash = []
        for t in tmp.values:
            end_geohash.append(str(round(t[0],5))+ '_' + str(round(t[1],5)))
        data['endGo'] = end_geohash#生成新的一列，值为合并后目的地经纬度
    return data
    
def getMostTimesCandidate(candidate):
    
    #取9月前的数据
    mostTimeCandidate = candidate[candidate['start_time']<='2018-08-30 23:59:59']
    #取'out_id','endGo','end_lat','end_lon','weekday'列
    mostTimeCandidate = mostTimeCandidate[['out_id','endGo','end_lat','end_lon','weekday']]
    #按车辆id、目的地、星期分组（agg）统计目的地出现的次数,放入生成的mostCandidateCount列
    mostTimeCandidate_7 = mostTimeCandidate.groupby(['out_id','endGo','weekday'],as_index=False)['endGo'].agg({'mostCandidateCount':'count'})
    #按出现次数和车辆id降序排列
    mostTimeCandidate_7.sort_values(['mostCandidateCount','out_id'],inplace=True,ascending=False)
    #按车辆id和星期分组，取去的最多的7条数据
    mostTimeCandidate_7 = mostTimeCandidate_7.groupby(['out_id','weekday']).head(7)

    return mostTimeCandidate_7
    
def geoHashToLatLoc(data):
    #取出目的地
    tmp = data[['endGo']]
    #设置空列表，预测的目的地纬度
    predict_end_lat = []
    #设置空列表，预测的目的地经度
    predict_end_lon = []
    #逐行遍历
    for i in tmp.values:
        #取出每行的经纬度，放入列表
        lats, lons = str(i[0]).split('_')
        predict_end_lat.append(lats)
        predict_end_lon.append(lons)
    #生成新的列
    data['predict_end_lat'] = predict_end_lat
    data['predict_end_lon'] = predict_end_lon
    return data
    
def calcGeoHasBetween(go1,go2):
    latA, lonA = str(go1).split('_')
    latB, lonB = str(go2).split('_')
    distence = getDistance(float(latA), float(lonA), float(latB), float(lonB))
    return distence
    
def calcGeoHasBetweenMain(data):
    distance = []
    tmp = data[['endGo','startGo']]
    for i in tmp.values:
        distance.append(calcGeoHasBetween(i[0],i[1]) / 1000 )
    data['distance'] = distance
    return data