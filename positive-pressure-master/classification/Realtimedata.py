#!/usr/bin/env python
# -*- coding: utf-8 -*-


import time
import random
import csv
import copy
import os
import argparse

import sys
import rospy
import numpy as np
import scipy.io as sio
from casia_msg_pkg.msg import Tactile, TactileXYZT

#
num_sensor_array = 16  # 传感器片数
num_sensor_unit = 16  # 每片传感器中三轴传感器个数
ans = np.zeros((3, num_sensor_array, num_sensor_unit), float)
print(ans)
#



# ######################################################订阅数据#####################################################################
def sub_tactile_data():
    # global num_sensor_unit
    rospy.init_node('Sub_tactile_data_node', anonymous=True)
    rospy.loginfo("node name: %s ", rospy.get_name())

    rate = rospy.Rate(50)  # 硬件采集的频率只有200Hz
    numtime = 0

    # 此部分根据实际接上的传感器阵列进行修改，对应的，每次修改之后要对对应的callback函数进行修改
    sub_0 = rospy.Subscriber('/tactile_data_topic1', Tactile, callback_0)
    sub_1 = rospy.Subscriber('/tactile_data_topic2', Tactile, callback_1)
    sub_2 = rospy.Subscriber('/tactile_data_topic3', Tactile, callback_2)
    sub_3 = rospy.Subscriber('/tactile_data_topic4', Tactile, callback_3)
    sub_4 = rospy.Subscriber('/tactile_data_topic5', Tactile, callback_4)
    sub_5 = rospy.Subscriber('/tactile_data_topic6', Tactile, callback_5)
    sub_6 = rospy.Subscriber('/tactile_data_topic7', Tactile, callback_6)
    sub_7 = rospy.Subscriber('/tactile_data_topic8', Tactile, callback_7)
    sub_8 = rospy.Subscriber('/tactile_data_topic9', Tactile, callback_8)
    sub_9 = rospy.Subscriber('/tactile_data_topic10', Tactile, callback_9)
    sub_10 = rospy.Subscriber('/tactile_data_topic11', Tactile, callback_10)
    sub_11 = rospy.Subscriber('/tactile_data_topic12', Tactile, callback_11)
    sub_12 = rospy.Subscriber('/tactile_data_topic13', Tactile, callback_12)
    sub_13 = rospy.Subscriber('/tactile_data_topic14', Tactile, callback_13)
    sub_15 = rospy.Subscriber('/tactile_data_topic16', Tactile, callback_15)
    sub_17 = rospy.Subscriber('/tactile_data_topic18', Tactile, callback_17)
    # sub_22 = rospy.Subscriber('/tactile_data_topic23', Tactile, callback_22)

    whole_pressure = list()
    while numtime < 200:
        # 1
        # print(ans[0][0][0], ans[1][0][0], ans[2][0][0])
        for i in range(num_sensor_unit):
            print(ans[0][0][i], ans[1][0][i], ans[2][0][i])

        # 2
        # print(ans[0][1][0], ans[1][1][0], ans[2][1][0])
        for i in range(num_sensor_unit):
            print(ans[0][1][i], ans[1][1][i], ans[2][1][i])

        # 3
        # print(ans[0][2][0], ans[1][2][0], ans[2][2][0])
        for i in range(num_sensor_unit):
            print(ans[0][2][i], ans[1][2][i], ans[2][2][i])

        # 4
        # print(ans[0][3][0], ans[1][3][0], ans[2][3][0])
        for i in range(num_sensor_unit):
            print(ans[0][3][i], ans[1][3][i], ans[2][3][i])

        # 5
        # print(ans[0][4][0], ans[1][4][0], ans[2][4][0])
        for i in range(num_sensor_unit):
            print(ans[0][4][i], ans[1][4][i], ans[2][4][i])

        # 6
        # print(ans[0][5][0], ans[1][5][0], ans[2][5][0])
        for i in range(num_sensor_unit):
            print(ans[0][5][i], ans[1][5][i], ans[2][5][i])

        # 7
        # print(ans[0][6][0], ans[1][6][0], ans[2][6][0])
        for i in range(num_sensor_unit):
            print(ans[0][6][i], ans[1][6][i], ans[2][6][i])

        # 8
        # print(ans[0][7][0], ans[1][7][0], ans[2][7][0])
        for i in range(num_sensor_unit):
            print(ans[0][7][i], ans[1][7][i], ans[2][7][i])

        # 9
        # print(ans[0][8][0], ans[1][8][0], ans[2][8][0])
        for i in range(num_sensor_unit):
            print(ans[0][8][i], ans[1][8][i], ans[2][8][i])

        # 10
        # print(ans[0][9][0], ans[1][9][0], ans[2][9][0])
        for i in range(num_sensor_unit):
            print(ans[0][9][i], ans[1][9][i], ans[2][9][i])

        # 11
        # print(ans[0][10][0], ans[1][10][0], ans[2][10][0])
        for i in range(num_sensor_unit):
            print(ans[0][10][i], ans[1][10][i], ans[2][10][i])

        # 12
        # print(ans[0][11][0], ans[1][11][0], ans[2][11][0])
        for i in range(num_sensor_unit):
            print(ans[0][11][i], ans[1][11][i], ans[2][11][i])

        # 13
        # print(ans[0][12][0], ans[1][12][0], ans[2][12][0])
        for i in range(num_sensor_unit):
            print(ans[0][12][i], ans[1][12][i], ans[2][12][i])

        # 14
        # print(ans[0][12][0], ans[1][12][0], ans[2][12][0])
        for i in range(num_sensor_unit):
            print(ans[0][12][i], ans[1][12][i], ans[2][12][i])

        # 16
        # print(ans[0][13][0], ans[1][13][0], ans[2][13][0])
        for i in range(num_sensor_unit):
            print(ans[0][13][i], ans[1][13][i], ans[2][13][i])

        # 18
        # print(ans[0][14][0], ans[1][14][0], ans[2][14][0])
        for i in range(num_sensor_unit):
            print(ans[0][14][i], ans[1][14][i], ans[2][14][i])

        # 23
        # print(ans[0][15][0], ans[1][15][0], ans[2][15][0])
        # for i in range(num_sensor_unit):
        # print(ans[0][15][i], ans[1][15][i], ans[2][15][i])

        # add to pressure list
        whole_pressure.append(copy.deepcopy(ans))

        numtime = numtime + 1
        time.sleep(0.1)

        rate.sleep()

    whole_pressure = np.array(whole_pressure)
    print(whole_pressure)

    return whole_pressure


# #################################################################################################################################

# ###################################################################callback函数###########################################
# 1
def callback_0(sensor_msg):
    global ans
    for i in range(num_sensor_unit):
        ans[0][0][i], ans[1][0][i], ans[2][0][i] = sensor_msg.tactile[i].x, sensor_msg.tactile[i].y, sensor_msg.tactile[
            i].z


# 2
def callback_1(sensor_msg):
    global ans
    for i in range(num_sensor_unit):
        ans[0][1][i], ans[1][1][i], ans[2][1][i] = sensor_msg.tactile[i].x, sensor_msg.tactile[i].y, sensor_msg.tactile[
            i].z


# 3
def callback_2(sensor_msg):
    global ans
    for i in range(num_sensor_unit):
        ans[0][2][i], ans[1][2][i], ans[2][2][i] = sensor_msg.tactile[i].x, sensor_msg.tactile[i].y, sensor_msg.tactile[
            i].z


# 4
def callback_3(sensor_msg):
    global ans
    for i in range(num_sensor_unit):
        ans[0][3][i], ans[1][3][i], ans[2][3][i] = sensor_msg.tactile[i].x, sensor_msg.tactile[i].y, sensor_msg.tactile[
            i].z


# 5
def callback_4(sensor_msg):
    global ans
    for i in range(num_sensor_unit):
        ans[0][4][i], ans[1][4][i], ans[2][4][i] = sensor_msg.tactile[i].x, sensor_msg.tactile[i].y, sensor_msg.tactile[
            i].z


# 6
def callback_5(sensor_msg):
    global ans
    for i in range(num_sensor_unit):
        ans[0][5][i], ans[1][5][i], ans[2][5][i] = sensor_msg.tactile[i].x, sensor_msg.tactile[i].y, sensor_msg.tactile[
            i].z


# 7
def callback_6(sensor_msg):
    global ans
    for i in range(num_sensor_unit):
        ans[0][6][i], ans[1][6][i], ans[2][6][i] = sensor_msg.tactile[i].x, sensor_msg.tactile[i].y, sensor_msg.tactile[
            i].z


# 8
def callback_7(sensor_msg):
    global ans
    for i in range(num_sensor_unit):
        ans[0][7][i], ans[1][7][i], ans[2][7][i] = sensor_msg.tactile[i].x, sensor_msg.tactile[i].y, sensor_msg.tactile[
            i].z


# 9
def callback_8(sensor_msg):
    global ans
    for i in range(num_sensor_unit):
        ans[0][8][i], ans[1][8][i], ans[2][8][i] = sensor_msg.tactile[i].x, sensor_msg.tactile[i].y, sensor_msg.tactile[
            i].z


# 10
def callback_9(sensor_msg):
    global ans
    for i in range(num_sensor_unit):
        ans[0][9][i], ans[1][9][i], ans[2][9][i] = sensor_msg.tactile[i].x, sensor_msg.tactile[i].y, sensor_msg.tactile[
            i].z


# 11
def callback_10(sensor_msg):
    global ans
    for i in range(num_sensor_unit):
        ans[0][10][i], ans[1][10][i], ans[2][10][i] = sensor_msg.tactile[i].x, sensor_msg.tactile[i].y, \
                                                      sensor_msg.tactile[i].z


# 12
def callback_11(sensor_msg):
    global ans
    for i in range(num_sensor_unit):
        ans[0][11][i], ans[1][11][i], ans[2][11][i] = sensor_msg.tactile[i].x, sensor_msg.tactile[i].y, \
                                                      sensor_msg.tactile[i].z


# 13
def callback_12(sensor_msg):
    global ans
    for i in range(num_sensor_unit):
        ans[0][12][i], ans[1][12][i], ans[2][12][i] = sensor_msg.tactile[i].x, sensor_msg.tactile[i].y, \
                                                      sensor_msg.tactile[i].z


# 14
def callback_13(sensor_msg):
    global ans
    for i in range(num_sensor_unit):
        ans[0][12][i], ans[1][12][i], ans[2][12][i] = sensor_msg.tactile[i].x, sensor_msg.tactile[i].y, \
                                                      sensor_msg.tactile[i].z


# 16
def callback_15(sensor_msg):
    global ans
    for i in range(num_sensor_unit):
        ans[0][13][i], ans[1][13][i], ans[2][13][i] = sensor_msg.tactile[i].x, sensor_msg.tactile[i].y, \
                                                      sensor_msg.tactile[i].z


# 18
def callback_17(sensor_msg):
    global ans
    for i in range(num_sensor_unit):
        ans[0][14][i], ans[1][14][i], ans[2][14][i] = sensor_msg.tactile[i].x, sensor_msg.tactile[i].y, \
                                                      sensor_msg.tactile[i].z


# 23
# def callback_22(sensor_msg):
# global ans
# for i in range(num_sensor_unit):
# ans[0][15][i], ans[1][15][i], ans[2][15][i] = sensor_msg.tactile[i].x, sensor_msg.tactile[i].y, sensor_msg.tactile[i].z

# ######################################################################################################################################################


if __name__ == '__main__':
    pressure = sub_tactile_data()  # 订阅数据并将每次订阅到的数据保存在对应的变量中
    print(pressure)


