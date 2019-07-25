#! -*- coding=utf-8 -*-
import os, sys
import matplotlib.pyplot as plt
from PIL import Image

def PIL_show(filepath):

    plt.figure(figsize=(4, 4))
    plt.ion()  # 打开交互模式
    plt.axis('off')  # 不需要坐标轴

    with open('/cvg_1/石亮亮合作数据_日本/测试数据/标牌/image/imagelists.txt') as f:
        for line in f:
            line = line.strip()
            imagename = line
            img = Image.open('/cvg_1/石亮亮合作数据_日本/测试数据/标牌/image/'+imagename)  # 打开图片，返回PIL image对象
            plt.imshow(img)

            mngr = plt.get_current_fig_manager()
            mngr.window.wm_geometry("+380+310")  # 调整窗口在屏幕上弹出的位置
            plt.pause(0.01)  # 该句显示图片1秒
            plt.ioff()  # 显示完后一定要配合使用plt.ioff()关闭交互模式，否则可能出奇怪的问题

    # plt.clf()  # 清空图片
    plt.close()  # 清空窗口



import cv2
import numpy as np

def show_merge_res():
    with open('/media/mahj/Elements/SD/imagelists.txt') as f:
        for line in f:
            print line
            line = line.strip()
            img1 = cv2.imread('/media/mahj/Elements/SD/map分割结果/mask/' + line)
            img1 = cv2.resize(img1, (4096/4, 2168/4))
            if img1 is None:
                continue

            subname = line.split('.')
            imagename = subname[0] + '.' + subname[1] + '.jpg'
            print imagename
            img2 = cv2.imread('/media/mahj/Elements/SD/新分割模型结果/' + imagename)

            if img2 is None:
                continue
            img2 = cv2.resize(img2, (4096 / 4, 2168 / 4))
            img3 = cv2.imread('/media/mahj/Elements/SD/虚实线结果/' + imagename)
            img3 = cv2.resize(img3, (4096 / 4, 2168 / 4))
            img4 = cv2.imread('/media/mahj/Elements/SD/curb_pole结果/' + imagename)
            img4 = cv2.resize(img4, (4096 / 4, 2168 / 4))

            im = np.zeros((2168*2/4, 4096*2/4, 3), dtype=np.uint8)
            im[0:2168/4, 0:4096/4, :] = img1
            im[0:2168/4, 4096/4:8192/4, :] = img2
            im[2168/4:2168*2/4, 0:4096/4, :] = img3
            im[2168/4:2168 * 2/4, 4096/4:8192/4, :] = img4

            cv2.namedWindow('1', 0)
            cv2.imshow('1', im)
            cv2.waitKey(10)
            cv2.imwrite('/media/mahj/Elements/SD/merge_resize结果/' + imagename, im)

if __name__ == "__main__":
    # PIL_show('')
    show_merge_res()

