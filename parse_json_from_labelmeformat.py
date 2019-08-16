#! -*- coding=utf-8 -*-
import os, sys
import json
import cv2
import numpy as np


def read_jsons():
    with open('train.txt') as fv:
        for line in fv:
            line = line.strip()
            im = cv2.imread('./images/'+line+'.jpg')
            
            # read json file
            f_json = open('./jsons/' + line + '.json')
            signanno = json.load(f_json)
            imageH = signanno['imageHeight']
            imageW = signanno['imageWidth']
            imagename = signanno['imagePath']
            print(imagename,imageW,imageH)
            
            signshapes = signanno['shapes']
            font = cv2.FONT_HERSHEY_SIMPLEX

                
            mask = np.zeros((int(imageH), int(imageW), 3))
            for i in range(0, len(signshapes)):
                label = signshapes[i]['label']
                #print label
                pts = signshapes[i]['points']
                cv2.putText(im, label, (int(pts[0][0]), int(pts[0][1])-10), font, 0.5, (0,0,255), 2)
           
                x = []
                y = []
                for p in pts:
                    x.append(int(p[0]))
                    y.append(int(p[1]))
                minx, miny = min(x), min(y)
                maxx, maxy = max(x), max(y)
                cv2.rectangle(im, (minx, miny), (maxx, maxy), (255,0,0),1)
               
                polys = np.array(pts, np.int32)
                if label == 'traffic4' or label == 'traffic-4-occ-partially' or label == 'traffic-4-occ-largely':
                    cv2.fillConvexPoly(mask, polys, (0,255,255)) 
                if label == 'traffic4-back':
                    cv2.fillConvexPoly(mask, polys, (0,0,255)) 
                if label == 'traffic3':
                    cv2.fillConvexPoly(mask, polys, (255,0,0)) 
                if label == 'traffic3-back':
                    cv2.fillConvexPoly(mask, polys, (0,255,0)) 
                
                for p in pts:
                    # print p[0],p[1]
                    cv2.circle(im, (int(p[0]), int(p[1])), 2, (0,255,255),2)
                
            cv2.namedWindow("1",0)
            cv2.imshow("1",im)
            cv2.waitKey(0)

#cv2.namedWindow("2",0)
#cv2.imshow("2", mask)
#cv2.waitKey(10)

        

if __name__ == "__main__":
    print('start...')
    read_jsons()
