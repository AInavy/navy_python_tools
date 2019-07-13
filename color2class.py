#coding=utf-8

import cv2
import numpy  as np
from collections import namedtuple
from PIL import Image, ImageDraw, ImageFont

'''Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  '道路(road)'                 ,  7 ,        0 , '道路'            , 1       , False        , False        , (128, 64,128) ),
    Label(  '人行道(sidewalk)'             ,  8 ,        1 , '人行道'            , 1       , False        , False        , (244, 35,232) ),
    Label(  '建筑(building)'             , 11 ,        2 , '建筑'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  '墙(wall)'                 , 12 ,        3 , '墙'    , 2       , False        , False        , (102,102,156) ),
    Label(  '栅栏(fence)'                , 13 ,        4 , '栅栏'    , 2       , False        , False        , (190,153,153) ),
    Label(  '杆(pole)'                 , 17 ,        5 , '杆'          , 3       , False        , False        , (153,153,153) ),
    Label(  '交通灯(traffic light)'        , 19 ,        6 , '交通灯'          , 3       , False        , False        , (250,170, 30) ),
    Label(  '标识牌(traffic sign)'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  '植物(vegetation)'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  '地形(terrain)'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  '天空(sky)'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  '人(person)'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  '骑手(rider)'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  '汽车(car)'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  '卡车(truck)'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  '公交车(bus)'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  '火车(train)'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  '摩托车(motorcycle)'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  '自行车(bicycle)'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
]''
Label = namedtuple( 'Label' , ['name','trainId', 'color',] )


labels = [
	#	trainId		color
	Label('道路(road)', 0,	(128,  64, 128)),
	Label('杆(pole)',  1,	(244,  35, 232)),
	Label('植物(vegetation)',  2,	(107, 142,  35)),
	Label('井盖(manhole)', 3,	(  0,   0, 200)),
	Label('其他(other)', 4,	(255, 255, 255)),
]'''
Label = namedtuple( 'Label' , ['name','id','trainId','color'])


labels = [
    #       name                    id    trainId     color
    Label(  'bird'               ,  0 ,      0 , (  165,  42, 42) ),
    Label(  'ground animal'      ,  1 ,      1 , (    0, 192,   0) ),
    Label(  'curb'               ,  2 ,      2 , (  196, 196, 196) ),
    Label(  'fence'              ,  3 ,      3 , (  190, 153, 153) ),
    Label(  'guard rail'         ,  4 ,      4 , (  180, 165, 180) ),
    Label(  'barrier'            ,  5 ,      5 , (   90, 120, 150) ),
    Label(  'wall'               ,  6 ,      6 , (  102, 102, 156) ),
    Label(  'bike lane'          ,  7 ,      7 , (  128,  64, 255) ),
    Label(  'plain'              ,  8 ,      8 , (  140, 140, 200) ),
    Label(  'curb cut'           ,  9 ,      9 , (  170, 170, 170) ),
    Label(  'parking'            , 10 ,     10 , (  250, 170, 160) ),
    Label(  'pedestrian Area'    , 11 ,     11 , (   96,  96,  96) ),
    Label(  'rail track'         , 12 ,     12 , (  230, 150, 140) ),
    Label(  'road'               , 13 ,     13 , (  128,  64, 128) ),
    Label(  'service lane'       , 14 ,     14 , (  110, 110, 110) ),
    Label(  'sidewalk'           , 15 ,     15 , (  244,  35, 232) ),
    Label(  'bridge'             , 16 ,     16 , (  150, 100, 100) ),
    Label(  'building'           , 17 ,     17 , (   70,  70,  70) ),
    Label(  'tunnel'             , 18 ,     18 , (  150, 120,  90) ),
    Label(  'person'             , 19 ,     19 , (  220,  20,  60) ),
    Label(  'bicyclist'          , 20 ,     20 , (  250,   0,   0) ),
    Label(  'motorcyclist'       , 21 ,     21 , (  250,   0, 100) ),
    Label(  'rider'              , 22 ,     22 , (  250,   0, 200) ),
    Label(  'zebra'              , 23 ,     23 , (  200, 128, 128) ),
    Label(  'marking'            , 24 ,     24 , (  255, 255, 255) ),
    Label(  'mountain'           , 25 ,     25 , (   64, 170,  64) ),
    Label(  'sand'               , 26 ,     26 , (  230, 160,  50) ),
    Label(  'sky'                , 27 ,     27 , (   70, 130, 180) ),
    Label(  'snow'               , 28 ,     28 , (  190, 255, 255) ),
    Label(  'terrain'            , 29 ,     29 , (  152, 251, 152) ),
    Label(  'vegetation'         , 30 ,     30 , (  107, 142,  35) ),
    Label(  'water'              , 31 ,     31 , (    0, 170,  30) ),
    Label(  'banner'             , 32 ,     32 , (  255, 255, 128) ),
    Label(  'bench'              , 33 ,     33 , (  250,   0,  30) ),
    Label(  'bike-rack'          , 34 ,     34 , (  100, 140, 180) ),
    Label(  'billboard'          , 35 ,     35 , (  220, 220, 220) ),
    Label(  'basin'              , 36 ,     36 , (  220, 128, 128) ),
    Label(  'camera'             , 37 ,     37 , (  222,  40,  40) ),
    Label(  'fire-hydrant'       , 38 ,     38 , (  100, 170,  30) ),
    Label(  'junction-box'       , 39 ,     39 , (   40,  40,  40) ),
    Label(  'mailbox'            , 40 ,     40 , (   33,  33,  33) ),
    Label(  'manhole'            , 41 ,     41 , (  100, 128, 160) ),
    Label(  'phone-booth'        , 42 ,     42 , (  142,   0,   0) ),
    Label(  'pothole'            , 43 ,     43 , (   70, 100, 150) ),
    Label(  'street-light'       , 44 ,     44 , (  210, 170, 100) ),
    Label(  'pole'               , 45 ,     45 , (  150, 150, 150) ),
    Label(  'traffic-sign-frame' , 46 ,     46 , (  128, 128, 128) ),
    Label(  'utility-pole'       , 47 ,     47 , (    0,   0,  80) ),
    Label(  'traffic-light'      , 48 ,     48 , (  250, 170,  30) ),
    Label(  'traffic-sign-back'  , 49 ,     49 , (  192, 192, 192) ),
    Label(  'traffic-sign-front' , 50 ,     50 , (  220, 220,   0) ),
    Label(  'trash-can'          , 51 ,     51 , (  140, 140,  20) ),
    Label(  'bicycle'            , 52 ,     52 , (  119,  11,  32) ),
    Label(  'boat'               , 53 ,     53 , (  150,   0, 255) ),
    Label(  'bus'                , 54 ,     54 , (    0,  60, 100) ),
    Label(  'car'                , 55 ,     55 , (    0,   0, 142) ),
    Label(  'carvan'             , 56 ,     56 , (    0,   0,  90) ),
    Label(  'motorcycle'         , 57 ,     57 , (    0,   0, 230) ),
    Label(  'on-rails'           , 58 ,     58 , (    0,  80, 100) ),
    Label(  'other-vehicle'      , 59 ,     59 , (  128,  64,  64) ),
    Label(  'trailer'            , 60 ,     60 , (    0,   0, 110) ),
    Label(  'truck'              , 61 ,     61 , (    0,   0,  70) ),
    Label(  'wheeled-slow'       , 62 ,     62 , (    0,   0, 192) ),
    Label(  'mount'              , 63 ,     63 , (   32,  32,  32) ),
    Label(  'ego-vehicle'        , 64 ,     64 , (  120,  10,  10) ),
    Label(  'unlabeled'          , 65 ,     65 , (    0,   0,   0) ),
]



img = np.zeros((200*66,1500,3),np.uint8)+255
for l in labels:
	i = l.trainId
	cat = l.name
	color = l.color
	tid = l.trainId
	font = ImageFont.truetype('NotoSansCJK-Black.ttc', 80)
	fillColor = (0,0,0)
	for j in range(3):
		img[i*200+50:i*200+150,50:500,j] = color[2-j]
		img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
		position = (600,i*200+30)
		draw = ImageDraw.Draw(img_PIL)    
		draw.text(position, str(tid)+'    '+cat, font=font, fill=fillColor)
		img = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)
		#cv2.putText(img,cat,(650,i*200+150),cv2.FONT_HERSHEY_COMPLEX,4,(0,0,0),18)
cv2.imwrite('color2calss.png',img)
		

