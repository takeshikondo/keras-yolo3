#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
from PIL import Image

#from yolo import YOLO, detect_video
from tiny_yolo import YOLO, detect_video
from tiny_yolo_video import detect_img_2019

def tiny_yolo_detection():
    
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        sys.exit()
        
    yolo = YOLO()
        
    while True:
        _, img = cam.read()

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img_rgb)
        
        r_image = detect_img_2019(yolo, image)
    
        r_img = np.array(r_image)
        r_img_bgr = cv2.cvtColor(r_img, cv2.COLOR_RGB2BGR)
        
        cv2.imshow('Push Enter Key.', r_img_bgr)
        
        if cv2.waitKey(1) == 13:
            break
 
    yolo.close_session()
    cam.release()
    cv2.destroyAllWindows()        
        
    
if __name__ == '__main__':
    
    tiny_yolo_detection()
     