from detect import *

#Detect(conf=0.1).static(["imgs/img2.png"])
Detect().stream(camera=0, cls=[3])