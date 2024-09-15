from funcs import *

detection = detect()
#detection.static(["note.jpg"])
#detection.stream(camera=0  ,gui=True,cls=[3])
detection.stream(camera=0,cls=[3])