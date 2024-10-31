from os import system

from detect import OpenVino,Detect

#enVino(model_path="FRC_openvino_model/FRC.xml",device="CPU",conf_threshold=0.0).detect_from_cam()
Detect("FRC.pt").stream(camera=0, gui=True)