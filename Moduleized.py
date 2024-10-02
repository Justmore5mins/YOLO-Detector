from detect import OpenVino,Detect

#OpenVino("FRC_openvino_model/FRC.xml", device_name="CPU").stream(camera=0, gui=True)
Detect("runs/detect/train/weights/best.pt").stream(camera=0, gui=True)