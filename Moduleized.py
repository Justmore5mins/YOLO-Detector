from detect import OpenVino,Detect

#OpenVino("FRC_openvino_model/FRC.xml", device_name="CPU").stream(camera=0, gui=True)
Detect("FRC.pt").stream(camera=0, cls=[3], gui=True)