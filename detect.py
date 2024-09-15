from ultralytics import YOLO

class detect:
    def __init__(self,model:str = "best.pt", conf:float|None = 0.8) -> None:
        '''
        init detection
        '''
        self.conf = conf
        self.model = YOLO(model)
        self.ClassInt:list[int] = [i for i in range(len(self.model.names))]
     
    def __CamInit__(self,cam:int,resoultion:tuple[int,int] = (480,640)):
        '''
        init camera
        '''
        from cv2 import VideoCapture
        cam = VideoCapture(cam)
        cam.set(3,resoultion[0])
        cam.set(4,resoultion[1])
        return cam
        
    def static(self,imgs:list[str],save:bool = True):
        '''
        detect object from normal image
        '''
        for img in imgs:
            self.model(img,save=save,conf=self.conf)
    
    def stream(self,camera:int = 0,resolution:tuple[int,int] = (480,640), cls:list[str] = None) -> None:
        '''
        detect image from webcam
        '''
        cam = self.__CamInit__(camera,resolution)
        from cv2 import waitKey,destroyAllWindows,rectangle,FONT_HERSHEY_SIMPLEX,putText,imshow
        from math import ceil
        while True:
            sccess, img = cam.read()
            results = self.model(img,stream=True,conf=self.conf,classes=cls)

            #get the nearest note if more than one note detected
            
            for res in results:
                if len(res.boxes) > 1:
                    boxlist:list[int] = []
                    for box in res.boxes:
                        boxlist.append((box.xyxy[0][0]-box.xyxy[0][2]))
                    nearest = boxlist.index(min([place for place in boxlist]))
                elif len(res.boxes) == 1:
                    nearest = 0
                
                boxes = res.boxes
                i:int = 0
                for box in boxes:
                    x1,y1,x2,y2 = box.xyxy[0]
                    x1,x2,y1,y2 = int(x1), int(x2),int(y1),int(y2)
                    if i == nearest and (resolution[1]/2 + 10) >= ((x1+x2)/2) >= (resolution[1]/2 - 10):
                        rectangle(img,(x1,y1),(x2,y2),(255,255,0),3) #color bgr
                    elif i == nearest and not (resolution[1]/2 + 10) >= ((x1+x2)/2) >= (resolution[1]/2 - 10):
                        rectangle(img,(x1,y1),(x2,y2),(255,0,0),3)
                    else:
                        rectangle(img,(x1,y1),(x2,y2),(0,0,255),3)
                    #gyi
                    org = [x1,y1]
                    font = FONT_HERSHEY_SIMPLEX
                    fontscale = 1
                    color = (255,255,0)
                    thickness = 2
                    putText(img,f"detected {len(boxes)} item",[0,23],font,fontscale,color,thickness)
                    putText(img,f"{ceil(box.conf[0]*100)}%  {x1}x{y1}, {x2}x{y2}",org,font,fontscale,color,thickness)
                    imshow("CamDetected",img)  
                            
                    i += 1
            
            if waitKey(1) == ord('q'):
                cam.release()
                destroyAllWindows()

if __name__ == "__main__":
    detect().stream(camera=0,cls=[3])