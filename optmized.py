from ultralytics import YOLO
import another
import numpy as np
from math import ceil

class Detect:
    def __init__(self, model: str = "best.pt", conf: float = 0.8) -> None:
        '''
        Initialize detection
        '''
        self.conf = conf
        self.model = YOLO(model)
        self.ClassInt: list[int] = [i for i in range(len(self.model.names))]

    def __CamInit__(self, cam_id: int, resolution: tuple[int, int] = (480, 640)):
        '''
        Initialize camera
        '''
        cam = another.VideoCapture(cam_id)
        cam.set(another.CAP_PROP_FRAME_WIDTH, resolution[0])
        cam.set(another.CAP_PROP_FRAME_HEIGHT, resolution[1])
        return cam

    def static(self, imgs: list[str], save: bool = True):
        '''
        Detect object from static images
        '''
        for img in imgs:
            self.model(img, save=save, conf=self.conf)

    def stream(self, camera: int = 0, resolution: tuple[int, int] = (480, 640), cls: list[int] = None) -> None:
        '''
        Detect objects from webcam stream
        '''
        cam = self.__CamInit__(camera, resolution)
        font = another.FONT_HERSHEY_SIMPLEX
        fontscale = 1
        color = (255, 255, 0)
        thickness = 2

        try:
            while True:
                success, img = cam.read()
 # If the frame is not captured successfully, break the loop
                
                # Perform YOLO inference on the current frame
                results = self.model(img, stream=True, conf=self.conf, classes=cls)

                # Skip processing if no detections are made
                for res in results:
                    if not res.boxes:
                        continue

                    # Get the nearest detection based on box center
                    boxes = res.boxes
                    box_centers = (boxes.xyxy[:, 0] + boxes.xyxy[:, 2]) / 2
                    distances = np.abs(box_centers - resolution[1] / 2)
                    nearest_idx = np.argmin(distances.cpu().numpy())

                    # Draw rectangles and labels on detected objects
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        if i == nearest_idx:
                            # Check if the nearest box is centered
                            if (resolution[1] / 2 + 10) >= box_centers[i] >= (resolution[1] / 2 - 10):
                                # Nearest box is centered
                                another.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 3)
                            else:
                                # Nearest box is not centered
                                another.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                        else:
                            # Non-nearest boxes
                            another.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

                        # Display detection details
                        another.putText(img, f"Detected {len(boxes)} item(s)", (10, 23), font, fontscale, color, thickness)
                        another.putText(img, f"{ceil(box.conf[0].item() * 100)}%  {x1}x{y1}, {x2}x{y2}", 
                                    (x1, y1), font, fontscale, color, thickness)

                another.imshow("CamDetected", img)

                # Break the loop on 'q' key
                if another.waitKey(1) == ord('q'):
                    break
        finally:
            # Ensure camera and windows are released properly
            cam.release()
            another.destroyAllWindows()

if __name__ == "__main__":
    Detect().stream(camera=0, cls=[3])
