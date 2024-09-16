from ultralytics import YOLO
import cv2
import numpy as np
from math import ceil

class Detect:
    def __init__(self, model: str = "best.pt", conf: float = 0.8) -> None:
        '''
        init detection
        '''
        self.conf = conf
        self.model = YOLO(model)
        self.ClassInt: list[int] = [i for i in range(len(self.model.names))]

    def __CamInit__(self, cam_id: int, resolution: tuple[int, int] = (480, 640)):
        '''
        init camera
        '''
        cam = cv2.VideoCapture(cam_id)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        return cam

    def static(self, imgs: list[str], save: bool = True):
        '''
        detect object from normal image
        '''
        for img in imgs:
            self.model(img, save=save, conf=self.conf)

    def stream(self, camera: int = 0, resolution: tuple[int, int] = (480, 640), cls: list[int] = None) -> None:
        '''
        detect image from webcam
        '''
        cam = self.__CamInit__(camera, resolution)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontscale = 1
        color = (255, 255, 0)
        thickness = 2

        while True:
            success, img = cam.read()
            
            # Perform inference with streaming on the current frame
            results = self.model(img, stream=True, conf=self.conf, classes=cls)

            for res in results:
                if not res.boxes:
                    continue  # Skip if no boxes detected

                # Vectorized calculation of distances between box centers and image center
                boxes = res.boxes
                box_centers = (boxes.xyxy[:, 0] + boxes.xyxy[:, 2]) / 2
                distances = np.abs(box_centers - resolution[1] / 2)
                nearest_idx = np.argmax(distances.cpu().numpy())

                # Draw rectangles around detected objects
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if i == nearest_idx:
                        if (resolution[1] / 2 + 10) >= box_centers[i] >= (resolution[1] / 2 - 10): #if nearest detected and centered
                            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 3)
                        else:
                            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)#if nearest detected but not centered
                    else:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)#others(multi-detected processing)

                    # Display detection details
                    cv2.putText(img, f"Detected {len(boxes)} item(s)", (10, 23), font, fontscale, color, thickness)
                    cv2.putText(img, f"{ceil(box.conf[0].item() * 100)}%  {x1}x{y1}, {x2}x{y2}", (x1, y1), font, fontscale, color, thickness)

            cv2.imshow("CamDetected", img)

            # Break the loop on 'q' key
            if cv2.waitKey(1) == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    Detect().stream(camera=0, cls=[3])