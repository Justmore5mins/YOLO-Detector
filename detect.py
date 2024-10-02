from ultralytics import YOLO
import cv2
import numpy as np
from math import ceil

class Detect:
    def __init__(self, model: str = "best.pt", conf: float = 0.8) -> None:
        '''
        init detection
        OpenVino Support Natively?
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

    def stream(self, camera: int = 0, resolution: tuple[int, int] = (480, 640), cls: list[int] = None,gui:bool=True) -> None:
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

            cv2.imshow("CamDetected", img) if gui else None

            # Break the loop on 'q' key
            if cv2.waitKey(1) == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()
        
class OpenVino():
    def __init__(self, model_xml: str = "yolov8.xml", conf: float = 0.8, device_name: str = "MYRIAD") -> None:
        '''
        Initialize detection with OpenVINO
        '''
        from openvino.runtime import Core
        self.conf = conf
        
        # Initialize OpenVINO runtime and load model
        self.ie = Core()
        self.model = self.ie.read_model(model=model_xml,weights=model_xml.replace(".xml", ".bin"))
        self.compiled_model = self.ie.compile_model(model=self.model, device_name=device_name)
        
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

    def __CamInit__(self, cam_id: int, resolution: tuple[int, int] = (480, 640)):
        '''
        Initialize camera
        '''
        cam = cv2.VideoCapture(cam_id)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        return cam

    def preprocess(self, img):
        '''
        Preprocess the image before inference (resize, normalize, etc.)
        '''
        input_shape = self.input_layer.shape  # Example: [1, 3, 640, 640]
        img_resized = cv2.resize(img, (input_shape[3], input_shape[2]))
        img_preprocessed = img_resized.transpose(2, 0, 1).reshape(1, 3, input_shape[2], input_shape[3])
        return img_preprocessed

    def stream(self, camera: int = 0, resolution: tuple[int, int] = (480, 640), gui: bool = True) -> None:
        '''
        Detect object from webcam using OpenVINO inference
        '''
        cam = self.__CamInit__(camera, resolution)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontscale = 1
        color = (255, 255, 0)
        thickness = 2

        while True:
            success, img = cam.read()
            if not success:
                break

            # Preprocess image for inference
            input_data = self.preprocess(img)

            # Perform inference using OpenVINO
            result = self.compiled_model([input_data])[self.output_layer]

            # Process the result (assuming result contains bounding boxes and scores)
            # Here, you would decode the result based on your model's output format
            # Example: bounding boxes, class labels, confidence scores, etc.
            boxes, scores, labels = self.decode_result(result, img.shape)

            for box, conf, labels in zip(boxes, scores, labels):
                if conf >= self.conf:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 3)
                    cv2.putText(img, f"Conf: {conf:.2f}", (x1, y1 - 10), font, fontscale, color, thickness)
                    cv2.putText(img, f"Label: {labels}", (x1, y1 - 30), font, fontscale, color, thickness)



            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                conf_score = scores[i]

                if conf_score >= self.conf:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 3)
                    cv2.putText(img, f"Conf: {conf_score:.2f}", (x1, y1 - 10), font, fontscale, color, thickness)

            if gui:
                cv2.imshow("CamDetected", img)

            # Break the loop on 'q' key
            if cv2.waitKey(1) == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()

    def decode_result(self, result, img_shape):
        '''
        Decode the inference result to extract bounding boxes, confidence scores, and class labels
        '''
        # This function should map the OpenVINO output back to bounding boxes and scores
        # You need to implement decoding logic based on your model’s structure
        boxes = []  # Extract bounding boxes
        scores = []  # Extract confidence scores
        labels = []  # Extract class labels
        return boxes, scores, labels


if __name__ == "__main__":  
    Detect().stream(camera=0, cls=[3])