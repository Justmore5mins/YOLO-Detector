from ultralytics import YOLO
import cv2
import numpy as np
from math import ceil
from time import perf_counter

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

class OpenVino:
    def __init__(self, model_path, input_shape=(640, 640), device="MYRIAD", conf_threshold=0.5):
        from openvino.runtime import Core
        
        self.core = Core()
        self.model_path = model_path
        self.input_shape = input_shape
        self.device = device
        self.conf_threshold = conf_threshold

        # Load and compile the OpenVINO model
        self.compiled_model = self.core.compile_model(model_path, device)
        self.input_blob = next(iter(self.compiled_model.inputs))

    def preprocess(self, frame):
        resized_frame = cv2.resize(frame, self.input_shape)
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        resized_frame = resized_frame / 255.0
        input_data = np.transpose(resized_frame, (2, 0, 1))  # Channels first
        input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
        return input_data.astype(np.float32)
 
    def postprocess(self, frame, detections):
        height, width = frame.shape[:2]
        
        # Ensure that detections are in the expected shape
        if detections.ndim == 1:
            detections = detections.reshape(1, -1)  # Reshape for single detection case
        
        # Debug: print the raw detections shape
        print("Raw Detections Shape:", detections.shape)  

        valid_detections = []  # List to store valid detections

        for detection in detections:
            # Check if the detection is valid; confidence is usually at index 4
            confidence = detection[4]
            if confidence < self.conf_threshold:
                continue  # Skip low confidence detections

            # Valid detection, so let's store it for later processing
            valid_detections.append(detection)

            # Extract bounding box information
            x_center, y_center, box_width, box_height = detection[:4]
            x1 = int((x_center - box_width / 2) * width)
            y1 = int((y_center - box_height / 2) * height)
            x2 = int((x_center + box_width / 2) * width)
            y2 = int((y_center + box_height / 2) * height)

            # Draw the bounding box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box for detected items

            # Display detection details on the frame
            class_id = int(detection[5])  # Get class ID
            label = f"Class {class_id}, Conf: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the total number of valid detected items
        cv2.putText(frame, f"Detected {len(valid_detections)} item(s)", (10, 23), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        for something in valid_detections:
            print(something)
        return frame

    def run_inference(self, frame):
        preprocessed_frame = self.preprocess(frame)
        infer_request = self.compiled_model.create_infer_request()
        infer_request.infer({self.input_blob: preprocessed_frame})

        output = infer_request.get_output_tensor().data

        # Check the output shape and dimensions
        print("Inference Output Shape:", output)
        print(perf_counter())

        # Assuming output is in the format of (num_detections, 6) or more (x_center, y_center, width, height, confidence, class_probs)
        detections = output[0]  # Adjust according to the actual output structure
        return self.postprocess(frame, detections)


    def detect_from_cam(self, source=0):
        cap = cv2.VideoCapture(source)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_with_boxes = self.run_inference(frame)
            cv2.imshow("YOLOv8 Real-Time Object Detection", frame_with_boxes)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":  
    Detect().stream(camera=0, cls=[3])