from ultralytics import YOLO

# Load your trained model
model = YOLO('FRC.pt')
model.export(format="ncnn")