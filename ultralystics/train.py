from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.pt")  # load a pretrained model. change this to the model you want to use, n, s, l etc. (only ending)

# Train the model
results = model.train(data="./yaml/mytrainingset.yaml", imgsz=640, batch=8, epochs=50, plots=False)
