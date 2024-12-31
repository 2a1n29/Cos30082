from ultralytics import YOLO
model = YOLO("yolov10n.pt")
results = model("image.jpg")
results[0].show()