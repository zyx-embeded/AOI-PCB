from ultralytics import YOLO

model = YOLO("./models/best.pt")

results = model.predict(
    source="./dataset/test/images_demo",
    save=True,
    conf=0.1
)