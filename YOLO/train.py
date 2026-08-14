from ultralytics import YOLO

def main():
    model = YOLO("models/yolo11m.pt")

    model.train(
        data="./dataset/data.yaml",
        imgsz=640,
        epochs=100,
        batch=16,
        device=0,
    )

if __name__ == "__main__":
    main()