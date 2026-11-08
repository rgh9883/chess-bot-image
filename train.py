from ultralytics import YOLO



if __name__ == "__main__":
    model = YOLO("yolo26n.pt")
    model.train(
        data="dataset/data.yaml",
        epochs=100,
        imgsz=1280,
        batch=8,
        device=0,
    )
