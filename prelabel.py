import os
import cv2
from ultralytics import YOLO

# Play a move, press space: saves the photo + the model's boxes as a YOLO label.
# Upload the data/ folder to Roboflow (photos + matching .txt labels + data.yaml) and fix classes there.
MODEL = "models/v2.pt"
PREFIX = "b2_img_"  # change per batch so names never clash with earlier photos
# Stable camera path (works even if /dev/videoN numbering changes)
CAMERA = "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._onn_4K_Webcam_SN0001-video-index0"

if __name__ == "__main__":
    model = YOLO(MODEL)

    os.makedirs("data", exist_ok=True)
    # Roboflow reads class names from this; order must match the model (0 = dark, 1 = white)
    with open("data/data.yaml", "w") as f:
        f.write("nc: 2\nnames: ['dark_piece', 'white_piece']\n")

    # Continue numbering after any photos already in the folder, so nothing gets overwritten
    count = len([f for f in os.listdir("data") if f.startswith(PREFIX) and f.endswith(".png")]) + 1

    cap = cv2.VideoCapture(CAMERA, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    cv2.namedWindow("board", cv2.WINDOW_NORMAL)
    cv2.namedWindow("detections", cv2.WINDOW_NORMAL)
    print("space = save photo + labels, q = quit")

    while True:
        ok, img = cap.read()
        if not ok:
            print("Camera read failed")
            break

        cv2.imshow("board", img)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break
        elif key == ord(' '):
            name = f"{PREFIX}{count:03d}"
            # Save the clean frame (nothing drawn on it)
            cv2.imwrite(f"data/{name}.png", img)

            # conf 0.4: an extra box is quicker to delete in Roboflow than a missed one is to draw
            r = model(img, imgsz=1280, conf=0.4, verbose=False)[0]  # type: ignore
            r.save_txt(f"data/{name}.txt")  # class x_center y_center w h, normalized

            # Preview on a copy so you can spot problems right away
            display = img.copy()
            cls = r.boxes.cls.cpu().numpy()  # type: ignore
            for box, c in zip(r.boxes.xyxy.cpu().numpy(), cls):  # type: ignore
                x1, y1, x2, y2 = box
                color = (0, 0, 255) if int(c) == 0 else (255, 0, 0)
                cv2.rectangle(display, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.imshow("detections", display)

            dark = int((cls == 0).sum())
            print(f"{name}: {dark} dark, {len(cls) - dark} white")
            count += 1

    cap.release()
    cv2.destroyAllWindows()
