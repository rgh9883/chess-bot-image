import cv2
from ultralytics import YOLO

# True = live camera, False = test on a saved photo
USE_CAMERA = False
TEST_IMAGE = "data/img_1.png"

if __name__ == "__main__":
    model = YOLO("models/best.pt")

    if USE_CAMERA:
        cap = cv2.VideoCapture(4)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    else:
        test_img = cv2.imread(TEST_IMAGE)
        if test_img is None:
            print(f"Can't load {TEST_IMAGE}")
            exit()

    cv2.namedWindow("board", cv2.WINDOW_NORMAL)
    cv2.namedWindow("detections", cv2.WINDOW_NORMAL)

    while True:
        if USE_CAMERA:
            ok, img = cap.read()
            # if the read failed, stop the loop
            if not ok:
                break
        else:
            img = test_img

        cv2.imshow("board", img)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break
        elif key == ord(' '):
            # run the model on img (imgsz=1280, conf=0.5), take the first result
            res = model(img, imgsz=1280, conf=0.5)
            r = res[0]

            # pull out xyxy, cls, conf as numpy arrays
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()

            display = img.copy()
            white = 0
            dark = 0

            print(xyxy)
            print(cls)
            print(conf)
            # loop over the boxes:
            #   - pick a color based on the class (BGR, e.g. (0, 0, 255) is red)
            #   - draw the rectangle (remember int())
            #   - count how many of each class
            for box, c, cf in zip(xyxy, cls, conf):
                x1, y1, x2, y2 = box
                if int(c) == 0:
                    dark += 1
                    color = (0, 0, 255)
                else:
                    white += 1
                    color = (255, 0 , 0)
                cv2.putText(display, f"{cf:.2f}", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.rectangle(display, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # print the two counts
            print(f"White count: ${white}")
            print(f"Dark count: ${dark}")


            cv2.imshow("detections", display)

    if USE_CAMERA:
        cap.release()
    cv2.destroyAllWindows()
