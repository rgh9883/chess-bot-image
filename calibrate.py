import cv2
import numpy as np


points = []

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))


if __name__ == "__main__":
    cap = cv2.VideoCapture(4)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    cv2.namedWindow("board", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("board", on_click)

    ok, img = cap.read()
    if img is None:
        print("Can't find Image")
        exit()

    display = img.copy()
    while len(points) < 4:
        cv2.imshow("board", display)
        for x, y in points:
            cv2.circle(display, (x, y), 8, (0, 0, 255), -1)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyWindow("board")
    if len(points) != 4:
        exit()
    src = np.float32(points)
    dst = np.float32([[0, 0], [800, 0], [800, 800], [0, 800]])
    M = cv2.getPerspectiveTransform(src, dst)
    np.save("corners.npy", src)
    warped = cv2.warpPerspective(img, M, (800, 800))
    for i in range(9):
        x = i * 100
        y = i * 100
        cv2.line(warped, (x, 0), (x, 800), (0, 0, 255), 2)
        cv2.line(warped, (0, y), (800, y), (0, 0, 255), 2)
    cv2.imwrite("warped.png", warped)
