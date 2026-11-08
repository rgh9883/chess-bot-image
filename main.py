import cv2
import numpy as np
import chess

if __name__ == "__main__":
    cap = cv2.VideoCapture(4)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    cv2.namedWindow("board", cv2.WINDOW_NORMAL)

    count = 1

    while True:
        ok, img = cap.read()
        cv2.imshow("board", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            exit()
        elif key == ord(' '):
            print(count)
            cv2.imwrite(f"data/img_{count}.png", img)
            count += 1


    cv2.destroyAllWindows()

    cap.release()
