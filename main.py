import cv2
import numpy as np
import chess

if __name__ == "__main__":
    cap = cv2.VideoCapture(4)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    cv2.namedWindow("board", cv2.WINDOW_NORMAL)

    OFFSET_X = 5

    while True:
        ok, img = cap.read()
        cv2.imshow("board", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            exit()
        elif key == ord('s'):
            break

    src = np.load("corners.npy")
    dst = np.float32([[0, 0], [800, 0], [800, 800], [0, 800]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (800, 800))
    for i in range(9):
        x = i * 100
        y = i * 100
        cv2.line(warped, (x, 0), (x, 800), (0, 0, 255), 2)
        cv2.line(warped, (0, y), (800, y), (0, 0, 255), 2)

    for i in range(8):
        for j in range(8):
            sq = warped[i*100:(i+1)*100, j*100:(j+1)*100]
            crop = sq[20:80, 20:80]
            sq_index = chess.square(7-j, i)
            name = chess.square_name(sq_index)
            b, g, r = crop.mean(axis=(0,1))
            color_str = '(' + str(int(b)) + ', ' + str(int(g)) + ', ' + str(int(r)) + ')'
            cv2.putText(warped, color_str, ((j*100)+OFFSET_X, (i*100)+60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

    cv2.imwrite("images/test.png", warped)

    cv2.destroyAllWindows()

    cap.release()
