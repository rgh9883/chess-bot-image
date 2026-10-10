import cv2
import numpy
import chess

if __name__ == "__main__":
    cap = cv2.VideoCapture(4)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    warped = cv2.imread("warped.png")
    if warped is None:
        print("Can't find Image")
        exit()
    for i in range(8):
        for j in range(8):
            sq = warped[i*100:(i+1)*100, j*100:(j+1)*100]
            crop = sq[20:80, 20:80]
            sq_index = chess.square(7-j, i)
            name = chess.square_name(sq_index)
            cv2.putText(warped, name, ((j*100)+30, (i*100)+60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imwrite("board_label.png", warped)

    cv2.destroyAllWindows()

    cap.release()
