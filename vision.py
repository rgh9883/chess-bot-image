"""Camera + piece detection + mapping each piece to a chess square.

Shared by main.py (the game) and any test scripts.
"""
import chess
import cv2
import numpy as np
from ultralytics import YOLO

MODEL = "models/v3.pt"
CORNERS = "corners.npy"
# Stable camera path (works even if /dev/videoN numbering changes)
CAMERA = "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._onn_4K_Webcam_SN0001-video-index0"
IMGSZ = 1280      # same size the model was trained at
CONF = 0.5        # ignore boxes the model is less sure about
BASE_OFFSET = 10  # px above the box bottom = center of the piece's base

DARK, WHITE = 0, 1  # model class ids (from dataset/data.yaml)


def open_camera():
    cap = cv2.VideoCapture(CAMERA, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))  # FOURCC before size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    return cap


def load_homography():
    """Matrix that maps a camera pixel onto the flat 800x800 board (100 px per square)."""
    src = np.load(CORNERS).astype(np.float32)
    dst = np.float32([[0, 0], [800, 0], [800, 800], [0, 800]])
    return cv2.getPerspectiveTransform(src, dst)


def draw_board_outline(img):
    """Red outline of the calibrated board. If it doesn't sit on the grid, recalibrate."""
    corners = np.load(CORNERS).astype(np.int32)
    cv2.polylines(img, [corners], True, (0, 0, 255), 2)


def detect(model, M, img):
    """Return ({square: chess.WHITE/chess.BLACK}, annotated copy of img).

    If two boxes land on one square (YOLO26 can output a white and a dark box for the
    same piece), the more confident one wins.
    """
    r = model(img, imgsz=IMGSZ, conf=CONF, verbose=False)[0]  # type: ignore
    xyxy = r.boxes.xyxy.cpu().numpy()  # type: ignore
    cls = r.boxes.cls.cpu().numpy()  # type: ignore
    conf = r.boxes.conf.cpu().numpy()  # type: ignore

    display = img.copy()
    draw_board_outline(display)
    found = {}  # square -> (color, confidence)

    for (x1, y1, x2, y2), c, cf in zip(xyxy, cls, conf):
        is_white = int(c) == WHITE
        color = (255, 0, 0) if is_white else (0, 0, 255)
        bx, by = (x1 + x2) / 2, y2 - BASE_OFFSET

        x, y = cv2.perspectiveTransform(np.float32([[[bx, by]]]), M)[0][0]
        cv2.rectangle(display, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.circle(display, (int(bx), int(by)), 5, color, -1)
        if not (0 <= x < 800 and 0 <= y < 800):
            continue  # off the board, e.g. a captured piece at the edge of the frame

        col, row = int(x // 100), int(y // 100)
        sq = chess.square(7 - col, row)  # white at the top of the image, a1 top-right
        cv2.putText(display, f"{chess.square_name(sq)} {cf:.2f}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if sq not in found or cf > found[sq][1]:
            found[sq] = (chess.WHITE if is_white else chess.BLACK, cf)

    return {sq: side for sq, (side, _) in found.items()}, display


def load():
    return YOLO(MODEL), load_homography()
