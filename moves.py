"""Step 5: figure out which legal move was played from what the camera sees.

Two representations:
  board: chess.Board, the full game state (piece types, whose turn, castling rights...)
  seen:  {square: chess.WHITE/BLACK} from vision.detect, colors + locations only

Every legal move (depth 1) is pushed to get the colored occupancy it would produce, and
compared to `seen`. Each disagreeing square counts 1 (missing, extra, or wrong color).
The lowest count wins, if it's good enough and clearly ahead of the runner-up.
"""
import chess

MAX_ERRORS = 2  # best candidate may disagree with the camera on at most this many squares
MARGIN = 2      # and must beat the runner-up by at least this many


def occupancy(board):
    return {sq: p.color for sq, p in board.piece_map().items()}


def mismatches(expected, seen):
    """Squares where the expected colored occupancy and the camera disagree."""
    return [sq for sq in set(expected) | set(seen) if expected.get(sq) != seen.get(sq)]


def rank_moves(board, seen):
    """[(error_count, move), ...] best first. move None = nothing changed."""
    ranked = [(len(mismatches(occupancy(board), seen)), None)]
    for move in board.legal_moves:
        # The camera can't tell piece types apart, so all 4 promotions look identical: keep the queen
        if move.promotion not in (None, chess.QUEEN):
            continue
        board.push(move)
        ranked.append((len(mismatches(occupancy(board), seen)), move))
        board.pop()
    ranked.sort(key=lambda c: c[0])
    return ranked


def find_move(board, seen):
    """Returns (status, move, ranked). status: "move", "no change", or "unsure"."""
    ranked = rank_moves(board, seen)
    best_errors, best = ranked[0]
    runner_up = ranked[1][0] if len(ranked) > 1 else best_errors + MARGIN
    if best_errors <= MAX_ERRORS and runner_up - best_errors >= MARGIN:
        return ("no change" if best is None else "move"), best, ranked
    return "unsure", None, ranked


def confirm(board, seen):
    """Check the physical board matches `board` exactly (e.g. after the bot's move is played).
    Returns the names of disagreeing squares; empty list = confirmed."""
    return sorted(chess.square_name(sq) for sq in mismatches(occupancy(board), seen))


if __name__ == "__main__":
    # Step 5 test: play both sides by hand.
    # space = detect the move for the side to move, c = confirm the board matches, u = undo, q = quit
    import cv2
    import vision

    model, M = vision.load()
    cap = vision.open_camera()
    board = chess.Board()
    cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
    print("space = detect move, c = confirm position, u = undo, q = quit\n")
    print(board, "\n")

    while True:
        ok, img = cap.read()
        if not ok:
            print("Camera read failed")
            break
        preview = img.copy()
        vision.draw_board_outline(preview)
        cv2.imshow("camera", preview)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break
        elif key == ord('u') and board.move_stack:
            print("Undid", board.pop(), "\n")
        elif key == ord('c'):
            seen, _ = vision.detect(model, M, img)
            bad = confirm(board, seen)
            print("Confirmed" if not bad else f"Board doesn't match on: {', '.join(bad)}", "\n")
        elif key == ord(' '):
            seen, _ = vision.detect(model, M, img)
            status, move, ranked = find_move(board, seen)
            top = ", ".join(f"{board.san(m) if m else 'no change'}={n}" for n, m in ranked[:3])
            if status == "move":
                print(f"{'White' if board.turn else 'Black'} played {board.san(move)}   (errors: {top})")
                board.push(move)
                print(board, "\n")
            elif status == "no change":
                print(f"No move seen yet   (errors: {top})\n")
            else:
                print(f"Unsure, retake the photo   (errors: {top})\n")

    cap.release()
    cv2.destroyAllWindows()
