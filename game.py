"""Game state + Stockfish. No drawing here; main.py shows whatever this holds.

States:
  "confirm"   the physical board must match `board` (start position, or after the bot's move)
  "your move" waiting for the human (white) to move
  "over"      game finished
"""
import shutil

import chess
import chess.engine

from moves import find_move, mismatches, occupancy

THINK_TIME = 0.5  # seconds Stockfish thinks per move
SKILL_LEVEL = 5   # 0 (weakest) to 20 (full strength)
EVAL_TIME = 0.2   # seconds spent on the eval bar after each move


class Game:
    def __init__(self):
        path = shutil.which("stockfish")
        if path is None:
            raise RuntimeError("Stockfish not found. Install it: sudo pacman -S stockfish")
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.level = SKILL_LEVEL
        self.engine.configure({"Skill Level": self.level})
        self.new_game()

    def new_game(self, level=None):
        """Reset the game. level: Stockfish skill 0 to 20, or None to keep the current one."""
        if level is not None:
            self.level = level
            self.engine.configure({"Skill Level": level})
        self.board = chess.Board()
        self.state = "confirm"
        self.message = "Set up the starting position, then press Confirm."
        self.last_move = None   # your last move, highlighted on the board
        self.bot_move = None    # the move you still have to play by hand, shown as an arrow
        self.bot_text = ""
        self.bot_arrows = []    # [(from, to)]: the move, plus the rook when castling
        self.bot_remove = None  # square of the piece you must take off (capture / en passant)
        self.mismatch = []      # squares where the camera disagrees during confirm
        self.eval_cp = 0        # centipawns from white's view (+ = white better)
        self.eval_text = "0.0"

    def turn_text(self):
        if self.state == "over":
            return f"Game over: {self.board.result()}"
        if self.state == "your move":
            return "Your turn (white)"
        if self.bot_move:
            return "Bot's turn (black): play the arrow move, then Confirm"
        return "Confirm the board"

    def photo(self, seen):
        """Handle one camera result. seen = {square: color} from vision.detect."""
        if self.state == "confirm":
            self.mismatch = mismatches(occupancy(self.board), seen)
            if self.mismatch:
                names = ", ".join(sorted(chess.square_name(s) for s in self.mismatch))
                self.message = f"Board doesn't match on {names} (red). Fix it, then Confirm."
            else:
                self.state = "your move"
                self.bot_move, self.bot_text, self.bot_arrows, self.bot_remove = None, "", [], None
                self.message = "Make your move, then press Confirm."

        elif self.state == "your move":
            status, move, ranked = find_move(self.board, seen)
            if status == "no change":
                self.message = "No move seen yet. Make your move, then press Confirm."
                return
            if status == "unsure":
                top = ", ".join(f"{self.board.san(m) if m else 'no change'} ({n})" for n, m in ranked[:3])
                self.message = f"Not sure which move that was: {top}. Retake."
                return

            self.message = f"You played {self.board.san(move)}."
            self.board.push(move)
            self.last_move = move
            if self.board.is_game_over():
                self.state = "over"
                self.update_eval()
                return

            reply = self.engine.play(self.board, chess.engine.Limit(time=THINK_TIME)).move
            assert reply is not None
            self.bot_text = describe(self.board, reply)
            self.bot_arrows = [(reply.from_square, reply.to_square)]
            if self.board.is_castling(reply):
                self.bot_arrows.append(castling_rook(reply))
            self.bot_remove = captured_square(self.board, reply)
            self.bot_move = reply
            self.board.push(reply)
            self.state = "over" if self.board.is_game_over() else "confirm"
            self.update_eval()

    def undo(self):
        """Take back your last move and the bot's reply."""
        if len(self.board.move_stack) < 2:
            return
        self.board.pop()
        self.board.pop()
        self.last_move = self.board.peek() if self.board.move_stack else None
        self.bot_move, self.bot_text, self.bot_arrows, self.bot_remove, self.mismatch = None, "", [], None, []
        self.state = "confirm"
        self.message = "Undone. Put the pieces back as shown, then Confirm."
        self.update_eval()

    def update_eval(self):
        score = self.engine.analyse(self.board, chess.engine.Limit(time=EVAL_TIME))["score"].white()
        self.eval_cp = score.score(mate_score=10000)
        mate = score.mate()
        self.eval_text = f"M{abs(mate)}" if mate is not None else f"{self.eval_cp / 100:+.1f}"

    def close(self):
        self.engine.quit()


def describe(board, move):
    """Plain-words instructions for playing `move` by hand (board = position before it)."""
    piece = board.piece_at(move.from_square)
    color = "white" if piece.color == chess.WHITE else "black"
    frm, to = chess.square_name(move.from_square), chess.square_name(move.to_square)
    steps = []
    captured = captured_square(board, move)
    if captured is not None:
        name = chess.piece_name(board.piece_at(captured).piece_type)
        steps.append(f"Remove your {name} on {chess.square_name(captured)}.")
    steps.append(f"Move the {color} {chess.piece_name(piece.piece_type)} from {frm} to {to}.")
    if board.is_castling(move):
        rook_from, rook_to = castling_rook(move)
        steps.append(f"Move the rook from {chess.square_name(rook_from)} to {chess.square_name(rook_to)}.")
    if move.promotion:
        steps.append(f"Replace the pawn on {to} with a queen.")
    return " ".join(steps)


def captured_square(board, move):
    """Square of the captured piece, or None. En passant captures beside the destination."""
    if board.is_en_passant(move):
        return chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
    return move.to_square if board.is_capture(move) else None


def castling_rook(move):
    rank = chess.square_rank(move.from_square)
    if chess.square_file(move.to_square) > chess.square_file(move.from_square):
        return chess.square(7, rank), chess.square(5, rank)  # kingside: h -> f
    return chess.square(0, rank), chess.square(3, rank)      # queenside: a -> d
