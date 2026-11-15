"""Play against Stockfish on the physical board. You are white.

Left: live camera with whose turn it is. Right: eval bar + top-down board showing the bot's
move as an arrow. Buttons for every command; Space = Confirm.
Photo + detection + Stockfish run in a background thread so the window never freezes.
"""
import io
import threading

import chess
import chess.svg
import cv2
import numpy as np
import pygame

import vision
from game import Game

# Layout
CAM_W, CAM_H = 720, 405
SQ = 70                     # board square size, board = 560 px
BOARD = 8 * SQ
M = 20                      # margin
CAM_X, CAM_Y = M, M + 40    # camera sits under the turn banner
EVAL_X, EVAL_W = CAM_X + CAM_W + M, 24
BOARD_X, BOARD_Y = EVAL_X + EVAL_W + 8, M
WIN_W, WIN_H = BOARD_X + BOARD + M, M + BOARD + 120

# Colors (lichess-like)
LIGHT, DARK = (240, 217, 181), (181, 136, 99)
LAST_MOVE = (205, 210, 106, 140)
BOT_SQ = (90, 170, 255, 150)
REMOVE = (230, 40, 40, 170)
ARROW = (21, 120, 27, 200)
BG, TEXT, MUTED = (40, 40, 40), (235, 235, 235), (150, 150, 150)


def piece_images():
    """Lichess-style (cburnett) pieces rendered from python-chess's SVGs."""
    images = {}
    for color in chess.COLORS:
        for kind in chess.PIECE_TYPES:
            piece = chess.Piece(kind, color)
            svg = chess.svg.piece(piece, size=SQ).encode()
            images[piece.symbol()] = pygame.image.load(io.BytesIO(svg), "piece.svg").convert_alpha()
    return images


def square_rect(sq):
    """Screen rect of a square; white at the bottom."""
    return pygame.Rect(BOARD_X + chess.square_file(sq) * SQ, BOARD_Y + (7 - chess.square_rank(sq)) * SQ, SQ, SQ)


def tint(screen, sq, rgba):
    overlay = pygame.Surface((SQ, SQ), pygame.SRCALPHA)
    overlay.fill(rgba)
    screen.blit(overlay, square_rect(sq))


def draw_arrow(layer, frm, to):
    start = pygame.Vector2(square_rect(frm).center)
    end = pygame.Vector2(square_rect(to).center)
    direction = (end - start).normalize()
    tip = end - direction * SQ * 0.15
    base = tip - direction * SQ * 0.4
    side = pygame.Vector2(-direction.y, direction.x) * SQ * 0.25
    pygame.draw.line(layer, ARROW, start, base, int(SQ * 0.18))
    pygame.draw.polygon(layer, ARROW, [tip, base + side, base - side])


def draw_board(screen, game, images, font):
    board = game.board.copy()
    if game.bot_move and game.state == "confirm":
        board.pop()  # show pieces where they physically are until you play the bot's move
    for sq in chess.SQUARES:
        light = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1
        pygame.draw.rect(screen, LIGHT if light else DARK, square_rect(sq))

    if game.last_move:
        for sq in (game.last_move.from_square, game.last_move.to_square):
            tint(screen, sq, LAST_MOVE)
    if game.bot_move:
        for sq in (game.bot_move.from_square, game.bot_move.to_square):
            tint(screen, sq, BOT_SQ)
    for sq in game.mismatch:
        tint(screen, sq, REMOVE)

    for sq, piece in board.piece_map().items():
        screen.blit(images[piece.symbol()], square_rect(sq))
    if game.bot_remove is not None:  # red X: the captured piece you must take off
        r = square_rect(game.bot_remove).inflate(-SQ * 0.3, -SQ * 0.3)
        pygame.draw.line(screen, REMOVE[:3], r.topleft, r.bottomright, 6)
        pygame.draw.line(screen, REMOVE[:3], r.topright, r.bottomleft, 6)

    arrows = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    for frm, to in game.bot_arrows:
        draw_arrow(arrows, frm, to)
    screen.blit(arrows, (0, 0))

    for i in range(8):  # coordinates inside the edge squares, like lichess
        screen.blit(font.render("abcdefgh"[i], True, (90, 70, 50)), (BOARD_X + i * SQ + SQ - 11, BOARD_Y + BOARD - 15))
        screen.blit(font.render(str(8 - i), True, (90, 70, 50)), (BOARD_X + 3, BOARD_Y + i * SQ + 3))


def draw_eval(screen, game, font):
    # Win-chance style scale: +4.0 pawns fills about 91% of the bar for white
    white_share = 1 / (1 + 10 ** (-game.eval_cp / 400))
    bar = pygame.Rect(EVAL_X, BOARD_Y, EVAL_W, BOARD)
    pygame.draw.rect(screen, (30, 30, 30), bar)
    white_h = int(BOARD * white_share)
    pygame.draw.rect(screen, (240, 240, 240), (EVAL_X, BOARD_Y + BOARD - white_h, EVAL_W, white_h))
    label = font.render(game.eval_text, True, (120, 120, 120))
    screen.blit(label, (EVAL_X + EVAL_W // 2 - label.get_width() // 2, BOARD_Y + BOARD - 16))


def draw_camera(screen, frame, outline):
    small = cv2.resize(frame, (CAM_W, CAM_H))
    cv2.polylines(small, [outline], True, (0, 0, 255), 2)  # if this leaves the grid, recalibrate
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    screen.blit(pygame.image.frombuffer(rgb.tobytes(), (CAM_W, CAM_H), "RGB"), (CAM_X, CAM_Y))


def wrap(text, font, width):
    lines, line = [], ""
    for word in text.split():
        test = f"{line} {word}".strip()
        if font.size(test)[0] > width and line:
            lines.append(line)
            line = word
        else:
            line = test
    return lines + [line] if line else lines


if __name__ == "__main__":
    pygame.init()
    # SCALED: draw at WIN_W x WIN_H and let SDL stretch it to the screen (keeps aspect ratio,
    # maps mouse clicks back). Works around tiling WMs like Hyprland resizing the window.
    screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.SCALED | pygame.FULLSCREEN)
    pygame.display.set_caption("Chess Bot")
    big, font, small_font = pygame.font.Font(None, 34), pygame.font.Font(None, 26), pygame.font.Font(None, 20)
    images = piece_images()

    game = Game()
    model, H = vision.load()
    cap = vision.open_camera()
    outline = (np.load(vision.CORNERS) * (CAM_W / 1920)).astype(np.int32)

    busy = False
    show_eval = True
    frame = np.zeros((1080, 1920, 3), np.uint8)
    camera_ok = True

    def run(action):
        """Run a slow action (detection, Stockfish) off the GUI thread; one at a time."""
        global busy
        if busy:
            return
        busy = True

        def work():
            global busy
            try:
                action()
            except Exception as e:  # keep the GUI alive and show what went wrong
                game.message = f"Error: {e}"
            busy = False
        threading.Thread(target=work, daemon=True).start()

    def confirm():
        snapshot = frame.copy()
        run(lambda: game.photo(vision.detect(model, H, snapshot)[0]))

    def toggle_eval():
        global show_eval
        show_eval = not show_eval

    # New game dialog: pick the Stockfish level first
    picking = False
    pending_level = game.level

    def open_new_game():
        global picking, pending_level
        picking, pending_level = True, game.level

    def start_new_game():
        global picking
        picking = False
        level = pending_level
        run(lambda: game.new_game(level))

    def change_level(step):
        global pending_level
        pending_level = max(0, min(20, pending_level + step))

    def cancel_new_game():
        global picking
        picking = False

    panel = pygame.Rect(WIN_W // 2 - 270, WIN_H // 2 - 130, 540, 260)
    dialog_buttons = [
        ("-", lambda: change_level(-1), pygame.Rect(panel.x + 40, panel.y + 90, 70, 60)),
        ("+", lambda: change_level(1), pygame.Rect(panel.right - 110, panel.y + 90, 70, 60)),
        ("Start", start_new_game, pygame.Rect(panel.x + 40, panel.bottom - 70, 160, 44)),
        ("Cancel", cancel_new_game, pygame.Rect(panel.right - 200, panel.bottom - 70, 160, 44)),
    ]

    buttons = [
        ("Confirm (Space)", confirm),
        ("Undo", lambda: run(game.undo)),
        ("New game", open_new_game),
        ("Eval bar", toggle_eval),
        ("Quit", None),
    ]
    button_w = (CAM_W - 10 * (len(buttons) - 1)) // len(buttons)
    button_rects = [pygame.Rect(CAM_X + i * (button_w + 10), WIN_H - M - 44, button_w, 44) for i in range(len(buttons))]

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
                pygame.display.toggle_fullscreen()
            elif picking and event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_LEFT, pygame.K_MINUS):
                    change_level(-1)
                elif event.key in (pygame.K_RIGHT, pygame.K_PLUS, pygame.K_EQUALS):
                    change_level(1)
                elif event.key == pygame.K_RETURN:
                    start_new_game()
                elif event.key == pygame.K_ESCAPE:
                    picking = False
            elif picking and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for _, action, rect in dialog_buttons:
                    if rect.collidepoint(event.pos):
                        action()
            elif picking:
                continue  # the dialog blocks everything else
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                confirm()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for (label, action), rect in zip(buttons, button_rects):
                    if rect.collidepoint(event.pos):
                        if action is None:
                            running = False
                        elif not busy or label == "Eval bar":
                            action()

        ok, new_frame = cap.read()
        if ok:
            frame = new_frame
        camera_ok = ok

        screen.fill(BG)
        banner = "Working..." if busy else game.turn_text()
        screen.blit(big.render(banner, True, TEXT), (CAM_X, M + 4))
        draw_camera(screen, frame, outline)
        if not camera_ok:
            screen.blit(font.render("Camera read failed", True, (255, 80, 80)), (CAM_X + 10, CAM_Y + 10))

        y = CAM_Y + CAM_H + 12
        for text, color in ((game.bot_text, (130, 190, 255)), (game.message, TEXT)):
            for line in wrap(text, font, CAM_W):
                screen.blit(font.render(line, True, color), (CAM_X, y))
                y += 24

        for (label, action), rect in zip(buttons, button_rects):
            disabled = busy and label not in ("Eval bar", "Quit")
            pygame.draw.rect(screen, (70, 70, 70) if disabled else (95, 95, 95), rect, border_radius=6)
            text = font.render(label + (": on" if label == "Eval bar" and show_eval else ": off" if label == "Eval bar" else ""),
                               True, MUTED if disabled else TEXT)
            screen.blit(text, text.get_rect(center=rect.center))

        draw_board(screen, game, images, small_font)
        if show_eval:
            draw_eval(screen, game, small_font)
        screen.blit(font.render(f"Stockfish level {game.level} / 20", True, TEXT), (BOARD_X, BOARD_Y + BOARD + 12))

        if picking:
            shade = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
            shade.fill((0, 0, 0, 160))
            screen.blit(shade, (0, 0))
            pygame.draw.rect(screen, (60, 60, 60), panel, border_radius=10)
            title = font.render("New game: Stockfish level (0 weakest, 20 strongest)", True, TEXT)
            screen.blit(title, title.get_rect(center=(panel.centerx, panel.y + 40)))
            level_text = pygame.font.Font(None, 72).render(str(pending_level), True, TEXT)
            screen.blit(level_text, level_text.get_rect(center=(panel.centerx, panel.y + 120)))
            for label, _, rect in dialog_buttons:
                pygame.draw.rect(screen, (95, 95, 95), rect, border_radius=6)
                text = big.render(label, True, TEXT)
                screen.blit(text, text.get_rect(center=rect.center))

        pygame.display.flip()
        clock.tick(30)

    game.close()
    cap.release()
    pygame.quit()
