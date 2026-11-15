# Chess Bot Vision Prototype

Python prototype of the vision system for a physical chess bot. A camera watches the board, the program figures out what move the human (white) made, asks Stockfish for black's reply, and prints it. The hardware arm doesn't exist yet, so the human plays the bot's move by hand.

## How to work with me (read first)

- **I am learning. Do not write the program for me.** Teach me how to write each piece myself: explain what the concept is and why it's needed, show the specific API calls with what each argument means, then give me a small task to write and run. Review my code when I paste it.
- Go slowly, one step at a time. Each step should be runnable on its own before moving on.
- Explain the *why* behind every setting or line you ask me to add. Don't tell me to "just add this."
- Use concrete worked examples with real numbers instead of abstract explanations.
- Be concise. No filler, no em dashes.
- Prefer inline logic over extra helper functions when the logic is simple.
- **The approach is object detection (YOLO). Do not steer back toward color thresholds, square classification, or other alternatives.** I already rejected those.

## The game loop (target)

1. I (white) make a move on the physical board.
2. I press a key to signal the move. The program takes a photo.
3. YOLO detects every piece (box + class `dark_piece`/`white_piece`).
4. Each box's **base point** is mapped to a chess square using the calibration homography.
5. The resulting occupancy is matched against `board.legal_moves` (python-chess) to find the move I played.
6. Stockfish picks black's move; the program prints it; I play it physically.
7. Repeat.

## Environment

- OS: EndeavourOS (Arch). Python 3.14 system.
- Project managed with **uv 0.12**. uv 0.12 defaults to a `src/` package layout, so the project was created with `uv init --no-package chessbot` (flat `main.py`). Run code with `uv run <file>.py`.
- Dependencies so far: `opencv-python`, `numpy`, `chess` (the PyPI name is `chess`, not `python-chess`). `ultralytics` 8.4, `pygame-ce` 2.5.8 (GUI; `import pygame`; can load SVG). This laptop is CPU-only (i7-8550U). Stockfish 19 installed (`/usr/bin/stockfish`).
- OpenCV 5.x: use `cv2.VideoWriter.fourcc(...)`, not `cv2.VideoWriter_fourcc` (the old name runs but the type stubs/LSP don't know it).
- My LSP flags `cv2.imread` results as possibly `None`; check `if img is None` after loading.
- Training should run on my desktop (NVIDIA GPU, Bazzite) or the laptop (ASUS ROG Zephyrus M16).

## Physical setup

- Camera: **onn 4K webcam** (75° diagonal FOV) for prototyping. Final build will likely use an Arducam 1080p HDR USB UVC autofocus camera (78° FOV), so code should work unchanged with any UVC camera.
- Camera device: `/dev/video4` (index 4). Laptop webcam occupies video0 to video3. Index can change after reboot/replug; stable path is under `/dev/v4l/by-id/`.
- Capture settings: MJPG at 1920x1080 (set FOURCC **before** width/height). MJPG is needed because raw YUYV 1080p exceeds USB bandwidth and drops to ~5 fps.
- Board: pink and green construction paper squares (matte) for contrast with the pieces. Pieces are **white and brown**.
- Camera is on **black's side**, so **white is at the top of the image**.
- Camera and board must not move after calibration.

## Square mapping

After the perspective warp to 800x800 (100 px per square), with white at the top of the image:
- rank = image row (row 0 is rank 1)
- file = 7 - image col (col 0 is the h-file)
- `chess.square(7 - col, row)`. Example: row 0, col 3 is e1.
- Check: a1 is top-right, h8 is bottom-left.

For detection, don't warp the whole image. Transform only each piece's base point with `cv2.perspectiveTransform` using the matrix from `corners.npy`, then `col = x // 100`, `row = y // 100`. Base point = horizontal center of the box, slightly above the bottom edge (pieces extend toward the top of the image, away from the camera).

## Files

Run everything from the project root. Tools are run as modules so they can import `vision.py`: `uv run python -m tools.detect` (not `uv run tools/detect.py`).

- `main.py`: pygame GUI, `uv run main.py`. Left: live camera + calibrated outline + turn banner + instructions. Right: eval bar + lichess-style board (python-chess SVG pieces). While a bot move is pending the board shows the position *before* it, with a green arrow (plus rook arrow when castling), blue from/to squares and a red X on a piece to remove. Red squares = confirm mismatch. Fullscreen via `pygame.SCALED | pygame.FULLSCREEN` (fixed 1372x700 layout stretched to the screen; needed on Hyprland), F11 toggles. Buttons: Confirm (also Space), Undo, New game (opens a Stockfish level 0-20 picker: -/+ or arrow keys, Enter/Esc), Eval bar on/off, Quit. Current level shown under the board. Detection + Stockfish run in a background thread (banner shows "Working...").
- `game.py`: `Game` class, no drawing. States confirm -> your move -> (Stockfish reply) -> confirm, or over. `SKILL_LEVEL` 5, `THINK_TIME` 0.5, `EVAL_TIME` 0.2. `describe()` gives plain-words instructions for the bot move; `captured_square()` handles en passant; `castling_rook()`.
- `vision.py`: shared settings and code. `MODEL` (`models/v3.pt`), `CORNERS`, stable by-id `CAMERA` path, `IMGSZ`, `CONF`, `BASE_OFFSET`. `open_camera()`, `load_homography()`, `draw_board_outline()`, `detect(model, M, img)` returns `{square: chess.WHITE/BLACK}` plus an annotated copy (step 4: base point, `perspectiveTransform`, off-board skip, highest confidence wins when two boxes share a square). Written by Claude at my request. `uv run vision.py` = live test: space prints a W/D/. text board.
- `moves.py`: step 5. Scores every legal move (depth 1) by how many squares its colored occupancy disagrees with the camera; accepts the best if errors <= `MAX_ERRORS` (2) and it beats the runner-up by `MARGIN` (2), else "unsure". "No change" is a candidate too. Promotions: queen only. `confirm(board, seen)` lists mismatched squares (for checking the bot/arm move was played). `uv run moves.py` = live test playing both sides (space detect, c confirm, u undo).
- `tools/calibrate.py`: click the 4 inner grid corners (TL, TR, BR, BL), saves `corners.npy`, then shows the warped grid in a window (no file saved).
- `tools/detect.py`: my detection test script. `USE_CAMERA` flag switches between live camera and `TEST_IMAGE`.
- `tools/prelabel.py`: space saves `data/<PREFIX>NNN.png` + the model's labels `.txt` + `data/data.yaml` for Roboflow upload. Change `PREFIX` for each new batch (last used `b2_img_`).
- `tools/train.py`: trains `yolo26n.pt` on `dataset/data.yaml` (100 epochs, imgsz 1280, `hsv_s=0.2`, GPU). Run on the PC.
- `corners.npy`: calibration, redone Nov 15. Order must be TL, TR, BR, BL (the first click was BL, which rotated every square; fixed by reordering the saved array).
- `models/`: `v1.pt`, `v2.pt` (`hsv_s=0.2`), `v3.pt` (more data, current; colors correct).
- `dataset/`: Roboflow export v2 (71 train / 17 valid). `names: ['dark_piece', 'white_piece']`, so **class 0 = dark, class 1 = white**. Its `../train/images` paths work as-is.
- `data/`: raw photos and prelabels, **gitignored** (already in Roboflow). Local only.
- Gitignored local junk: `runs/`, `yolo26n.pt` (auto-downloaded pretrained weights).

## Progress so far

1. Done: camera capture and saving frames (lesson 1).
2. Done: perspective calibration with click-to-select corners and grid check (lesson 2).
3. Partly covered: slicing squares and square naming (lesson 3), before switching to detection.
4. Switched from color-based occupancy to **object detection**, because color was unreliable (pieces overlap neighboring squares at an angle, lighting changes).
5. Done: collected **42 raw (unwarped) photos** playing through Byrne vs Fischer, "Game of the Century", one lighting condition.
6. Done: annotated in **Roboflow** (classes exported as `white_piece` and `dark_piece`) (Smart Polygon, then converted to boxes). Partly hidden pieces get a box covering the **whole** piece including the hidden base, since the base point is what maps to a square.
7. Done: generated a dataset version (about 80% train / 20% valid, no stretch resize, no augmentation) and exported in **YOLO12** format. Roboflow class names are `dark_piece` (0) and `white_piece` (1).

## Next steps

1. Done: checked `data.yaml` and a sample label (format `class x_center y_center width height`, normalized 0 to 1). Only add a `path:` key if training can't find the dataset.
2. Done: trained nano model v1 on the PC; box bottom edges looked good on validation images.
3. Done: model runs on photos/live feed; v3 classifies colors correctly after batch 2 (pre-labeled with `tools/prelabel.py`).
4. Done: `vision.detect` maps all 32 pieces of the starting position to the correct squares and colors (Nov 15).
5. Done: move matching by error scoring (`moves.py`), tested on simulated cases (normal, missed piece, wrong color, ambiguous capture, castling, promotion) and live on the starting position. Game flow will confirm the board after each bot move (needed for the robot arm). Done: Stockfish + game loop in `main.py`. Done: pygame GUI (`main.py`) with game logic in `game.py`.
6. Done for batch 2 (v3). Repeat with a new `PREFIX` if accuracy drops in new conditions.

## Labeling rules (for future batches)

- Label every piece in every photo; a missed piece teaches the model it's background.
- Correct class every time.
- Boxes roughly tight; the bottom edge (base) matters most, within about 5 to 10 px.
- Partly hidden: box the full piece, estimating the hidden base. Almost fully hidden (e.g., under a hand): no box.
- Keep captured pieces out of the camera's view.
- Never train on images with anything drawn on them.
