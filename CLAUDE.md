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
- Dependencies so far: `opencv-python`, `numpy`, `chess` (the PyPI name is `chess`, not `python-chess`). `ultralytics` 8.4 (added). This laptop is CPU-only (i7-8550U). Stockfish not installed yet (`sudo pacman -S stockfish`).
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

- `main.py`: live camera feed (lesson 1). Space saves numbered photos to `data/img_N.png`, q quits. Open review notes: check `ok` from `cap.read()` and break if False; q calls `exit()` so `cap.release()` is never reached (use `break`).
- `calibrate.py`: grabs one live frame from the camera, I click the 4 inner grid corners (top-left, top-right, bottom-right, bottom-left), computes `M = cv2.getPerspectiveTransform(src, dst)`, warps to 800x800, draws a grid, saves `images/warped.png` and `corners.npy`. Earlier review notes are fixed (dots drawn on `display = img.copy()`, quitting before 4 clicks exits cleanly).
- `corners.npy`: calibration corners for the pink/green board (Oct 11), verified with `images/warped.png`; grid lines match square boundaries.
- `data/`: raw training photos (currently missing from the project folder; the 42 photos exist only inside `dataset/`).
- `train.py`: trains `yolo26n.pt` on `dataset/data.yaml`. Real run was on the PC (GPU), 100 epochs, imgsz 1280.
- `models/best.pt`: trained v1 model, copied from the PC run and committed. `runs/` and `yolo26n.pt` are gitignored.
- `dataset/`: Roboflow export, 34 train / 8 valid images, no test split. Classes in `data.yaml`: `names: ['dark_piece', 'white_piece']`, so **class 0 = dark (brown), class 1 = white**. `data.yaml` is unedited; its `../train/images` paths work because Ultralytics strips a leading `../` when the path doesn't exist (verified in training).

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
3. Run the trained model on new photos and on the live feed; draw boxes on a `.copy()` to inspect.
4. Map detections to squares (base point + `perspectiveTransform`), ignore points outside the board.
5. Match occupancy to legal moves, then add Stockfish via `chess.engine.SimpleEngine.popen_uci`.
6. Collect more games under **different lighting** (and a few with a hand in frame), pre-label them with the first model, correct in Roboflow, retrain.

## Labeling rules (for future batches)

- Label every piece in every photo; a missed piece teaches the model it's background.
- Correct class every time.
- Boxes roughly tight; the bottom edge (base) matters most, within about 5 to 10 px.
- Partly hidden: box the full piece, estimating the hidden base. Almost fully hidden (e.g., under a hand): no box.
- Keep captured pieces out of the camera's view.
- Never train on images with anything drawn on them.
