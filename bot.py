import numpy as np
import pygetwindow as gw
import mss
import cv2
import os
import pyautogui
import time
import traceback
import argparse

"""
Optimized Match-3 Game Bot with multi-template support, detailed logging and exception protection
"""

# -------------------
# CONFIG
# -------------------
WINDOW_NAME = "LibVNCServer - VNC Viewer Plus"  # Change to your game window title
TEMPLATES_DIR = "./templates"
TEMPLATES_GUI_DIR = "./templates/gui"
MATCH_THRESHOLD = 0.85

# NORMAL TILES with multiple templates per logical type
NORMAL_TILES = {
    "apple": ["apple.png", "apple_grass.png", "apple_purple.png"],
    "pear": ["pear.png", "pear_grass.png", "pear_purple.png"],
    "grape": ["grape.png", "grape_grass.png", "grape_purple.png"],
    "leaf": ["leaf.png", "leaf_grass.png", "leaf_purple.png"],
    "drop": ["drop.png", "drop_grass.png", "drop_purple.png"],
    "flower": ["flower.png", "flower_purple.png"]
}

# SPECIAL_TILES with multiple templates per logical type
SPECIAL_TILES = {
    "rocket": {"templates": ["rocket.png", "rocket_grass.png", "rocket_purple.png"], "priority": 1},
    "bomb": {"templates": ["bomb.png", "bomb_grass.png", "bomb_purple.png"], "priority": 2},
    "dynamite": {"templates": ["dynamite.png", "dynamite_grass.png", "dynamite_purple.png"], "priority": 3},
    "tnt": {"templates": ["tnt.png", "tnt_grass.png", "tnt_purple.png"], "priority": 4},
}

# MAGIC_TILES
MAGIC_TILES = ["magic_black.png", "magic_red.png", "magic_black_purple.png", "magic_red_purple.png", "magic_red_grass.png"]

# GUI config
play_templates = ["play.png"]
complete_templates = ["complete.png"]
already_complete_templates = ["profile.png"]
stage_position = (810, 365)
booster_positions = [(333, 310), (451, 310), (570, 310)]  # x,y relative to window

# -------------------
# LOAD TEMPLATES
# -------------------
templates = {}
tile_type_map = {}  # maps template filename -> logical tile type

# Load normal tiles
for tile_type, files in NORMAL_TILES.items():
    for filename in files:
        path = os.path.join(TEMPLATES_DIR, filename)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            templates[filename] = img
            tile_type_map[filename] = tile_type
        else:
            print(f"⚠️ Normal template not found: {filename}")

# Load special tiles
for tile_type, info in SPECIAL_TILES.items():
    for filename in info["templates"]:
        path = os.path.join(TEMPLATES_DIR, filename)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            templates[filename] = img
            tile_type_map[filename] = tile_type
        else:
            print(f"⚠️ Special template not found: {filename}")

# Load magic tiles
for filename in MAGIC_TILES:
    path = os.path.join(TEMPLATES_DIR, filename)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        templates[filename] = img
        tile_type_map[filename] = "magic"
    else:
        print(f"⚠️ Magic template not found: {filename}")

print(f"🟢 Loaded {len(templates)} templates: {list(templates.keys())}")

# -------------------
# ARGUMENTS
# -------------------
parser = argparse.ArgumentParser(description="Match-3 Bot")
parser.add_argument("--skip-start", action="store_true",
                    help="Skip starting level at first run (assume already in game)")
args = parser.parse_args()
game_started = args.skip_start
if game_started:
    print("⚡ Skipping first-time level start. Assuming game already started.")

# -------------------
# SCREEN CAPTURE
# -------------------
def capture_window(window_name):
    try:
        win = gw.getWindowsWithTitle(window_name)[0]
    except IndexError:
        raise Exception(f"❌ Window not found: {window_name}")

    x, y, w, h = win.left, win.top, win.width, win.height
    with mss.mss() as sct:
        monitor = {"left": x, "top": y, "width": w, "height": h}
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame.astype(np.uint8)

# -------------------
# LOGGING
# -------------------
def log_move(title, tile1, tile2, extra=""):
    print(f"➡️ [{title}] {tile1['type']} {tile1['center']} -> {tile2['type']} {tile2['center']} {extra}")

def log_grid(grid):
    rows, cols = safe_len_grid(grid)
    print("🟦 Current Grid:")
    for r in range(rows):
        row = []
        for c in range(cols):
            tile = grid[r][c]
            if tile:
                row.append(tile['name'].split('.')[0])
            else:
                row.append("None")
        print("   ", row)

# -------------------
# SAFE GRID
# -------------------
def safe_len_grid(grid):
    if grid is None or not grid or not grid[0]:
        return 0, 0
    return len(grid), len(grid[0])

# -------------------
# TILE DETECTION
# -------------------
def detect_tiles(board_img):
    raw_detections = []
    for filename, template in templates.items():
        th, tw = template.shape[:2]
        result = cv2.matchTemplate(board_img, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= MATCH_THRESHOLD)
        for pt in zip(*loc[::-1]):
            score = float(result[pt[1], pt[0]])
            raw_detections.append([pt[0], pt[1], pt[0]+tw, pt[1]+th, score, filename])
    # NMS
    boxes_for_nms = [det[:5] for det in raw_detections]
    if boxes_for_nms:
        nms_boxes = non_max_suppression(boxes_for_nms, overlapThresh=0.3)
        final_detections = []
        for box in nms_boxes:
            for det in raw_detections:
                if np.all(box[:4] == det[:4]):
                    final_detections.append(det)
                    break
        return final_detections
    return []

def non_max_suppression(boxes, overlapThresh=0.3):
    if len(boxes) == 0: return []
    boxes = np.array(boxes, dtype=float)
    pick = []
    x1, y1, x2, y2, scores = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3], boxes[:,4]
    area = (x2 - x1 +1)*(y2 - y1 +1)
    idxs = np.argsort(scores)
    while len(idxs) >0:
        last = len(idxs)-1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2-xx1+1)
        h = np.maximum(0, yy2-yy1+1)
        overlap = (w*h)/area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap>overlapThresh)[0])))
    return boxes[pick].astype(int)

def get_tile_centers(detections):
    tiles = []
    for det in detections:
        x1,y1,x2,y2,score,filename = det
        cx, cy = (x1+x2)//2, (y1+y2)//2
        tiles.append({"type": tile_type_map[filename], "filename": filename, "center": (cx,cy), "pos": (x1,y1,x2,y2)})
    return tiles

def assign_tiles_to_grid(tiles):
    # Get tile center values
    x_centers = [tile["center"][0] for tile in tiles]
    y_centers = [tile["center"][1] for tile in tiles]

    # Threshold
    col_offset = 5
    row_offset = 5
    col_distance = 55
    row_distance = 50

    # Cluster positions
    col_positions = cluster_positions(x_centers, col_offset, col_distance)
    row_positions = cluster_positions(y_centers, row_offset, row_distance)

    # Create empty (grid col x row)
    grid = [[None for _ in col_positions] for _ in row_positions]

    # Map tiles to grid
    for tile in tiles:
        x, y = tile["center"]
        # Find the nearest row index
        row_idx = min(range(len(row_positions)),
                      key=lambda i: abs(y-row_positions[i]) if row_positions[i] is not None else float('inf'))
        # Find the nearest col index
        col_idx = min(range(len(col_positions)),
                      key=lambda i: abs(x-col_positions[i]) if col_positions[i] is not None else float('inf'))
        grid[row_idx][col_idx] = tile

    return grid

def cluster_positions(positions, offset, distance):
        positions = sorted(positions)
        clusters = []
        for pos in positions:
            if not clusters:
                clusters.append([pos])
            else:
                gap = pos - clusters[-1][-1]
                if gap <= offset:
                    clusters[-1].append(pos)
                else:
                    # Gap exceeds threshold → Add None for missing column/row
                    n_gaps = int(round(gap / distance)) - 1
                    for _ in range(n_gaps):    
                        clusters.append([])
                    clusters.append([pos])
        # Return center of each cluster
        cluster_centers = []
        for c in clusters:
            if c:
                cluster_centers.append(int(np.mean(c)))
            else:
                cluster_centers.append(None)
        return cluster_centers

# -------------------
# MATCH LOGIC
# -------------------
def find_matches(grid, min_match=3):
    matched_positions = set()
    rows, cols = safe_len_grid(grid)

    # Horizontal
    for r in range(rows):
        count = 1
        for c in range(1, cols):
            if grid[r][c] and grid[r][c-1] and grid[r][c]['type'] == grid[r][c-1]['type']:
                count += 1
            else:
                if count >= min_match:
                    for k in range(c-count, c):
                        matched_positions.add((r,k))
                count = 1
        if count >= min_match:
            for k in range(cols-count, cols):
                matched_positions.add((r,k))

    # Vertical
    for c in range(cols):
        count = 1
        for r in range(1, rows):
            if grid[r][c] and grid[r-1][c] and grid[r][c]['type'] == grid[r-1][c]['type']:
                count += 1
            else:
                if count >= min_match:
                    for k in range(r-count, r):
                        matched_positions.add((k,c))
                count=1
        if count >= min_match:
            for k in range(rows-count, rows):
                matched_positions.add((k,c))

    return matched_positions

# -------------------
# HELPER: SAFE DRAG
# -------------------
def safe_click_drag(tile1, tile2, window, delay=0.2):
    try:
        x1, y1 = tile1["center"]
        x2, y2 = tile2["center"]
        x1 += window.left
        y1 += window.top
        x2 += window.left
        y2 += window.top

        pyautogui.moveTo(x1, y1, duration=0.1)
        pyautogui.dragTo(x2, y2, duration=delay, button="left")
    except Exception as e:
        print(f"⚠️ Mouse drag error: {e}\n{traceback.format_exc()}")

# -------------------
# CLICK SPECIAL / MAGIC / NORMAL / CREATE SPECIAL
# -------------------
def click_magic_tiles(grid, window, delay=0.2):
    moves=[]
    rows,cols=safe_len_grid(grid)

    # Possible directions for moving (up, down, left, right)
    directions=[(-1,0),(1,0),(0,-1),(0,1)]

    # Iterate through every tile in the grid
    for r in range(rows):
        for c in range(cols):
            tile = grid[r][c]
            if not tile or tile["type"] != "magic": continue
            for dr, dc in directions:
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc <cols:
                    neighbor = grid[nr][nc]
                    if neighbor:
                        # magic + magic
                        if neighbor["type"] == "magic":
                            moves.append(((r,c), (nr,nc), 2))
                        # magic + normal
                        elif neighbor["type"] in NORMAL_TILES:
                            moves.append(((r,c), (nr,nc), 1))
    if not moves: return False

    # Sort priority
    moves.sort(key=lambda x: x[2], reverse=True)

    # Get the move with the highest priority
    (r1,c1), (r2,c2), priority = moves[0]
    tile1, tile2 = grid[r1][c1], grid[r2][c2]

    # Perform the click and drag action for the best move
    safe_click_drag(tile1, tile2, window, delay)

    print(f"✨ [MAGIC-P{priority}] {tile1['type']} -> {tile2['type']}")
    return True

def click_special_tiles(grid, window, delay=0.2):
    moves = []
    rows, cols = safe_len_grid(grid)

    # Possible directions for moving (up, down, left, right)
    directions = [(-1,0),(1,0),(0,-1),(0,1)]

    # Iterate through every tile in the grid
    for r in range(rows):
        for c in range(cols):
            tile = grid[r][c]
            if not tile or tile["type"] not in SPECIAL_TILES:
                continue
            for dr, dc in directions:
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbor = grid[nr][nc]
                    if neighbor:
                        priority = SPECIAL_TILES[tile["type"]]["priority"]
                        # special + special
                        if neighbor["type"] in SPECIAL_TILES:
                            neighbor_priority = SPECIAL_TILES[neighbor["type"]]["priority"]
                            moves.append(((r,c), (nr,nc), priority + neighbor_priority))
                        # special + normal
                        elif neighbor["type"] in NORMAL_TILES:
                            moves.append(((r,c), (nr,nc), priority + 1))
    if not moves: return False

    # Sort priority
    moves.sort(key=lambda x:x[2], reverse=True)

    # Get the move with the highest priority
    (r1,c1), (r2,c2), priority = moves[0]
    tile1, tile2 = grid[r1][c1], grid[r2][c2]

    # Perform the click and drag action for the best move
    safe_click_drag(tile1, tile2, window, delay)

    print(f"💥 [SPECIAL-P{priority}] {tile1['type']} -> {tile2['type']}")
    return True

def click_normal_tiles(grid, window, delay=0.2):
    possible_moves = []
    rows, cols = safe_len_grid(grid)

    # Possible directions for moving (up, down, left, right)
    directions = [(-1,0),(1,0),(0,-1),(0,1)]

    # Iterate through every tile in the grid
    for r in range(rows):
        for c in range(cols):
            tile = grid[r][c]
            if not tile or tile["type"] not in NORMAL_TILES:
                continue
            for dr, dc in directions:
                nr, nc = r+dr, c+dc
                if 0 <= nr <rows and 0 <= nc < cols:
                    neighbor = grid[nr][nc]
                    if neighbor:
                        temp_grid=[row[:] for row in grid]
                        # Swap tile1 to tile2
                        temp_grid[r][c], temp_grid[nr][nc] = temp_grid[nr][nc], temp_grid[r][c]
                        # Check tile matched
                        matches = find_matches(temp_grid)
                        if matches:
                            possible_moves.append({"from": (r,c), "to": (nr,nc), "matches": matches, "type_from": tile["type"], "type_to": neighbor["type"]})
    if not possible_moves: return False

    # Get the best move                      
    move = max(possible_moves, key=lambda m: len(m["matches"]))
    if not move: return False

    (r1,c1), (r2,c2) = move["from"], move["to"]
    tile1, tile2 = grid[r1][c1], grid[r2][c2]

    # Perform the click and drag action for the best move
    safe_click_drag(tile1, tile2, window, delay)

    print(f"➡️ [NORMAL] {tile1['type']} -> {tile2['type']}")
    return True

# -------------------
# SPECIAL / MAGIC WRAPPERS
# -------------------
def try_magic(grid, window):
    try:
        rows, cols = safe_len_grid(grid)
        if rows==0 or cols==0:
            return False
        return click_magic_tiles(grid, window)
    except Exception as e:
        print(f"⚠️ Magic tile error: {e}\n{traceback.format_exc()}")
        return False
    
def try_special(grid, window):
    try:
        rows, cols = safe_len_grid(grid)
        if rows==0 or cols==0:
            return False
        return click_special_tiles(grid, window)
    except Exception as e:
        print(f"⚠️ Special tile error: {e}\n{traceback.format_exc()}")
        return False

def try_normal(grid, window):
    try:
        rows, cols = safe_len_grid(grid)
        if rows==0 or cols==0:
            return False
        return click_normal_tiles(grid, window)
    except Exception as e:
        print(f"⚠️ Normal tile error: {e}\n{traceback.format_exc()}")
        return False

# -------------------
# CHECK GAME COMPLETE
# -------------------
def check_game_complete(window, complete_templates=["complete.png"], templates_dir=TEMPLATES_GUI_DIR, match_threshold=MATCH_THRESHOLD):
    """
    Check if the game level is finished using any template from complete_templates.
    Returns True if detected and clicks center of window.
    """
    try:
        board_img = capture_window(WINDOW_NAME)
        for template_file in complete_templates:
            template_path = os.path.join(templates_dir, template_file)
            template = cv2.imread(template_path, cv2.IMREAD_COLOR)
            if template is None:
                print(f"⚠️ Complete template missing: {template_file}")
                continue
            result = cv2.matchTemplate(board_img, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(result >= match_threshold)
            if loc[0].size > 0:
                # click center of window
                pyautogui.click(window.left + window.width//2, window.top + window.height//2)
                print(f"🏁 Game finished detected: {template_file}")
                return True
        return False
    except Exception as e:
        print(f"⚠️ Error checking game complete: {e}\n{traceback.format_exc()}")
        return False

def check_game_already_complete(complete_templates=["complete.png"], templates_dir=TEMPLATES_GUI_DIR, match_threshold=MATCH_THRESHOLD):
    """
    Check if the game is finished using any template from complete_templates.
    Returns True if detected.
    """
    try:
        board_img = capture_window(WINDOW_NAME)
        for template_file in complete_templates:
            template_path = os.path.join(templates_dir, template_file)
            template = cv2.imread(template_path, cv2.IMREAD_COLOR)
            if template is None:
                print(f"⚠️ Already complete template missing: {template_file}")
                continue
            result = cv2.matchTemplate(board_img, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(result >= match_threshold)
            if loc[0].size > 0:
                print(f"🏁 Game already finished detected: {template_file}")
                return True
        return False
    except Exception as e:
        print(f"⚠️ Error checking game already complete: {e}\n{traceback.format_exc()}")
        return False

# -------------------
# START LEVEL
# -------------------
def start_level(window,
                stage_position=(861, 416),
                booster_positions=[],
                play_templates=["play.png"],
                templates_dir= TEMPLATES_GUI_DIR,
                match_threshold=MATCH_THRESHOLD):
    """
    Detects and clicks:
    1️⃣ Stage template (custom positions)
    2️⃣ Boosters (custom positions)
    3️⃣ Play button (any from play_templates)
    Returns True if Play clicked, False otherwise.
    """
    try:
        board_img = capture_window(WINDOW_NAME)
        if board_img is None:
            print("⚠️ Failed to capture game window.")
            return False

        # Step 1: Click stage
        stage_clicked = False
        sx, sy = stage_position
        pyautogui.click(window.left + sx, window.top + sy)
        print(f"🎯 Stage clicked at {stage_position}")
        time.sleep(1)

        # Step 2: Click boosters
        for idx, (bx, by) in enumerate(booster_positions):
            pyautogui.click(window.left + bx, window.top + by)
            print(f"✨ Booster {idx+1} clicked at ({bx},{by})")
            time.sleep(0.5)

        # Step 3: Click Play
        board_img = capture_window(WINDOW_NAME)
        for play_file in play_templates:
            play_path = os.path.join(templates_dir, play_file)
            template = cv2.imread(play_path, cv2.IMREAD_COLOR)
            if template is None:
                print(f"⚠️ Play template missing: {play_file}")
                continue
            result = cv2.matchTemplate(board_img, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(result >= match_threshold)
            if loc[0].size > 0:
                y, x = loc[0][0], loc[1][0]
                h, w = template.shape[:2]
                pyautogui.click(window.left + x + w//2, window.top + y + h//2)
                print(f"▶️ Play clicked: {play_file}")
                return True

        print("⚠️ Play button not detected.")
        return False

    except Exception as e:
        print(f"⚠️ Error in start_level: {e}\n{traceback.format_exc()}")
        return False

# -------------------
# DEBUG OVERLAY
# -------------------
def draw_tiles_overlay(board_img, tiles):
    img = board_img.copy()

    for tile in tiles:
        x1, y1, x2, y2 = tile["pos"]
        cx, cy = tile["center"]
        tile_type = tile["type"]

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw center point
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)

        # Label with tile type + center coordinates
        label = f"{tile_type} ({cx},{cy})"
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 0), 1)

    cv2.imshow("Tiles Overlay", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_grid_overlay(board_img, grid):
    img = board_img.copy()

    for row_idx, row in enumerate(grid):
        for col_idx, tile in enumerate(row):
            if tile is None:
                continue

            x1, y1, x2, y2 = tile["pos"]
            cx, cy = tile["center"]
            tile_type = tile["type"]

            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw center point
            cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)

            # Put text: name and grid position
            label = f"{tile_type} [{row_idx},{col_idx}]"
            cv2.putText(
                img, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1, cv2.LINE_AA
            )

    cv2.imshow("Grid Overlay", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -------------------
# MAIN LOOP
# -------------------
def main():
    global game_started

    print("🚀 Match-3 Bot Started")

    while True:
        try:
            # 1️⃣ Get game window
            windows = gw.getWindowsWithTitle(WINDOW_NAME)
            if not windows:
                print("⚠️ Game window not found. Retrying in 5s...")
                time.sleep(5)
                continue
            window = windows[0]

            # 2️⃣ Check if level is complete
            if check_game_complete(window, complete_templates=complete_templates):
                game_started = False
                print("✅ Level completed. Waiting before starting next...")
                time.sleep(5)
                continue

            if check_game_already_complete(complete_templates=already_complete_templates):
                game_started = False
                print("✅ Game already completed. Waiting before starting next...")
                time.sleep(5)

            # 3️⃣ Start level if not already started
            if not game_started:
                if start_level(window,
                               stage_position=stage_position,
                               booster_positions=booster_positions,
                               play_templates=play_templates):
                    game_started = True
                    print("✅ Level started successfully")
                    time.sleep(2)  # wait for game to load
                    continue
                else:
                    print("⚠️ Level start failed. Retrying...")
                    time.sleep(2)
                    continue

            # 4️⃣ Capture board, detect tiles, and play
            board_img = capture_window(WINDOW_NAME)
            detections = detect_tiles(board_img)
            tiles = get_tile_centers(detections)
            grid = assign_tiles_to_grid(tiles)

            # Debug overlay
            # draw_tiles_overlay(board_img, tiles)
            # draw_grid_overlay(board_img, grid)

            # PRIORITY: Magic → Special → Create Special → Normal
            if try_magic(grid, window):
                time.sleep(1)
                continue
            elif try_special(grid, window):
                time.sleep(1)
                continue
            elif try_normal(grid, window):
                time.sleep(1)
                continue

            # 5️⃣ Wait before next iteration
            time.sleep(1)
        except KeyboardInterrupt:
                print("🛑 Bot stopped by user")
                break
        except Exception as e:
            print(f"⚠️ Error: {e}\n{traceback.format_exc()}")
            time.sleep(1)

if __name__ == "__main__":
    main()