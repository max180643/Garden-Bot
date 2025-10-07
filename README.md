# Garden-Bot

An automated **match-3 game bot** built with Python.
This bot detects and plays Match-3 style games (for Gardenscapes) by capturing screenshots, identifying tile patterns, and simulating mouse interactions automatically.

## Requirements

- Python 3.9 or higher

### Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Usage

Run the bot from the terminal:

```bash
python bot.py
```

Optional flag:

```bash
python bot.py --skip-start
```

| Option         | Description                                                             |
| -------------- | ----------------------------------------------------------------------- |
| `--skip-start` | Skip the initial “start level” step (if you’re already inside the game) |

## 🧠 How It Works

1. **Window Detection** — Finds the target window by title.
2. **Screenshot Capture** — Grabs the game frame.
3. **Template Matching** — Uses OpenCV to locate all tiles on screen.
4. **Grid Mapping** — Assigns detected tiles to a logical grid based on position clustering.
5. **Move Prioritization**

   - Tries magic tile moves first (highest priority).
   - Then special tiles (bombs, rockets, etc.).
   - Lastly, normal matches.

6. **Mouse Control** — Performs drag/swipe operations
7. **Level Completion Check** — Detects “level complete” or “already complete” GUI states.
8. **Restart** — Clicks next level and repeats automatically.

## ⚠️ Notes

- Ensure the game window is **fully visible** and not minimized.
- Template images must **exactly match** in scale and color for accurate detection.
- This bot is not optimized for games with a **limited move count**. (Required **Infinite Moves**)

## ⚠️ Disclaimer

This project is for **educational purposes only**.
Using automation tools on online or commercial games may violate their Terms of Service.
Use responsibly.
