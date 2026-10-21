"""Lazy CJK font loading for offline video rendering."""

from pathlib import Path
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from functools import lru_cache


@lru_cache(maxsize=16)
def font(size):
    choices = [
        os.environ.get("VALORANT_FONT", ""),
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    path = next((p for p in choices if p and Path(p).is_file()), None)
    if not path:
        raise RuntimeError(
            "Rendering requires a CJK font. Install fonts-noto-cjk or set VALORANT_FONT to a .ttf/.ttc file."
        )
    return ImageFont.truetype(path, size)


def write(d, xy, text, size=24, color="#eef4fa", width=390):
    x, y = xy
    line = ""
    if not any("\u4e00" <= c <= "\u9fff" for c in str(text)):
        for paragraph in str(text).split("\n"):
            line = ""
            for word in paragraph.split():
                candidate = (line + " " + word).strip()
                if line and font(size).getlength(candidate) > width:
                    d.text((x, y), line, font=font(size), fill=color)
                    y += size + 9
                    line = word
                else:
                    line = candidate
            d.text((x, y), line, font=font(size), fill=color)
            y += size + 9
        return y
    for ch in str(text) + "\n":
        if ch == "\n" or font(size).getlength(line + ch) > width:
            d.text((x, y), line, font=font(size), fill=color)
            y += size + 9
            line = "" if ch == "\n" else ch
        else:
            line += ch
    return y


TILES = {}


def box_label(frame, text, x, y, color):
    key = (text, tuple(color))
    if key not in TILES:
        tw = min(650, int(font(21).getlength(text)) + 14)
        tile = Image.new("RGB", (tw, 33), (12, 22, 32))
        dd = ImageDraw.Draw(tile)
        dd.text((7, 3), text, font=font(21), fill=tuple(reversed(color)))
        TILES[key] = cv2.cvtColor(np.asarray(tile), cv2.COLOR_RGB2BGR)
    tile = TILES[key]
    h, w = frame.shape[:2]
    x = max(0, min(int(x), w - tile.shape[1]))
    y = max(0, min(int(y) - tile.shape[0], h - tile.shape[0]))
    frame[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
