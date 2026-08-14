# -*- coding: utf-8 -*-
"""生成 App 图标 AppIcon.icns（Apple 风格圆角、满铺无白边）。"""
import os
import subprocess
from PIL import Image, ImageDraw

SIZE = 1024
RADIUS = int(SIZE * 0.225)  # Apple 风格圆角


def rounded_mask(size, radius):
    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=255)
    return mask


def make_icon(size):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # 背景：深蓝满铺（圆角 -> 透明四角）
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=int(RADIUS * size / SIZE), fill=(27, 42, 74, 255))
    # 内屏面板
    m = int(size * 0.12)
    d.rounded_rectangle([m, m, size - 1 - m, size - 1 - m], radius=int(size * 0.10), fill=(32, 54, 95, 255))
    # 框选区域（青色描边）
    sx0, sy0, sx1, sy1 = int(size*0.26), int(size*0.30), int(size*0.74), int(size*0.66)
    lw = max(6, int(size * 0.018))
    d.rounded_rectangle([sx0, sy0, sx1, sy1], radius=int(size*0.03), outline=(34, 211, 238, 255), width=lw)
    # 框选四角把手
    h = int(size * 0.07)
    for (cx, cy) in [(sx0, sy0), (sx1, sy0), (sx0, sy1), (sx1, sy1)]:
        d.line([cx - h, cy, cx + h, cy], fill=(34, 211, 238, 255), width=lw)
        d.line([cx, cy - h, cx, cy + h], fill=(34, 211, 238, 255), width=lw)
    # 提示徽标（右上角红圆 + 白感叹号）
    bx, by, br = int(size*0.78), int(size*0.22), int(size*0.13)
    d.ellipse([bx - br, by - br, bx + br, by + br], fill=(229, 57, 53, 255))
    d.line([bx, by - int(br*0.5), bx, by + int(br*0.15)], fill=(255, 255, 255, 255), width=max(4, int(size*0.012)))
    d.ellipse([bx - int(br*0.10), by + int(br*0.34), bx + int(br*0.10), by + int(br*0.55)], fill=(255, 255, 255, 255))
    return img


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    iconset = os.path.join(here, "AppIcon.iconset")
    os.makedirs(iconset, exist_ok=True)
    base = make_icon(SIZE)
    specs = {
        "icon_16x16.png": 16,
        "icon_16x16@2x.png": 32,
        "icon_32x32.png": 32,
        "icon_32x32@2x.png": 64,
        "icon_128x128.png": 128,
        "icon_128x128@2x.png": 256,
        "icon_256x256.png": 256,
        "icon_256x256@2x.png": 512,
        "icon_512x512.png": 512,
        "icon_512x512@2x.png": 1024,
    }
    for name, s in specs.items():
        im = base.resize((s, s), Image.LANCZOS) if s != SIZE else base
        im.save(os.path.join(iconset, name))
    out = os.path.join(here, "AppIcon.icns")
    subprocess.run(["iconutil", "-c", "icns", iconset, "-o", out], check=True)
    print("生成图标:", out, os.path.getsize(out), "bytes")


if __name__ == "__main__":
    main()
