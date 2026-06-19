from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path("/Users/gsy/Desktop/topic/paper_arxiv_demo/figures")
OUT_PNG = ROOT / "compact_route_core_result.png"
OUT_PDF = ROOT / "compact_route_core_result.pdf"


def load_font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/System/Library/Fonts/Supplemental/DejaVu Sans Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/DejaVu Sans.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def rounded_box(draw, xy, radius, fill, outline, width=2):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def paste_fit(dst, src, box, pad=18, fill=(255, 255, 255)):
    x0, y0, x1, y1 = box
    inner = (x0 + pad, y0 + pad, x1 - pad, y1 - pad)
    w = max(1, inner[2] - inner[0])
    h = max(1, inner[3] - inner[1])
    src = Image.open(src).convert("RGBA")
    fitted = ImageOps.contain(src, (w, h), method=Image.Resampling.LANCZOS)
    bg = Image.new("RGBA", (w, h), fill + (255,))
    ox = (w - fitted.width) // 2
    oy = (h - fitted.height) // 2
    bg.alpha_composite(fitted, (ox, oy))
    dst.alpha_composite(bg, (inner[0], inner[1]))


def text_block(draw, xy, text, font, fill, spacing=8):
    draw.multiline_text(xy, text, font=font, fill=fill, spacing=spacing)


def arrow(draw, xy0, xy1, color, width=7, head=22):
    draw.line([xy0, xy1], fill=color, width=width)
    import math
    x0, y0 = xy0
    x1, y1 = xy1
    ang = math.atan2(y1 - y0, x1 - x0)
    left = (x1 - head * math.cos(ang - math.pi / 6), y1 - head * math.sin(ang - math.pi / 6))
    right = (x1 - head * math.cos(ang + math.pi / 6), y1 - head * math.sin(ang + math.pi / 6))
    draw.polygon([xy1, left, right], fill=color)


def main():
    W, H = 3400, 2320
    img = Image.new("RGBA", (W, H), "white")
    draw = ImageDraw.Draw(img)

    title_font = load_font(72, bold=True)
    subtitle_font = load_font(30)
    head_font = load_font(38, bold=True)
    body_font = load_font(28)
    small_font = load_font(23)
    tiny_font = load_font(20)

    # Header
    draw.text((W // 2, 52), "From Parsed Structure to Compact Theorem Guidance", font=title_font, fill=(28, 38, 55), anchor="ma")
    draw.text((W // 2, 138), "A compact route works best when it is target-bound, checkable, and short enough to steer the solver without overpowering it.",
              font=subtitle_font, fill=(102, 112, 130), anchor="ma")

    top_y = 220
    left = (90, top_y, 1145, 1110)
    mid = (1180, top_y, 2220, 1110)
    right = (2255, top_y, 3310, 1110)
    bottom = (90, 1160, 3310, 2245)

    # Panels
    for box, outline in [(left, (217, 223, 233)), (mid, (217, 223, 233)), (right, (217, 223, 233)), (bottom, (217, 223, 233))]:
        rounded_box(draw, box, 28, fill=(252, 253, 255), outline=outline, width=3)

    # Left panel: raw example
    draw.text((left[0] + 34, left[1] + 28), "Raw diagram + question", font=head_font, fill=(35, 45, 62))
    paste_fit(img, ROOT / "geometry3k_2407.png", (left[0] + 24, left[1] + 88, left[2] - 24, left[3] - 120), pad=0, fill=(255, 255, 255))
    draw.text((left[0] + 34, left[3] - 70), "The image alone gives geometry cues, but not the preferred action.", font=small_font, fill=(110, 120, 138))

    # Middle panel: compact route
    draw.text((mid[0] + 34, mid[1] + 28), "Compact theorem route", font=head_font, fill=(35, 45, 62))
    rounded_box(draw, (mid[0] + 28, mid[1] + 96, mid[2] - 28, mid[1] + 720), 24, fill=(246, 248, 252), outline=(206, 215, 227), width=3)
    route_text = (
        "1. Use midpoint / midsegment relation\n\n"
        "2. Bind the relation to the target side\n\n"
        "3. Compute from the matched ratio"
    )
    draw.text((mid[0] + 60, mid[1] + 150), route_text, font=body_font, fill=(52, 63, 82), spacing=14)
    draw.text((mid[0] + 60, mid[1] + 540), "Short route = theorem-level action, not a full natural-language solution.", font=small_font, fill=(110, 120, 138))

    rounded_box(draw, (mid[0] + 28, mid[1] + 770, mid[2] - 28, mid[1] + 1032), 24, fill=(255, 247, 236), outline=(255, 183, 95), width=3)
    draw.text((mid[0] + 58, mid[1] + 812), "Why it helps", font=head_font, fill=(175, 85, 20))
    bullets = [
        "Compress structure into theorem actions",
        "Bind the route to the target quantity",
        "Verify before trusting the hint",
    ]
    for i, b in enumerate(bullets):
        y = mid[1] + 862 + i * 60
        draw.text((mid[0] + 60, y), "•", font=body_font, fill=(175, 85, 20))
        draw.text((mid[0] + 95, y), b, font=body_font, fill=(52, 63, 82))

    # Right panel: accuracy chart
    draw.text((right[0] + 34, right[1] + 28), "Downstream accuracy", font=head_font, fill=(35, 45, 62))
    paste_fit(img, ROOT / "compact_route_multimodel_accuracy.png", (right[0] + 24, right[1] + 88, right[2] - 24, right[1] + 650), pad=0, fill=(255, 255, 255))

    rounded_box(draw, (right[0] + 40, right[1] + 675, right[2] - 40, right[1] + 1045), 24, fill=(244, 252, 247), outline=(93, 186, 121), width=3)
    draw.text((right[0] + 68, right[1] + 720), "Cross-model gain", font=head_font, fill=(32, 110, 67))
    draw.text((right[0] + 68, right[1] + 785), "Qwen2.5-VL-3B: 37.75% -> 39.75%", font=body_font, fill=(52, 63, 82))
    draw.text((right[0] + 68, right[1] + 840), "InternVL3-2B: 15.75% -> 31.75%", font=body_font, fill=(52, 63, 82))
    draw.text((right[0] + 68, right[1] + 895), "Qwen3-VL-8B: 54.08% -> 55.57%", font=body_font, fill=(52, 63, 82))

    # Arrows between top panels
    arrow(draw, (left[2] + 18, left[1] + 520), (mid[0] - 26, mid[1] + 520), (160, 170, 190), width=8, head=24)
    arrow(draw, (mid[2] + 18, mid[1] + 520), (right[0] - 26, right[1] + 520), (160, 170, 190), width=8, head=24)

    # Bottom panel: help/harm cases
    draw.text((bottom[0] + 34, bottom[1] + 28), "Route can help or harm", font=head_font, fill=(35, 45, 62))
    # left: example montage
    paste_fit(img, ROOT / "route_help_harm_cases.png", (bottom[0] + 18, bottom[1] + 82, bottom[0] + 2210, bottom[3] - 24), pad=0, fill=(255, 255, 255))
    # right: diagnostic note
    rounded_box(draw, (bottom[0] + 2255, bottom[1] + 82, bottom[2] - 18, bottom[3] - 24), 24,
                fill=(247, 250, 255), outline=(206, 215, 227), width=3)
    draw.text((bottom[0] + 2300, bottom[1] + 130), "Pairwise diagnosis", font=head_font, fill=(35, 45, 62))
    diag = (
        "Win  = raw wrong -> exposed correct\n\n"
        "Loss = raw correct -> exposed wrong\n\n"
        "The route is useful only when it points to the target quantity.\n"
        "If it points to the wrong theorem or wrong binding, it can flip a correct answer."
    )
    draw.text((bottom[0] + 2300, bottom[1] + 210), diag, font=body_font, fill=(52, 63, 82), spacing=14)
    rounded_box(draw, (bottom[0] + 2300, bottom[1] + 575, bottom[0] + 3060, bottom[1] + 690), 18,
                fill=(255, 245, 236), outline=(255, 183, 95), width=2)
    draw.text((bottom[0] + 2330, bottom[1] + 612), "Selective exposure > blind trust", font=small_font, fill=(175, 85, 20))
    draw.text((bottom[0] + 2300, bottom[3] - 130), "Key point: compact routes are useful when they expose target-relevant theorem actions; the next step is calibrated candidate exposure.",
              font=tiny_font, fill=(110, 120, 138), anchor="mb")

    rgb = img.convert("RGB")
    rgb.save(OUT_PNG, dpi=(220, 220))
    rgb.save(OUT_PDF)


if __name__ == "__main__":
    main()
