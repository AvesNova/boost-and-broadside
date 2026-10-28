"""Green / pink team pairs under colour-vision deficiency: rows are pairs, columns are vision types."""
# Inputs: screenshots of the three canvases in greenpink.html (ids current, darkgreen, lightgreen) saved as gp-<id>.png in the source folder.
import sys, numpy as np
from PIL import Image, ImageDraw, ImageFont
import colourblind as cb
ROWS = [("Green / magenta (current)", "gp-current.png", "#2E9D5B", "#D6338A"),
        ("Dark green / light pink", "gp-darkgreen.png", "#195729", "#EDABD7"),
        ("Light green / dark rose", "gp-lightgreen.png", "#4FD46A", "#A8243A")]
COLS = ["Normal vision", "Protanopia (no red cones, ~1% of men)", "Deuteranopia (no green cones, ~1% of men)",
        "Deuteranomaly (weak green, ~5% of men, most common)", "Tritanopia (no blue cones, rare)", "Achromatopsia (no colour, very rare)"]
src, dst = sys.argv[1], sys.argv[2]
tw, th, pad, lw, top = 300, 190, 14, 230, 60
W, H = lw + len(COLS) * (tw + pad) + pad, top + len(ROWS) * (th + 44) + pad
sheet = Image.new("RGB", (W, H), (233, 236, 240)); d = ImageDraw.Draw(sheet)
fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14); fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
for j, c in enumerate(COLS):
    d.text((lw + j * (tw + pad), 16), c.split(" (")[0], fill=(42, 47, 58), font=fb)
    if "(" in c: d.text((lw + j * (tw + pad), 34), "(" + c.split(" (")[1], fill=(90, 98, 114), font=fs)
for i, (label, f, a, b) in enumerate(ROWS):
    y = top + i * (th + 44)
    d.text((pad, y + 70), label, fill=(42, 47, 58), font=fb); d.text((pad, y + 90), f"{a} / {b}", fill=(90, 98, 114), font=fs)
    img = np.asarray(Image.open(f"{src}/{f}").convert("RGB").resize((tw, th), Image.LANCZOS)) / 255
    P = np.array([cb.hexrgb(a), cb.hexrgb(b)])
    for j, c in enumerate(COLS):
        x = lw + j * (tw + pad)
        sim = img if j == 0 else cb.simulate(img, c)
        q = P if j == 0 else cb.simulate(P, c)
        de = np.linalg.norm(np.subtract(*cb.lab(q)))
        sheet.paste(Image.fromarray((sim * 255).astype(np.uint8)), (x, y))
        for k in range(2): d.rectangle([x + k * 30, y + th + 8, x + k * 30 + 24, y + th + 26], fill=tuple((q[k] * 255).astype(int)), outline=(180, 186, 196))
        d.text((x + 66, y + th + 10), f"ally vs enemy ΔE {de:.0f}", fill=(178, 38, 30) if de < 30 else (42, 47, 58), font=fb)
sheet.save(dst)
