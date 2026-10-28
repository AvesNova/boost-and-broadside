"""Colour-vision-deficiency simulation of a mock-up (Machado, Oliveira and Fernandes 2009 matrices,
applied in linear sRGB) plus a palette check with CIEDE76 distances. Usage: python3 colourblind.py in.png out.png"""
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

M = {  # Machado et al. 2009; severity 1.0 for the -opias, 0.6 for the -omalies
    "Protanopia (no red cones, ~1% of men)": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
    "Deuteranopia (no green cones, ~1% of men)": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
    "Tritanopia (no blue cones, rare)": [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]],
    "Protanomaly (weak red, ~1% of men)": [[0.385450, 0.769005, -0.154455], [0.100526, 0.829802, 0.069673], [-0.007442, -0.022190, 1.029632]],
    "Deuteranomaly (weak green, ~5% of men, most common)": [[0.498864, 0.674741, -0.173604], [0.205199, 0.754872, 0.039929], [-0.011131, 0.030969, 0.980162]],
    "Tritanomaly (weak blue, rare)": [[1.104996, -0.046633, -0.058363], [-0.032137, 0.971635, 0.060503], [0.001336, 0.317922, 0.680742]],
}
to_lin = lambda c: np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
to_srgb = lambda c: np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)

def simulate(rgb, name):  # rgb in 0..1, shape (..., 3)
    lin = to_lin(rgb)
    if name.startswith("Achromat"):
        y = lin @ np.array([0.2126, 0.7152, 0.0722])
        out = np.repeat(y[..., None], 3, axis=-1)
    else:
        out = lin @ np.array(M[name]).T
    return np.clip(to_srgb(np.clip(out, 0, 1)), 0, 1)

def lab(rgb):
    lin = to_lin(rgb)
    xyz = lin @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)

PALETTE = [("ally green", "#2E9D5B"), ("enemy vermilion", "#F04E23"), ("thin air purple", "#A88BE0"),
           ("thick air ice", "#A8D8F0"), ("energy graphite", "#3A4150"), ("sky white", "#FFFFFF")]
PAIRS = [("ally green", "enemy vermilion"), ("thin air purple", "thick air ice"), ("enemy vermilion", "thin air purple"),
         ("ally green", "thick air ice"), ("ally green", "thin air purple"), ("enemy vermilion", "thick air ice")]
hexrgb = lambda h: np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)]) / 255

def main(src, dst):
    img = np.asarray(Image.open(src).convert("RGB")) / 255
    names = ["Normal vision"] + list(M) + ["Achromatopsia (no colour, very rare)"]
    w, h = img.shape[1] // 2, img.shape[0] // 2
    cols, pad, top, sw = 2, 24, 50, 150
    W = cols * (w + pad) + pad
    H = len(names) // cols * (h + top + sw) + pad
    sheet = Image.new("RGB", (W, H), (233, 236, 240))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = small = ImageFont.load_default()
    pal = np.array([hexrgb(c) for _, c in PALETTE])
    for k, name in enumerate(names):
        sim = img if k == 0 else simulate(img, name)
        x = pad + (k % cols) * (w + pad)
        y = pad + (k // cols) * (h + top + sw)
        d.text((x, y + 14), name, fill=(42, 47, 58), font=font)
        tile = Image.fromarray((sim * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS)
        sheet.paste(tile, (x, y + top))
        p = pal if k == 0 else simulate(pal, name)
        L = lab(p)
        for i, ((label, _), c) in enumerate(zip(PALETTE, p)):
            bx = x + i * (w // len(PALETTE))
            d.rectangle([bx, y + top + h + 10, bx + 40, y + top + h + 34], fill=tuple((c * 255).astype(int)), outline=(180, 186, 196))
            d.text((bx + 46, y + top + h + 14), label.split()[-1], fill=(90, 98, 114), font=small)
        idx = {n: i for i, (n, _) in enumerate(PALETTE)}
        txt = "   ".join(f"{a.split()[-1]}–{b.split()[-1]} ΔE {np.linalg.norm(L[idx[a]] - L[idx[b]]):.0f}" for a, b in PAIRS)
        d.text((x, y + top + h + 46), txt[: len(txt) // 2].rsplit("   ", 1)[0], fill=(42, 47, 58), font=small)
        d.text((x, y + top + h + 66), txt.split("   ", 3)[-1], fill=(42, 47, 58), font=small)
    sheet.save(dst)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
