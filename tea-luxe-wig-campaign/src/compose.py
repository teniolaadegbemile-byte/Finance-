"""Composite text/animation onto base edit + end card, encode with music."""
import subprocess, sys, numpy as np, imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont, ImageFilter

S = sys.argv[1]
OUT = sys.argv[2]
FF = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS = 1080, 1920, 30
TOTAL = 20.0
BASE_END = 15.5

SERIF_I = "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf"
SERIF = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
SANS = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
SANS_B = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
CREAM = (248, 240, 230)
GOLD = (222, 176, 120)


def spaced(draw, xy, text, font, fill, tracking, anchor_center=True):
    widths = [draw.textlength(c, font=font) for c in text]
    total = sum(widths) + tracking * (len(text) - 1)
    x, y = xy
    if anchor_center:
        x -= total / 2
    for c, w in zip(text, widths):
        draw.text((x, y), c, font=font, fill=fill)
        x += w + tracking
    return total


def text_layer(eyebrow, headline, size=92):
    """Returns RGBA layer (full frame) with eyebrow + headline centred in lower third."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    fe = ImageFont.truetype(SANS_B, 30)
    fh = ImageFont.truetype(SERIF_I, size)
    lines = headline.split("\n")
    lh = int(size * 1.12)
    block_h = 60 + lh * len(lines)
    y0 = 1330 - block_h // 2
    # eyebrow with gold rules either side
    tw = spaced(d, (W / 2, y0), eyebrow, fe, GOLD + (255,), 9)
    ry = y0 + 18
    d.line([(W / 2 - tw / 2 - 70, ry), (W / 2 - tw / 2 - 22, ry)], fill=GOLD + (255,), width=2)
    d.line([(W / 2 + tw / 2 + 22, ry), (W / 2 + tw / 2 + 70, ry)], fill=GOLD + (255,), width=2)
    y = y0 + 62
    for ln in lines:
        d.text((W / 2, y), ln, font=fh, fill=CREAM + (255,), anchor="ma")
        y += lh
    # soft shadow for legibility
    sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sh.putalpha(img.getchannel("A").filter(ImageFilter.GaussianBlur(10)).point(lambda v: int(v * 0.55)))
    out = Image.alpha_composite(sh, img)
    return np.asarray(out).astype(np.float32) / 255.0


def gradient(top, bottom, strength):
    """Vertical darkening mask (H,1) from row top to bottom."""
    g = np.zeros((H, 1, 1), np.float32)
    ys = np.arange(H)
    ramp = np.clip((ys - top) / max(1, bottom - top), 0, 1)
    g[:, 0, 0] = (ramp ** 1.4) * strength
    return g


LOW_GRAD = gradient(820, 1750, 0.62)
TOP_GRAD = 1 - np.clip(np.arange(H) / 330.0, 0, 1)[:, None, None] * 1.0
TOP_GRAD = (TOP_GRAD ** 2 * 0.35).astype(np.float32)

SEGMENTS = [
    (0.0, 2.5, "NEW ARRIVAL", "Meet your new\nsignature shade."),
    (2.5, 6.0, "THE DETAILS", "A natural-looking\nlace part."),
    (6.0, 9.0, "THE COLOUR", "Honey balayage\nhighlights."),
    (9.0, 11.5, "THE ROOT", "Soft, rooted colour\nfor a lived-in look."),
    (11.5, 15.5, "THE MOVEMENT", "Waves that move\nwith you."),
]
layers = [text_layer(e, h) for _, _, e, h in SEGMENTS]

# persistent wordmark
wm = Image.new("RGBA", (W, H), (0, 0, 0, 0))
dw = ImageDraw.Draw(wm)
spaced(dw, (W / 2, 128), "TEA LUXE", ImageFont.truetype(SERIF, 40), CREAM + (235,), 14)
WM = np.asarray(wm).astype(np.float32) / 255.0

# ---- end card ----
logo = Image.open(f"{S}/logo.png").convert("RGB")
end_txt = Image.new("RGBA", (W, H), (0, 0, 0, 0))
de = ImageDraw.Draw(end_txt)
de.text((W / 2, 1180), "Balayage Body Wave", font=ImageFont.truetype(SERIF_I, 96), fill=CREAM + (255,), anchor="ma")
spaced(de, (W / 2, 1310), "LACE FRONT WIG", ImageFont.truetype(SANS_B, 32), GOLD + (255,), 12)
end_sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
end_sh.putalpha(end_txt.getchannel("A").filter(ImageFilter.GaussianBlur(12)).point(lambda v: int(v * 0.6)))
END_TXT = np.asarray(Image.alpha_composite(end_sh, end_txt)).astype(np.float32) / 255.0

cta = Image.new("RGBA", (W, H), (0, 0, 0, 0))
dc = ImageDraw.Draw(cta)
bw, bh = 600, 104
bx, by = (W - bw) // 2, 1440
dc.rounded_rectangle([bx, by, bx + bw, by + bh], radius=52, fill=CREAM + (250,))
spaced(dc, (W / 2, by + 34), "SHOP NOW", ImageFont.truetype(SANS_B, 36), (70, 42, 28, 255), 8)
spaced(dc, (W / 2, by + bh + 38), "LINK IN BIO", ImageFont.truetype(SANS, 28), CREAM + (230,), 10)
CTA = np.asarray(cta).astype(np.float32) / 255.0
END_GRAD = gradient(900, 1700, 0.72)


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def over(frame, layer, alpha, dy=0):
    if alpha <= 0:
        return frame
    L = layer
    if dy:
        L = np.roll(layer, int(dy), axis=0)
    a = L[..., 3:4] * alpha
    return frame * (1 - a) + L[..., :3] * a


def end_frame(t):
    """t = seconds since end card start."""
    z = 1.0 + 0.07 * ease(t / 4.5) + 0.0
    cw, ch = W / z, H / z
    cx, cy = W / 2, H * 0.42
    box = (cx - cw / 2, cy - ch * 0.42, cx + cw / 2, cy + ch * 0.58)
    img = logo.resize((W, H), Image.BICUBIC, box=box)
    f = np.asarray(img).astype(np.float32) / 255.0
    f = f * (1 - END_GRAD)
    a1 = ease((t - 0.35) / 0.6)
    f = over(f, END_TXT, a1, dy=(1 - a1) * 30)
    a2 = ease((t - 0.9) / 0.6)
    f = over(f, CTA, a2, dy=(1 - a2) * 30)
    return f


dec = subprocess.Popen(
    [FF, "-v", "error", "-i", f"{S}/base.mp4", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
    stdout=subprocess.PIPE,
)
enc = subprocess.Popen(
    [FF, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
     "-i", f"{S}/music.wav", "-map", "0:v", "-map", "1:a",
     "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-profile:v", "high",
     "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-t", str(TOTAL), OUT],
    stdin=subprocess.PIPE,
)

nframes = int(TOTAL * FPS)
last = None
fsize = W * H * 3
for n in range(nframes):
    t = n / FPS
    if t < BASE_END - 1e-6:
        buf = dec.stdout.read(fsize)
        if len(buf) == fsize:
            last = np.frombuffer(buf, np.uint8).reshape(H, W, 3).astype(np.float32) / 255.0
        f = last.copy()
        # subtle vignette/gradients
        f = f * (1 - LOW_GRAD) * (1 - TOP_GRAD)
        f = over(f, WM, 1.0)
        for (s, e, _, _), L in zip(SEGMENTS, layers):
            if s <= t < e:
                ain = ease((t - s - 0.08) / 0.45)
                aout = 1.0 if e >= BASE_END else min(1.0, (e - t) / 0.2)
                f = over(f, L, ain * aout, dy=(1 - ain) * 36)
        # quick exposure flash on hard cuts
        for s, _, _, _ in SEGMENTS[1:]:
            if 0 <= t - s < 0.1:
                f = f + (0.12 * (1 - (t - s) / 0.1))
        # fade out base text before end card
        if t > BASE_END - 0.25:
            pass
    else:
        te = t - BASE_END
        f = end_frame(te)
        if te < 0.4:  # cross-dissolve from last base frame
            k = ease(te / 0.4)
            prev = last * (1 - LOW_GRAD) * (1 - TOP_GRAD)
            f = prev * (1 - k) + f * k
        if te < 0.15:
            f = f + 0.25 * (1 - te / 0.15)
    enc.stdin.write((np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())

enc.stdin.close()
enc.wait()
dec.stdout.close()
dec.wait()
print("done", enc.returncode)
