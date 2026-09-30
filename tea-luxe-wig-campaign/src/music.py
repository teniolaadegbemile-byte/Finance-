"""Original 20s music bed, 120 BPM, synthesized with numpy only."""
import numpy as np, wave, sys

SR = 44100
DUR = 20.0
N = int(SR * DUR)
BPM = 120
BEAT = 60 / BPM
rng = np.random.default_rng(7)
t_all = np.arange(N) / SR


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def lowpass(x, cutoff, order=2):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X /= (1 + (f / cutoff) ** (2 * order)) ** 0.5
    return np.fft.irfft(X, len(x))


def highpass(x, cutoff):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X *= 1 - 1 / (1 + (f / cutoff) ** 4) ** 0.5
    return np.fft.irfft(X, len(x))


def fftconv(a, b):
    n = len(a) + len(b)
    return np.fft.irfft(np.fft.rfft(a, n) * np.fft.rfft(b, n), n)[: len(a)]


def place(buf, sig, start):
    i = int(start * SR)
    if i >= len(buf):
        return
    sig = sig[: len(buf) - i]
    buf[i : i + len(sig)] += sig


def saw(freq, t):
    return 2 * ((freq * t) % 1.0) - 1


# Chords (bar = 2s): Am9, Fmaj7, Cmaj7/E, G6 — warm, luxe
chords = [
    [57, 60, 64, 67, 71],
    [53, 57, 60, 64, 69],
    [52, 55, 60, 64, 67],
    [55, 59, 62, 64, 67],
]
bass_notes = [45, 41, 40, 43]
BAR = 4 * BEAT
nbars = int(np.ceil(DUR / BAR))

pad = np.zeros(N)
bass = np.zeros(N)
arp = np.zeros(N)
for b in range(nbars):
    ch = chords[b % 4]
    t0 = b * BAR
    last = b == nbars - 1
    L = BAR + (1.5 if not last else 0.0)
    t = np.arange(int(L * SR)) / SR
    env = np.minimum(1, t / 0.25) * np.exp(-np.maximum(0, t - BAR) / 0.4)
    s = np.zeros_like(t)
    for n in ch:
        for det in (-0.08, 0.0, 0.08):
            s += saw(midi(n) * 2 ** (det / 12), t + rng.random())
    place(pad, s * env / 15, t0)
    # sub bass: sine, pulses on beats 1 and 3-and
    for off in (0, 1.5 * BEAT, 2 * BEAT, 3.5 * BEAT):
        tb = np.arange(int(0.45 * SR)) / SR
        f = midi(bass_notes[b % 4])
        e = np.minimum(1, tb / 0.01) * np.exp(-tb / 0.22)
        place(bass, (np.sin(2 * np.pi * f * tb) + 0.25 * np.sin(4 * np.pi * f * tb)) * e, t0 + off)
    # pluck arpeggio 16ths, up an octave
    pattern = [0, 2, 4, 3, 1, 3, 2, 4]
    for k in range(16):
        n = ch[pattern[k % 8]] + 12
        tp = np.arange(int(0.35 * SR)) / SR
        f = midi(n)
        e = np.exp(-tp / 0.09) * np.minimum(1, tp / 0.003)
        tone = np.sin(2 * np.pi * f * tp) + 0.3 * np.sin(4 * np.pi * f * tp) * np.exp(-tp / 0.03)
        vel = 0.9 if k % 4 == 0 else 0.6
        place(arp, tone * e * vel, t0 + k * BEAT / 4)

pad = lowpass(pad, 1400)
# Filter sweep on arp: opens up after the hook
arp = lowpass(arp, 5000)
arp *= np.clip((t_all - 0.5) / 2.0, 0.15, 1.0)

# Drums
drums = np.zeros(N)
tk = np.arange(int(0.4 * SR)) / SR
kick = np.sin(2 * np.pi * (45 * tk + 90 * 0.035 * (1 - np.exp(-tk / 0.035)))) * np.exp(-tk / 0.16)
kick += 0.4 * np.exp(-tk / 0.004) * rng.standard_normal(len(tk)) * 0.3
th = np.arange(int(0.08 * SR)) / SR
hat = highpass(rng.standard_normal(len(th)), 7000) * np.exp(-th / 0.018)
ts = np.arange(int(0.3 * SR)) / SR
snap = lowpass(highpass(rng.standard_normal(len(ts)), 1200), 6000) * np.exp(-ts / 0.07)
snap += 0.5 * np.sin(2 * np.pi * 190 * ts) * np.exp(-ts / 0.05)

nbeats = int(DUR / BEAT)
for i in range(nbeats):
    tb = i * BEAT
    if tb >= 18.0:  # let the end card breathe
        break
    intro = tb < 2.5
    if not intro or i % 2 == 0:
        place(drums, kick * (0.6 if intro else 1.0), tb)
    if not intro:
        place(drums, hat * 0.35, tb + BEAT / 2)
        if i % 2 == 1:
            place(drums, snap * 0.55, tb)
        if i % 4 == 3:
            place(drums, hat * 0.2, tb + 3 * BEAT / 4)

# Riser into the drop at 2.5s and a shimmer into the end card at 15.5s
def riser(start, length, gain):
    tr = np.arange(int(length * SR)) / SR
    nz = highpass(rng.standard_normal(len(tr)), 2500)
    env = (tr / length) ** 2
    place(drums, nz * env * gain, start)

riser(1.3, 1.2, 0.25)
riser(14.3, 1.2, 0.2)
# Impact at 15.5: low boom
tb = np.arange(int(2.5 * SR)) / SR
place(drums, np.sin(2 * np.pi * 50 * tb) * np.exp(-tb / 0.6) * 0.8, 15.5)

# Reverb on pad/arp
ir_t = np.arange(int(2.2 * SR)) / SR
ir = rng.standard_normal(len(ir_t)) * np.exp(-ir_t / 0.55)
ir = lowpass(ir, 5000)
ir /= np.sqrt(np.sum(ir ** 2))

wet = fftconv(pad * 0.6 + arp * 0.5, ir)
mix = pad * 0.55 + arp * 0.32 + bass * 0.55 + drums * 0.6 + wet * 0.35

# Sidechain pump from kicks after the hook
pump = np.ones(N)
for i in range(nbeats):
    tb = i * BEAT
    if 2.5 <= tb < 18.0:
        j = int(tb * SR)
        L = int(0.25 * SR)
        seg = 1 - 0.35 * np.exp(-np.arange(L) / SR / 0.08)
        pump[j : j + L] = np.minimum(pump[j : j + L], seg[: N - j])
mix = mix - (pad * 0.55 + wet * 0.35) * (1 - pump)

# Master: fade in/out, soft clip, normalize
fade = np.ones(N)
fade[: int(0.02 * SR)] = np.linspace(0, 1, int(0.02 * SR))
fo = int(1.5 * SR)
fade[-fo:] = np.linspace(1, 0, fo) ** 1.5
mix *= fade
mix = mix / np.max(np.abs(mix)) * 1.6
mix = np.tanh(mix)
mix = mix / np.max(np.abs(mix)) * 0.89

# Simple stereo: pad/arp widened via tiny delay
d = int(0.012 * SR)
left = mix
right = np.concatenate([mix[:d], mix[:-d]]) * 0.5 + mix * 0.5
st = np.stack([left, right], 1)
st = (st * 32767).astype(np.int16)
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(st.tobytes())
print("ok")
