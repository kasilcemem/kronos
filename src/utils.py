import numpy as np
from scipy.signal import butter, sosfilt


def nota_bul(frekans):
    notalar = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    if frekans <= 0:
        return "?"
    yarim_ton = 12 * np.log2(frekans / 440.0) + 69
    midi = int(round(yarim_ton))
    oktav = (midi // 12) - 1
    return f"{notalar[midi % 12]}{oktav}"


def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


def normalize(veri):
    maks = np.max(np.abs(veri))
    if maks == 0:
        return veri
    return veri / maks


def alcak_geciren_filtre(veri, kesim_hz, sample_rate, derece=4):
    nyquist = sample_rate / 2
    sos = butter(derece, kesim_hz / nyquist, btype="low", output="sos")
    return sosfilt(sos, veri)


def yuksek_geciren_filtre(veri, kesim_hz, sample_rate, derece=4):
    nyquist = sample_rate / 2
    sos = butter(derece, kesim_hz / nyquist, btype="high", output="sos")
    return sosfilt(sos, veri)


def bant_geciren_filtre(veri, dusuk_hz, yuksek_hz, sample_rate, derece=4):
    nyquist = sample_rate / 2
    sos = butter(derece, [dusuk_hz / nyquist, yuksek_hz / nyquist], btype="band", output="sos")
    return sosfilt(sos, veri)
