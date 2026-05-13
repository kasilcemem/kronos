import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks


SAMPLE_RATE = 44100


def fft_hesapla(veri, sample_rate):
    n = len(veri)
    frekanslar = fftfreq(n, d=1.0 / sample_rate)
    spektrum = fft(veri)
    genlikler = np.abs(spektrum) * 2 / n
    pozitif = frekanslar >= 0
    return frekanslar[pozitif], genlikler[pozitif]


def dominant_frekanslar(frekanslar, genlikler, top_n=5):
    peaks, _ = find_peaks(genlikler, height=genlikler.max() * 0.1)
    sirali = peaks[np.argsort(genlikler[peaks])[::-1]]
    sonuc = []
    for i in sirali[:top_n]:
        sonuc.append({
            "frekans": round(float(frekanslar[i]), 2),
            "genlik": round(float(genlikler[i]), 6)
        })
    return sonuc


def bant_guc(frekanslar, genlikler):
    def bant_enerjisi(f_min, f_max):
        maske = (frekanslar >= f_min) & (frekanslar < f_max)
        return float(np.sum(genlikler[maske] ** 2))
    return {
        "bass":   round(bant_enerjisi(20, 250), 6),
        "mid":    round(bant_enerjisi(250, 2000), 6),
        "treble": round(bant_enerjisi(2000, 8000), 6),
        "yuksek": round(bant_enerjisi(8000, 20000), 6),
    }


def db_donustur(genlikler):
    genlikler = np.clip(genlikler, 1e-10, None)
    return 20 * np.log10(genlikler)


def analiz_et(veri, sample_rate):
    frekanslar, genlikler = fft_hesapla(veri, sample_rate)
    return {
        "frekanslar": frekanslar,
        "genlikler":  genlikler,
        "dominant":   dominant_frekanslar(frekanslar, genlikler),
        "bantlar":    bant_guc(frekanslar, genlikler),
        "db":         db_donustur(genlikler),
        "sample_rate": sample_rate,
        "sure":       round(len(veri) / sample_rate, 3),
    }
