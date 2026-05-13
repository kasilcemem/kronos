import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import pytest
from analyzer import fft_hesapla, dominant_frekanslar, bant_guc, db_donustur, analiz_et
from utils import nota_bul, normalize, hz_to_mel, mel_to_hz, alcak_geciren_filtre, yuksek_geciren_filtre, bant_geciren_filtre

SAMPLE_RATE = 44100

def saf_ses_uret(frekans, sure=1.0):
    t = np.linspace(0, sure, int(SAMPLE_RATE * sure), endpoint=False)
    return np.sin(2 * np.pi * frekans * t).astype(np.float32)

def test_frekans_uzunlugu():
    f, g = fft_hesapla(saf_ses_uret(440), SAMPLE_RATE)
    assert len(f) == len(g)

def test_pozitif_frekanslar():
    f, _ = fft_hesapla(saf_ses_uret(440), SAMPLE_RATE)
    assert np.all(f >= 0)

def test_dogru_frekans_tespiti():
    f, g = fft_hesapla(saf_ses_uret(440), SAMPLE_RATE)
    assert abs(f[np.argmax(g)] - 440.0) < 2.0

def test_sessizlik():
    f, g = fft_hesapla(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
    assert g.max() < 1e-6

def test_dominant_liste():
    f, g = fft_hesapla(saf_ses_uret(1000), SAMPLE_RATE)
    sonuc = dominant_frekanslar(f, g, top_n=3)
    assert isinstance(sonuc, list)
    assert len(sonuc) <= 3

def test_bant_anahtarlar():
    f, g = fft_hesapla(saf_ses_uret(1000), SAMPLE_RATE)
    assert set(bant_guc(f, g).keys()) == {"bass", "mid", "treble", "yuksek"}

def test_db_tipi():
    f, g = fft_hesapla(saf_ses_uret(440), SAMPLE_RATE)
    assert isinstance(db_donustur(g), np.ndarray)

def test_analiz_anahtarlar():
    sonuc = analiz_et(saf_ses_uret(440), SAMPLE_RATE)
    assert {"frekanslar", "genlikler", "dominant", "bantlar", "db", "sample_rate", "sure"}.issubset(sonuc.keys())

def test_nota_a4():
    assert nota_bul(440.0) == "A4"

def test_normalize():
    veri = np.array([0.0, 2.0, -4.0, 1.0])
    assert np.max(np.abs(normalize(veri))) == pytest.approx(1.0)

def test_hz_mel():
    assert abs(mel_to_hz(hz_to_mel(1000.0)) - 1000.0) < 0.01

def test_alcak_filtre():
    veri = saf_ses_uret(440)
    assert len(alcak_geciren_filtre(veri, 1000, SAMPLE_RATE)) == len(veri)

def test_yuksek_filtre():
    veri = saf_ses_uret(440)
    assert len(yuksek_geciren_filtre(veri, 200, SAMPLE_RATE)) == len(veri)

def test_bant_filtre():
    veri = saf_ses_uret(1000)
    assert len(bant_geciren_filtre(veri, 500, 2000, SAMPLE_RATE)) == len(veri)
