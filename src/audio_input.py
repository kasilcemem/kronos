import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import librosa


SAMPLE_RATE = 44100


def mikrofon_kaydet(sure=3.0, sample_rate=SAMPLE_RATE):
    print(f"Kayıt başlıyor... {sure} saniye")
    veri = sd.rec(int(sure * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    print("Kayıt tamamlandı.")
    return veri.flatten(), sample_rate


def wav_oku(dosya_yolu):
    sample_rate, veri = wav.read(dosya_yolu)
    if veri.ndim > 1:
        veri = veri.mean(axis=1)
    veri = veri.astype(np.float32)
    if veri.max() > 1.0:
        veri = veri / 32768.0
    return veri, sample_rate


def mp3_oku(dosya_yolu):
    veri, sample_rate = librosa.load(dosya_yolu, sr=None, mono=True)
    return veri, sample_rate


def dosya_oku(dosya_yolu):
    if dosya_yolu.endswith(".wav"):
        return wav_oku(dosya_yolu)
    elif dosya_yolu.endswith(".mp3"):
        return mp3_oku(dosya_yolu)
    else:
        raise ValueError(f"Desteklenmeyen format: {dosya_yolu}")
