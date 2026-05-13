import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


def spektrum_ciz(frekanslar, genlikler):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(frekanslar, genlikler, color="#378ADD", linewidth=0.8)
    ax.set_xlabel("Frekans (Hz)")
    ax.set_ylabel("Genlik")
    ax.set_title("Frekans Spektrumu")
    ax.set_xlim(0, min(frekanslar.max(), 20000))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def dalga_formu_ciz(veri, sample_rate):
    sure = len(veri) / sample_rate
    zaman = np.linspace(0, sure, len(veri))
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(zaman, veri, color="#D85A30", linewidth=0.5, alpha=0.8)
    ax.set_xlabel("Zaman (sn)")
    ax.set_ylabel("Genlik")
    ax.set_title("Dalga Formu")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def spektrogram_ciz(veri, sample_rate):
    fig, ax = plt.subplots(figsize=(12, 4))
    S = librosa.feature.melspectrogram(y=veri, sr=sample_rate, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_db, sr=sample_rate, x_axis="time", y_axis="mel", ax=ax, cmap="magma")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel Spektrogramı")
    plt.tight_layout()
    plt.show()


def bant_grafigi_ciz(bantlar):
    renkler = ["#378ADD", "#1D9E75", "#EF9F27", "#D85A30"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(bantlar.keys(), bantlar.values(), color=renkler)
    ax.set_xlabel("Frekans Bandı")
    ax.set_ylabel("Enerji")
    ax.set_title("Bant Güç Dağılımı")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
