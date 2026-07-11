"""--dump: save a run's intermediate signals to .npz + .png so a bad run can
be re-analyzed offline without re-recording."""
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dsp import HR_BAND


def dump(result, out_dir="dumps"):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, time.strftime("run-%Y%m%d-%H%M%S"))

    np.savez(base + ".npz",
             t=result["t"], raw=result["raw"], filtered=result["filtered"],
             freqs=result["freqs"], power=result["power"],
             bpm=result["bpm"], snr=result["snr"], fs=result["fs"])

    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    ax[0].plot(result["t"] - result["t"][0], result["raw"], lw=0.8)
    ax[0].set_title(f"raw green trace (fs={result['fs']:.2f} Hz)")
    ax[1].plot(result["t"] - result["t"][0], result["filtered"],
               lw=0.8, color="green")
    ax[1].set_title(f"filtered {HR_BAND[0]}-{HR_BAND[1]} Hz")
    ax[1].set_xlabel("s")
    band = (result["freqs"] >= HR_BAND[0]) & (result["freqs"] <= HR_BAND[1])
    ax[2].plot(result["freqs"][band] * 60, result["power"][band],
               lw=0.8, color="steelblue")
    ax[2].axvline(result["bpm"], color="red", ls="--",
                  label=f"{result['bpm']:.1f} BPM (snr {result['snr']:.1f})")
    ax[2].set_title("spectrum")
    ax[2].set_xlabel("BPM")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(base + ".png", dpi=110)
    plt.close(fig)
    return base
