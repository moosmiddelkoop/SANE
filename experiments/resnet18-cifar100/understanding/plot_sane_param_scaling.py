"""How many parameters should a SANE meta network have for a tiny zoo of 422-parameter models?

Two measured points (SANE parameter counts from the pretrained checkpoints, input model
counts from one zoo checkpoint):
  ResNet-18 CIFAR-100, tk288 (86ae3 / d75a9)   SANE 867,999,132   input 11,220,132
  Small CNN CIFAR-10 (d496b)                   SANE 202,221,325   input 4,970
Two fits through them, extrapolated to 422 input parameters:
  linear:    SANE = a * input + c
  power law: SANE = c * input^b  (a straight line on log-log axes)

2x2 figure: left column SANE parameters, right column SANE / input ratio;
top row log-log axes, bottom row linear axes. Writes sane_param_scaling.png.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import EngFormatter, LogLocator, NullFormatter

measured = {  # name: (input params, SANE params)
    "ResNet-18 CIFAR-100": (11_220_132, 867_999_132),
    "Small CNN CIFAR-10": (4_970, 202_221_325),
}
TINY_ZOO_PARAMS = 422

(x_resnet, y_resnet), (x_cnn, y_cnn) = measured.values()
linear_slope = (y_resnet - y_cnn) / (x_resnet - x_cnn)
power_exponent = np.log(y_resnet / y_cnn) / np.log(x_resnet / x_cnn)
fits = {  # name: (function, color)
    f"linear fit (slope {linear_slope:.1f})": (
        lambda x: y_cnn + linear_slope * (x - x_cnn), "#2a78d6"),
    f"power-law fit (exponent {power_exponent:.3f})": (
        lambda x: y_cnn * (x / x_cnn) ** power_exponent, "#eb6834"),
}

# label offsets in points for the measured points, per (name, ratio panel)
label_offsets = {
    ("ResNet-18 CIFAR-100", False): (-12, -14), ("ResNet-18 CIFAR-100", True): (-12, -14),
    ("Small CNN CIFAR-10", False): (12, -14), ("Small CNN CIFAR-10", True): (12, 8),
}

x_line = np.logspace(np.log10(TINY_ZOO_PARAMS), np.log10(x_resnet), 300)
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for row, scale in enumerate(["log", "linear"]):
    for col, as_ratio in enumerate([False, True]):
        ax = axes[row, col]
        to_y = (lambda x, y: y / x) if as_ratio else (lambda x, y: y)
        to_text = (lambda y: f"{y:,.0f}×") if as_ratio else (lambda y: f"{y / 1e6:.1f} M")
        for fit_name, (fit, color) in fits.items():
            y_tiny = fit(TINY_ZOO_PARAMS)
            ax.plot(x_line, to_y(x_line, fit(x_line)), color=color, lw=2,
                    label=f"{fit_name}: {y_tiny / 1e6:.1f} M, "
                          f"{y_tiny / TINY_ZOO_PARAMS:,.0f}× at {TINY_ZOO_PARAMS}")
            ax.plot(TINY_ZOO_PARAMS, to_y(TINY_ZOO_PARAMS, y_tiny), "o", ms=9,
                    mfc="white", mec=color, mew=2)
        for i, (name, (x, y)) in enumerate(measured.items()):
            ax.plot(x, to_y(x, y), "o", ms=9, color="#0b0b0b",
                    label="measured" if i == 0 else None, zorder=3)
            if scale == "log":
                dx, dy = label_offsets[(name, as_ratio)]
                ax.annotate(f"{name}: {to_text(to_y(x, y))}", (x, to_y(x, y)),
                            xytext=(dx, dy), textcoords="offset points",
                            ha="right" if dx < 0 else "left", va="center",
                            fontsize=9, color="#52514e")
        ax.set_xscale(scale)
        ax.set_yscale(scale)
        ax.margins(y=0.12)  # room for the labels below the lowest points
        if scale == "log" and not as_ratio:  # about one decade: label 1-2-5 ticks
            ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
            ax.yaxis.set_major_formatter(EngFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("input model parameters")
        ax.set_ylabel("SANE / input parameter ratio" if as_ratio else "SANE parameters")
        ax.set_title(f"{'Ratio' if as_ratio else 'SANE parameters'} ({scale} axes)")
        ax.grid(True, which="major", color="#e5e4e0", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
axes[0, 0].plot([], [], "o", ms=9, mfc="white", mec="#52514e", mew=2,
                label=f"prediction at {TINY_ZOO_PARAMS} parameters")
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=2,
           frameon=False, fontsize=10)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig("sane_param_scaling.png", dpi=150)
