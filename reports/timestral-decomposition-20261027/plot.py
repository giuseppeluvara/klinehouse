"""Plot all fixed four-hour variance comparisons, including transfer failures."""

from pathlib import Path
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
btc = json.loads((HERE / "distribution/results.json").read_text())["cohorts"]
transfer = json.loads((HERE / "transfer/results.json").read_text())["assets"]
fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
labels = []
colors = {"BTCUSDT": "#2869a0", "ETHUSDT": "#23816b", "SOLUSDT": "#ae6a26"}
i = 0
for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
    for year in ("2024", "2025", "2026"):
        result = btc[year]["4"] if symbol == "BTCUSDT" else transfer[symbol][year]
        screen = result["screens"]["ewma"]
        scale = result["metrics"]["ewma"]["qlike"]
        gain = 100 * screen["gain"]
        lo, hi = [-100 * x / scale for x in reversed(screen["interval99_583333"])]
        passed = result["passes_both_baselines"]
        ax.errorbar(
            gain,
            i,
            xerr=[[gain - lo], [hi - gain]],
            fmt="o" if passed else "x",
            color=colors[symbol],
            capsize=3,
        )
        labels.append(
            symbol.replace("USDT", "")
            + " "
            + ("Jan-Jul " + year if year == "2026" else year)
            + ("  PASS" if passed else "  FAIL")
        )
        i += 1
ax.axvline(0, color="#555555", linewidth=0.8)
ax.axvline(
    1,
    color="#9f3535",
    linestyle="--",
    linewidth=1,
    label="Minimum primary gain (other criteria also required)",
)
ax.set_yticks(range(len(labels)), labels)
ax.invert_yaxis()
ax.set_xlabel(
    "QLIKE reduction versus calibrated EWMA (%)\nPaired weekly 99.5833% bootstrap intervals"
)
ax.set_title("Four-hour variance: BTC and ETH pass; SOL transfer remains incomplete")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="lower right", fontsize=8)
fig.savefig(HERE / "variance-results.png", dpi=160)
