"""Render the fixed training-history comparison from saved metrics."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

here = Path(__file__).resolve().parent
results = json.loads((here / 'history/results.json').read_text())['cohorts']
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout='constrained')
for axis, horizon in zip(axes, ('4', '24')):
    for offset, days, color in [(-.18, '180', '#a2b0c2'), (.18, '1095', '#2468a2')]:
        values = [100 * results[year][horizon][days]['gain'] for year in ('2024', '2025', '2026')]
        bars = axis.bar(np.arange(3) + offset, values, width=.34, label=f'{days} training days', color=color)
        axis.bar_label(bars, fmt='%.2f', padding=3, fontsize=8)
    axis.axhline(0, color='#444444', linewidth=.8)
    axis.axhline(1, color='#a03939', linestyle='--', label='Required MAE gain (other checks also required)')
    axis.set_xticks(np.arange(3), ['2024', '2025', 'Jan-Jul 2026'])
    axis.set_title(f'{horizon}-hour return')
    axis.set_ylim(-3, 1.55)
    axis.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel('MAE reduction vs unchanged price (%)')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='outside lower center', ncols=1, frameon=False, fontsize=8)
fig.suptitle('Longer training helps, but no cohort passes the fixed screen', fontsize=12)
fig.savefig(here / 'training-history.png', dpi=160)
