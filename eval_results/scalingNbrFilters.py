import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

font = {'family': 'Bitstream Vera Sans',
        'size': 15}

plt.rc('font', **font)

raw_data = {'#filters': ['8', '16', '32', '64', '128'],
            'communication': [4.99,
                              5.59,
                              4.58,
                              11.99,
                              29.79,    # ---- estimation
                              ],
            'total': [62.4,
                      63.8,
                      64.5,
                      132,
                      272,  # ---- estimation
                      ],
            'empty': [0,
                      0,
                      0,
                      0,
                      0,
                      ],
            }

df = pd.DataFrame(raw_data, raw_data['#filters'])

# Create the general plot and the "subplots" i.e. the bars
f, ax = plt.subplots(1, figsize=(7, 5))

# Set the bar width
bar_width = 0.5

# Positions of the left bar-boundaries
bar = [1, 1.7, 2.4, 3.1, 3.8]

# Positions of the x-axis ticks (center of the bars as bar labels)
tick_pos = [pos for pos in bar]

# label="n = 600/$N$"
ax.bar(bar, df['total'], width=bar_width, label="Computation", alpha=0.8, color='#002642')

ax.bar(bar, df['empty'], width=bar_width, label="Communication", hatch="xx", alpha=0.8, color='white')

ax.bar(bar, df['communication'], width=bar_width, alpha=0.8, hatch="xx", color='#C0BDA5', bottom=df['total']-df['communication'])

# Set the x ticks with names
plt.xticks(tick_pos, df['#filters'])

# Set the label and legends
ax.set_ylabel("Runtime (s)", fontsize=17)
ax.set_xlabel("Nbr Filters", fontsize=17)

ax.tick_params(axis='x', labelsize=17)
ax.tick_params(axis='y', labelsize=17)

ax.text(bar[0]-0.16, df['total'][0]+3, df['total'][0], fontsize=14, weight='bold', color='black')
ax.text(bar[1]-0.16, df['total'][1]+3, df['total'][1], fontsize=14, weight='bold', color='black')
ax.text(bar[2]-0.16, df['total'][2]+3, df['total'][2], fontsize=14, weight='bold', color='black')
ax.text(bar[3]-0.14, df['total'][3]+3, int(df['total'][3]), fontsize=14, weight='bold', color='black')
ax.text(bar[4]-0.14, df['total'][4]+3, int(df['total'][4]), fontsize=14, weight='bold', color='black')

plt.xlim([min(bar) - bar_width, max(bar) + bar_width])
plt.ylim(0, 300)

plt.legend()
plt.savefig('scalingNbrFilters.pdf', bbox_inches='tight', pad_inches=0)