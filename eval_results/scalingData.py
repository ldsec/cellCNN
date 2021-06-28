import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

font = {'family': 'Bitstream Vera Sans',
        'size': 15}

plt.rc('font', **font)

raw_data = {'#rows': ['1K', '2K', '4K', '8K', '16K', '32K'],
            'communication': [0.561,
                              1.815,
                              2.56,
                              4.783,
                              8.71,
                              17.23],
            'total': [6.5,
                      13.1,
                      26.3,
                      50.8,
                      100,
                      201],
            'empty': [0,
                      0,
                      0,
                      0,
                      0,
                      0],
            }

df = pd.DataFrame(raw_data, raw_data['#rows'])

# Create the general plot and the "subplots" i.e. the bars
f, ax = plt.subplots(1, figsize=(7, 5))

# Set the bar width
bar_width = 0.5

# Positions of the left bar-boundaries
bar = [1, 1.7, 2.4, 3.1, 3.8, 4.5]

# Positions of the x-axis ticks (center of the bars as bar labels)
tick_pos = [pos for pos in bar]

#label="n = 200*$N$"
ax.bar(bar, df['total']-df['communication'], width=bar_width, alpha=0.8, color='#002642', label="Computation")

ax.bar(bar, df['empty'], width=bar_width, label="Communication", hatch="xx", alpha=0.8, color='white')

ax.bar(bar, df['communication'], width=bar_width, alpha=0.8, hatch="xx", color='#C0BDA5', bottom=df['total']-df['communication'])

# Set the x ticks with names
plt.xticks(tick_pos, df['#rows'])

plt.legend(loc='upper left')

# Set the label and legends
ax.set_ylabel("Runtime (s)", fontsize=17)
ax.set_xlabel("n", fontsize=17)

#ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
#ax.set_yscale('log')
#ax.set_yticks([5, 50, 200, 500, 1000, 5000])
#ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

ax.tick_params(axis='x', labelsize=17)
ax.tick_params(axis='y', labelsize=17)

ax.text(bar[0]-0.13, df['total'][0]+3, df['total'][0], fontsize=14, weight='bold', color='black')
ax.text(bar[1]-0.19, df['total'][1]+3, df['total'][1], fontsize=14, weight='bold', color='black')
ax.text(bar[2]-0.19, df['total'][2]+3, df['total'][2], fontsize=14, weight='bold', color='black')
ax.text(bar[3]-0.19, df['total'][3]+3, df['total'][3], fontsize=14, weight='bold', color='black')
ax.text(bar[4]-0.17, df['total'][4]+3, int(df['total'][4]), fontsize=14, weight='bold', color='black')
ax.text(bar[5]-0.17, df['total'][5]+3, int(df['total'][5]), fontsize=14, weight='bold', color='black')

plt.xlim([min(bar) - bar_width, max(bar) + bar_width])
plt.ylim(0, 300)

plt.legend()
plt.savefig('scalingData.pdf', bbox_inches='tight', pad_inches=0)