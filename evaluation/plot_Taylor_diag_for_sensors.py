# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from taylorDiagram import TaylorDiagram

# ----- ----- ----- ----- ----- -----
# Command line arguments
# ----- ----- ----- ----- ----- -----
parser  = argparse.ArgumentParser()
parser.add_argument(
    '--wds',
    default = 'anytown',
    type    = str
    )
parser.add_argument(
    '--tag',
    default = 'placement',
    type    = str
    )
parser.add_argument(
    '--smin',
    default = 0,
    type    = float
    )
parser.add_argument(
    '--smax',
    default = 1.2,
    type    = float
    )
args    = parser.parse_args()

# ----- ----- ----- ----- ----- -----
# DB loading
# ----- ----- ----- ----- ----- -----
df  = pd.read_csv(os.path.join('..', 'experiments', 'Taylor_metrics_processed.csv'), index_col=0)
df  = df.loc[(df['tag'] == 'placement') & (df['wds'] == args.wds)]

# ----- ----- ----- ----- ----- -----
# Plot assembly
# ----- ----- ----- ----- ----- -----
fig = plt.figure()
dia = TaylorDiagram(
        1,
        fig     = fig,
        label   = 'reference',
        srange  = (args.smin, args.smax)
        )
dia.samplePoints[0].set_color('r')
dia.samplePoints[0].set_marker('P')
cmap    = plt.get_cmap('Dark2')
markers = ['s', 'x', '*', '.']

def add_samples(dia, df, color, marker):
    for idx_dst, row in df.iterrows():
        sigma   = row['sigma_pred'] / row['sigma_true']
        rho     = row['corr_coeff']
        dia.add_sample(sigma, rho,
            marker  = marker,
            ms  = 9,
            ls  = '',
            mec = color,
            mew = 2,
            mfc = 'none',
            )

seeds   = [1, 8, 5266, 739, 88867]
for i, seed in enumerate(seeds):
    mask    = (df['placement'] == 'random') & (df['seed'] == seed)
    add_samples(dia, df.loc[mask], cmap(i+4), markers[3])
add_samples(dia, df.loc[df['placement'] == 'dist'], cmap(1), markers[0])
add_samples(dia, df.loc[df['placement'] == 'hydrodist'], cmap(2), markers[1])
add_samples(dia, df.loc[df['placement'] == 'hds'], cmap(3), markers[2])

contours    = dia.add_contours(
                levels      = 6,
                colors      = '0.5',
                linestyles  ='dashed',
                alpha       = .8,
                linewidths  = 1
                )
plt.clabel(contours, inline=1, fontsize=10, fmt='%.2f')
dia.add_grid()
dia._ax.axis[:].major_ticks.set_tick_out(True)
dia._ax.axis['left'].label.set_text('Normalized standard deviation')

plt.show()