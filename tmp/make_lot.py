import matplotlib
import numpy as np
import os
from matplotlib import pyplot as plt 
import sys

bar_colors = ["#7293CB", "#E1974C", "#84BA5B", "#D35E60", "#808585", "#9067A7", "#AB6857", "#CCC210"]

def make_bar_chart(datas,
                   output_directory, output_fig_file, bar_names, output_fig_format='png',
                   errs=None, title=None, xlabel=None, xticklabels=None, ylabel=None):
  fig, ax = plt.subplots()
  ind = np.arange(len(datas[0]))
  width = 0.7/len(datas)
  bars = []
  for i, data in enumerate(datas):
    err = errs[i] if errs != None else None
    bars.append(ax.bar(ind+i*width, data, width, color=bar_colors[i], bottom=0, yerr=err))
  # Set axis/title labels
  if title is not None:
    ax.set_title(title)
  if xlabel is not None:
    ax.set_xlabel(xlabel)
  if ylabel is not None:
    ax.set_ylabel(ylabel)
  if xticklabels is not None:
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(xticklabels)
    plt.xticks(rotation=70)
  else:
    ax.xaxis.set_visible(False) 

  ax.legend(bars, bar_names)
  ax.autoscale_view()

  if not os.path.exists(output_directory):
    os.makedirs(output_directory)
  out_file = os.path.join(output_directory, f'{output_fig_file}.{output_fig_format}')
  plt.savefig(out_file, format=output_fig_format, bbox_inches='tight')

histogram_buckets = [0,1,2,5,10,20,50,100,200,500,1000,10000]

conf_data = [[0 for _ in histogram_buckets] for _ in range(2)]
for line in sys.stdin:
  pid, cites = line.strip().split()
  cites = int(cites)
  whichconf = 0 if ('emnlp-main' in pid) else 1
  for bid, bval in enumerate(histogram_buckets):
    if cites <= bval:
      conf_data[whichconf][bid] += 1
      break
norm_data = [[float(x)/sum(y) for x in y] for y in conf_data]
make_bar_chart(norm_data,'.','cites_diff', ['EMNLP 2020', 'EMNLP 2020 Findings'],xticklabels=[f'<={x}' for x in histogram_buckets], ylabel='ratio of papers')
