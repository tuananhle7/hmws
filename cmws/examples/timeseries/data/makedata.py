import pickle
import numpy as np
from scipy.stats import kurtosis
import math
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import random

num_timesteps = 128
min_autocorr = 0.3
max_per_task = 5

d = []

nice_plots = ["Symbols_TRAIN.tsv_1.0.png", "Symbols_TRAIN.tsv_4.0.png", "Symbols_TRAIN.tsv_5.0.png", "PLAID_TRAIN.tsv_0.0.png", "PLAID_TRAIN.tsv_4.0.png", "PLAID_TRAIN.tsv_5.0.png", "PLAID_TRAIN.tsv_9.0.png", "PigArtPressure_TRAIN.tsv_18.0", "PigAirwayPressure_TRAIN.tsv_46.0.png", "Phoneme_TRAIN.tsv_35.0.png", "NonInvasiveFetalECGThorax2_TRAIN.tsv_34.0.png", "NonInvasiveFetalECGThorax1_TRAIN.tsv_26.0.png", "InsectWingbeatSound_TRAIN.tsv_10.0.png", "InsectEPGSmallTrain_TRAIN.tsv_2.0.png", "Haptics_TRAIN.tsv_4.0.png", "HandOutlines_TRAIN.tsv_0.0.png", "HandOutlines_TRAIN.tsv_1.0.png", "GesturePebbleZ1_TRAIN.tsv_6.0.png", "GestureMidAirD2_TRAIN.tsv_22.0.png", "FordA_TRAIN.tsv_-1.0.png", "EthanolLevel_TRAIN.tsv_2.0.png", "EOGHorizontalSignal_TRAIN.tsv_10.0.png", "DodgerLoopWeekend_TRAIN.tsv_2.0.png", "DodgerLoopGame_TRAIN.tsv_1.0.png", "FreezerSmallTrain_TRAIN.tsv_1.0.png", "InlineSkate_TRAIN.tsv_1.0.png", "Rock_TRAIN.tsv_1.0.png", "WordSynonyms_TRAIN.tsv_15.0.png"]

nice_plot_seqs = []

for file in Path("UCRArchive_2018").glob("*/*_TRAIN.tsv"):
    print(file.name)
    with open(file, "r") as f:
        seqs = [[float(x) for x in line.split("\t")] for line in f.readlines()]
        seqs = {seq[0]: seq[1:] for seq in seqs
                                   if len(seq) > 1+num_timesteps}
        n = 0
        items = list(seqs.items())
        random.seed(0)
        random.shuffle(items)

        for k, v in items:
            seq = np.array(v)
            seq = seq[~np.isnan(seq)]

            s = pd.Series(seq).rolling(window=10).std()
            sd_is_outlier = np.array(s > s.quantile(0.5) + 1.5*(s.quantile(0.75)-s.quantile(0.25)))
            mid = len(seq)//2
            first_idx = np.argmax(np.arange(mid)*sd_is_outlier[:mid]) + 10
            last_idx = len(seq) - 1 - np.argmax(np.arange(len(seq)-mid)*sd_is_outlier[:mid-1:-1])
            seq = seq[first_idx:last_idx]

            if len(seq)<num_timesteps:
                print(f"{k} Too short")
                continue

            autocorr = np.corrcoef(seq[:num_timesteps-1], seq[1:num_timesteps])[0,1]
            if autocorr < min_autocorr:
                print(f"{k} Too uncorrelated ({autocorr})")
                continue

            while len(seq)>=num_timesteps*2:
                new_seq = seq[::2]
                autocorr = np.corrcoef(new_seq[:num_timesteps-1], new_seq[1:num_timesteps])[0,1] 
                if autocorr > min_autocorr:
                    seq = new_seq
                else:
                    break

            autocorr = np.corrcoef(seq[:num_timesteps-1], seq[1:num_timesteps])[0,1]

            l = len(seq) - num_timesteps
            seq = seq[int(math.floor(l/2)):len(seq)-int(math.ceil(l/2))]
            seq = (seq - seq.mean())/seq.std()

            if np.stack([seq[:-3], seq[1:-2], seq[2:-1], seq[3:]]).std(0).min() < 1e-3:
                print(f"{k} Too static")
                continue

            if any(np.isnan(seq)) or any(np.isinf(seq)) or seq.std()==0:
                print(f"{k} has nans")
                continue

            instance = {
                'source': file,
                'class': k,
                'data': [seq]
            }
            plt.plot(seq)
            plt.savefig(f"plots/{file.name}_{k}.png")
            plt.clf()

            if f"{file.name}_{k}.png" in nice_plots:
                nice_plot_seqs.append(seq)
                d.insert(0, instance)
            else:
                d.append(instance)
            
            n += 1

            if n >= max_per_task:
                break
        
        if n>0:
            print(f"Added {n} sequences from {file}")
        print()

plt.figure(figsize=(12, 10))
for i, seq in enumerate(nice_plot_seqs):
    plt.subplot(6, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.plot(seq)
    
plt.savefig(f"nice_plots.png")

with open("./data_new.p", "wb") as f:
    pickle.dump(d, f)