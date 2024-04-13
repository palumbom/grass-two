# imports
import numpy as np
import pandas as pd
import os, sys, pdb, pytz
import matplotlib.pyplot as plt
import astropy.units as u
from pandas.api.types import is_numeric_dtype, is_string_dtype

# set up paths
from showyourwork.paths import user as Paths
paths = Paths()
plotdir = str(paths.figures) + "/"
datadir = str(paths.data) + "/"
staticdir = str(paths.static) + "/"

# set input and output file names
datfile = datadir + ("rms_table.csv")

# read the data
df = pd.read_csv(datfile)

# fix formatting on line names
for i in range(len(df.line)):
    ln = df.line[i]
    ln = ln.replace("_", " ")
    idx = ln.index("I")
    ln = ln[0:idx] + " " + ln[idx:]
    df.iloc[i, 0] = ln

for colname, coldata in df.items():
    if is_string_dtype(df[colname]):
        continue
    df[colname] = np.round(df[colname], decimals=3)


# make table with raw rms values
tabfile1 = datadir + ("rms_table.tex")
df.raw_rms = np.round(df.raw_rms.values, decimals=2)
df.raw_rms_sig = np.round(df.raw_rms_sig.values, decimals=2)
df.to_latex(buf=tabfile1, na_rep="-", index=False, columns=["line", "raw_rms", "raw_rms_sig"])

# get percent improvement
df["bis_inv_slope_impr"] = (np.round(100 * (df["raw_rms"] - df["bis_inv_slope_rms"]) / df["raw_rms"], decimals=0)).astype(int)
df["bis_span_impr"] = (np.round(100 * (df["raw_rms"] - df["bis_span_rms"]) / df["raw_rms"], decimals=0)).astype(int)
df["bis_curve_impr"] = (np.round(100 * (df["raw_rms"] - df["bis_curve_rms"]) / df["raw_rms"], decimals=0)).astype(int)

df["bis_inv_slope_rms"] = np.round(df["bis_inv_slope_rms"], decimals=2)
df["bis_inv_slope_sig"] = np.round(df["bis_inv_slope_sig"], decimals=2)
df["bis_span_rms"] = np.round(df["bis_span_rms"], decimals=2)
df["bis_span_sig"] = np.round(df["bis_span_sig"], decimals=2)
df["bis_curve_rms"] = np.round(df["bis_curve_rms"], decimals=2)
df["bis_curve_sig"] = np.round(df["bis_curve_sig"], decimals=2)


# make table with decorrelation and correlation coefficietns
tabfile2 = datadir + ("decorr_table.tex")
df.to_latex(buf=tabfile2, na_rep="-", columns=["line", 'bis_inv_slope_corr',
            'bis_inv_slope_rms', 'bis_inv_slope_sig', 'bis_inv_slope_impr', 'bis_span_corr',
            'bis_span_rms', 'bis_span_sig', 'bis_span_impr', 'bis_curve_corr', 'bis_curve_rms',
            'bis_curve_sig', 'bis_curve_impr'], index=False)

# pdb.set_trace()

# now make the tuned BIS table
datfile = datadir + ("tuned_params.csv")
df_tuned = pd.read_csv(datfile)

for i in range(len(df.line)):
    ln = df_tuned.line[i]
    ln = ln.replace("_", " ")
    idx = ln.index("I")
    ln = ln[0:idx] + " " + ln[idx:]
    df_tuned.iloc[i, 0] = ln

# round to appropriate number of decimal places
df_tuned.bis_med_pearson = np.round(df_tuned.bis_med_pearson, decimals=3)
df_tuned["bis_tuned_rms"] = np.round(df["bis_tuned_rms"], decimals=2)
df_tuned["bis_tuned_sig"] = np.round(df["bis_tuned_sig"], decimals=2)
df_tuned["bis_tuned_impr"] = (np.round(100 * (df["raw_rms"] - df["bis_tuned_rms"]) / df["raw_rms"], decimals=0)).astype(int)

df_tuned["b1"] *= 100
df_tuned["b2"] *= 100
df_tuned["b3"] *= 100
df_tuned["b4"] *= 100

df_tuned["b1"] = (df_tuned["b1"]).astype(int)
df_tuned["b2"] = (df_tuned["b2"]).astype(int)
df_tuned["b3"] = (df_tuned["b3"]).astype(int)
df_tuned["b4"] = (df_tuned["b4"]).astype(int)

df_tuned.curve_med_pearson = np.round(df_tuned.curve_med_pearson, decimals=3)
df_tuned["curve_tuned_rms"] = np.round(df["curve_tuned_rms"], decimals=2)
df_tuned["curve_tuned_sig"] = np.round(df["curve_tuned_sig"], decimals=2)
df_tuned["curve_tuned_impr"] = (np.round(100 * (df["raw_rms"] - df["curve_tuned_rms"]) / df["raw_rms"], decimals=0)).astype(int)

df_tuned["c1"] *= 100
df_tuned["c2"] *= 100
df_tuned["c3"] *= 100
df_tuned["c4"] *= 100
df_tuned["c5"] *= 100
df_tuned["c6"] *= 100

df_tuned["c1"] = (df_tuned["c1"]).astype(int)
df_tuned["c2"] = (df_tuned["c2"]).astype(int)
df_tuned["c3"] = (df_tuned["c3"]).astype(int)
df_tuned["c4"] = (df_tuned["c4"]).astype(int)
df_tuned["c5"] = (df_tuned["c5"]).astype(int)
df_tuned["c6"] = (df_tuned["c6"]).astype(int)

tabfile3 = datadir + ("tuned_params.tex")
df_tuned.to_latex(buf=tabfile3, na_rep="-", columns=["line", 'b1', 'b2', 'b3', 'b4',
                                                     'bis_med_pearson', 'bis_tuned_rms',
                                                     'bis_tuned_rms', 'bis_tuned_impr',
                                                     'c1', 'c2', 'c3', 'c4', 'c5', 'c6',
                                                     'curve_med_pearson', 'curve_tuned_rms',
                                                     'curve_tuned_rms', 'curve_tuned_impr'], index=False)

# find largest improvement from
# print(df["raw_rms"] - df["bis_tuned_rms"])
print(np.max(df["raw_rms"]))
print(np.min(df["raw_rms"]))

print(np.max(df["bis_tuned_rms"]))
print(np.min(df["bis_tuned_rms"]))
