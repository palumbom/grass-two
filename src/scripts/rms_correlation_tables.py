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
tabfile1 = datadir + ("rms_table.tex")
tabfile2 = datadir + ("decorr_table.tex")

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
df.to_latex(buf=tabfile1, na_rep="-", index=False, columns=["line", "raw_rms", "raw_rms_sig"])

# make table with decorrelation and correlation coefficietns
df.to_latex(buf=tabfile2, na_rep="-", index=False)

# pdb.set_trace()

# now make the tuned BIS table
datfile = datadir + ("tuned_params.csv")
df = pd.read_csv(datfile)

for i in range(len(df.line)):
    ln = df.line[i]
    ln = ln.replace("_", " ")
    idx = ln.index("I")
    ln = ln[0:idx] + " " + ln[idx:]
    df.iloc[i, 0] = ln

# round to appropriate number of decimal places
df.med_pearson = np.round(df.med_pearson, decimals=3)

tabfile3 = datadir + ("tuned_params.tex")
df.to_latex(buf=tabfile3, na_rep="-", index=False)
