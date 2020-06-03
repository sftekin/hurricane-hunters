import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


from data_creator import DataCreator

data_creator = DataCreator(hurricane_path='../data/ibtracs.NA.list.v04r00.csv', season_range=(1994, 2020))

hurricane_lengths = [len(hurr)*3 for hurr in data_creator.hurricane_list]

plt.figure()
sns.distplot(hurricane_lengths, bins=20, kde=False, rug=True)
plt.xlabel("Hurricane duration (hours)")
plt.ylabel("Count")
plt.title("Histogram of hurricane durations")
plt.savefig("../figures/length_histogram.png")


plt.figure()
hurricane_speeds = [np.max(hurr[:, -2]) for hurr in data_creator.hurricane_list]
sns.distplot(hurricane_speeds, bins=20, kde=False, rug=True)
plt.xlabel("Hurricane speed (knots)")
plt.ylabel("Count")
plt.title("Histogram of maximum hurricane speeds")
plt.savefig("../figures/storm_speed_histogram.png")


plt.figure()
hurricane_speeds = [np.max(hurr[:, 3]) for hurr in data_creator.hurricane_list]
sns.distplot(hurricane_speeds, bins=10, kde=False, rug=True)
plt.xlabel("Wind speed (knots)")
plt.ylabel("Count")
plt.title("Histogram of maximum sustained wind speeds")
plt.savefig("../figures/wind_speed_histogram.png")


plt.figure()
hurricane_speeds = [np.mean(hurr[:, 4]) for hurr in data_creator.hurricane_list]
sns.distplot(hurricane_speeds, bins=20, kde=False, rug=True)
plt.xlabel("Pressure (mb)")
plt.ylabel("Count")
plt.title("Histogram of pressure")
plt.savefig("../figures/pressure_histogram.png")