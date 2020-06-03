import os
import numpy as np
import matplotlib.pyplot as plt

from data_creator import DataCreator


data_creator = DataCreator(hurricane_path='../data/ibtracs.NA.list.v04r00.csv',
                           season_range=(1994, 2020),
                           weather_spatial_range=[[0, 65], [-110, 10]],
                           weather_im_size=(25, 25),
                           weather_freq=3,
                           weather_raw_dir='data/weather_raw',
                           rebuild=False)

sample_hur_weather_path = data_creator.weather_list[20]
hur_weather = np.load(sample_hur_weather_path)

save_dir = '../figures/weather'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

for i in range(len(hur_weather)):
    save_path = os.path.join(save_dir, '{}.png'.format(i))
    plt.imshow(hur_weather[i, 0, :, :, 3])
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

print()
