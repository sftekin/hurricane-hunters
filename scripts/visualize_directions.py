import matplotlib.pyplot as plt
import numpy as np

from data_creator import DataCreator

data_creator = DataCreator(hurricane_path='../data/ibtracs.NA.list.v04r00.csv', season_range=(1994, 2020))

hurricane_coordinates = [hurr[:, :2] for hurr in data_creator.hurricane_list]
hurricane_directions = [hurr[:, -1] for hurr in data_creator.hurricane_list]

for idx, hurricane in enumerate(data_creator.hurricane_list):

    if idx not in np.arange(20, 40):
        continue

    coordinates = hurricane[:, :2]
    storm_speeds = hurricane[:, -2] / np.max(hurricane[:, -2])
    storm_dirs = hurricane[:, -1]

    plt.figure(figsize=(7, 7))
    plt.scatter(coordinates[:, 1], coordinates[:, 0])

    for i in range(len(coordinates)):
        coord = coordinates[i]
        coord = [coord[1], coord[0]]

        storm_speed = storm_speeds[i]
        storm_dir = storm_dirs[i]
        storm_dir = (90 - storm_dir) / 180 * np.pi

        vec_x = np.cos(storm_dir)*storm_speed
        vec_y = np.sin(storm_dir)*storm_speed
        v = np.array([vec_x, vec_y])[None, :]

        plt.quiver(*coord, v[:, 0], v[:, 1], color=['r'], units="xy", linewidth=0.01, headlength=0.5, headwidth=1, width=.03)

    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.title(f"Center coordinates and storm directions for hurricane ID: {idx}")
    plt.savefig(f"../figures/{idx}.png", bbox_inches="tight", pad_inches=0, dpi=500)
