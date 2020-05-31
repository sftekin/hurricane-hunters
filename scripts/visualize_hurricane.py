import os
# os.environ['PROJ_LIB'] = '/Users/selimfurkantekin/miniconda3/envs/hurricane/share/basemap'

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from random import sample


m = Basemap(llcrnrlon=-100., llcrnrlat=0., urcrnrlon=-20., urcrnrlat=57.,
            projection='lcc', lat_1=20., lat_2=40., lon_0=-60.,
            resolution='i', area_thresh=1000.)

save_path = '../figures/sample_hurricanes.png'
hurricane_folder = '../data/hurricanes'
hurr_list = os.listdir(hurricane_folder)

hurr_list = sample(hurr_list, 50)
for hurr_file in hurr_list:
    hurr_path = os.path.join(hurricane_folder, hurr_file)

    hurr_arr = np.load(hurr_path, allow_pickle=True)
    lat = hurr_arr[:, 0]
    lon = hurr_arr[:, 1]
    x, y = m(lon, lat)
    m.plot(x, y, linewidth=1.5)

m.bluemarble()
# plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=500)
plt.show()
