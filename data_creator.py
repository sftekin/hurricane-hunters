import os
import pandas as pd
import numpy as np


class DataCreator:
    def __init__(self, hurricane_path, **params):
        self.hurricane_path = hurricane_path
        self.data_dir = os.path.dirname(hurricane_path)
        self.start_year, self.end_year = params['season_range']
        self.hurricane_list = self.__create_hur_data()

    def __create_hur_data(self):
        print('Loading Data...')
        data = pd.read_csv(self.hurricane_path)
        print('Processing Data...')

        # remove units row
        data = data.drop(0)

        # crop date
        data['SEASON'] = pd.to_numeric(data['SEASON'])
        data['ISO_TIME'] = pd.to_datetime(data['ISO_TIME'])
        data = data.loc[(self.start_year <= data['SEASON']) &
                        (data['SEASON'] <= self.end_year)]

        # crop columns
        data = data.loc[:, ['SID', 'ISO_TIME', 'NAME',
                            'LAT', 'LON', 'DIST2LAND',
                            'USA_WIND', 'USA_PRES', 'STORM_SPEED',
                            'STORM_DIR']]

        # save each hurricane as numpy array
        hurricane_folder = os.path.join(self.data_dir, 'hurricanes')
        if not os.path.isdir(hurricane_folder):
            os.makedirs(hurricane_folder)

        hurricane_list = []
        sid_list = pd.unique(data['SID'])
        for sid in sid_list:
            hur = data[data['SID'] == sid]
            hur_name = hur['NAME'].iloc[0]
            save_path = os.path.join(hurricane_folder,
                                     '{}_{}.npy'.format(sid, hur_name))
            arr = hur.loc[:, 'LAT':].values
            hurricane_list.append(arr)
            np.save(save_path, arr)

        return hurricane_list


if __name__ == '__main__':
    hurr_path = 'data/ibtracs.NA.list.v04r00.csv'
    parameters = {
        'season_range': (1979, 2020)
    }
    create_data = DataCreator(hurr_path, **parameters)
