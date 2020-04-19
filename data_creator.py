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
        hurricane_folder = os.path.join(self.data_dir, 'hurricanes')
        if os.path.isdir(hurricane_folder):
            print('Loading from saved folder')
            hurricane_list = []
            for hurricane_file in os.listdir(hurricane_folder):
                hurricane_file_path = os.path.join(hurricane_folder, hurricane_file)
                hurricane_list.append(np.load(hurricane_file_path, allow_pickle=True))
        else:
            # save each hurricane as numpy array
            os.makedirs(hurricane_folder)

            print('Loading Data...')
            data = pd.read_csv(self.hurricane_path, na_values=' ')
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
            hurricane_list = []
            sid_list = pd.unique(data['SID'])
            for sid in sid_list:
                hur = data[data['SID'] == sid]
                hur_name = hur['NAME'].iloc[0]
                save_path = os.path.join(hurricane_folder,
                                         '{}_{}.npy'.format(sid, hur_name))
                arr = hur.loc[:, 'LAT':].values
                arr = arr.astype(np.float)
                hurricane_list.append(arr)
                np.save(save_path, arr)

        return hurricane_list


if __name__ == '__main__':
    hurricane_path = 'data/ibtracs.NA.list.v04r00.csv'
    parameters = {
        'season_range': (1994, 2020)
    }
    data_creator = DataCreator(hurricane_path, **parameters)
