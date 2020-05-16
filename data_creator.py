import os
import pandas as pd
import numpy as np

from weather_transformer import WeatherTransformer


class DataCreator:
    def __init__(self, hurricane_path, **params):
        self.hurricane_path = hurricane_path
        self.data_dir = os.path.dirname(hurricane_path)
        self.start_year, self.end_year = params['season_range']
        self.weather_spatial_range = params['weather_spatial_range']
        self.weather_im_size = params['weather_im_size']
        self.weather_freq = params['weather_freq']
        self.check_files = params.get('check_weather_files', False)

        self.weather_tf = WeatherTransformer(check=self.check_files)
        self.hurricane_list = self.__create_hur_data()

    def __create_hur_data(self):
        hurricane_folder = os.path.join(self.data_dir, 'hurricanes')
        weather_folder = os.path.join(self.data_dir, 'weather')
        if os.path.isdir(hurricane_folder):
            print('Loading from saved folder')
            hurricane_list = []
            for hurricane_file in os.listdir(hurricane_folder):
                hurricane_file_path = os.path.join(hurricane_folder, hurricane_file)
                hurricane_list.append(np.load(hurricane_file_path, allow_pickle=True))
        else:
            # save each hurricane as numpy array
            os.makedirs(hurricane_folder)
            os.makedirs(weather_folder)

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
            weather_list = []
            sid_list = pd.unique(data['SID'])
            for sid in sid_list:
                hur = data[data['SID'] == sid]
                hur_name = hur['NAME'].iloc[0]

                # check all the dates are in 3 hours freq
                hur = self._check_dates(hur)

                # extract hur data
                print('\nExtracting Weather for {}'.format(hur_name))
                weather_im, weather_num = self._extract_data(hur)
                save_path_im = os.path.join(weather_folder, '{}_{}.npy'.format(sid, hur_name))
                save_path_num = os.path.join(hurricane_folder, '{}_{}.npy'.format(sid, hur_name))

                np.save(save_path_im, weather_im)
                np.save(save_path_num, weather_num)

                hurricane_list.append(weather_im)
                weather_list.append(weather_num)

            return hurricane_list

    def _check_dates(self, hur):
        date_range = pd.to_datetime(hur['ISO_TIME'].values)
        date_r = pd.date_range(start=date_range[0],
                               end=date_range[-1],
                               freq=str(self.weather_freq) + 'H')

        indices = []
        for day in date_range:
            if day in date_r:
                indices.append(True)
            else:
                indices.append(False)

        return hur[indices]

    def _extract_data(self, hur):
        hur_arr = hur.loc[:, 'LAT':].values
        hur_arr = hur_arr.astype(np.float)

        hur_df = hur.loc[:, ['ISO_TIME', 'LAT', 'LON']]
        x, y = self.weather_im_size[0] / 4, self.weather_im_size[1] / 4
        print(hur_df['ISO_TIME'].values[0])

        ims = []
        arr_list = []
        count = 0
        for t, lat, lon in hur_df.values:
            if count % 10 == 0:
                print(r'{:.2f}%'.format((count / len(hur_df.values)) * 100))
            spatial_r = [[lat - x/2, lat + x/2], [lon - y/2, lon + y/2]]

            if self._check_in_range(spatial_r, self.weather_spatial_range):
                # extract image
                im_arr = self.weather_tf.transform_one_step(t=t, spatial_range=spatial_r)
                im_arr = self._check_dimension(in_arr=im_arr)
                ims.append(im_arr)

                # extract arr
                arr = hur_arr[count]
                arr_list.append(arr)

            count += 1

        ims = np.stack(ims, axis=0)
        nums = np.stack(arr_list, axis=0)

        return ims, nums

    def _check_dimension(self, in_arr):
        l, m, n, d = in_arr.shape
        if (m, n) != self.weather_im_size:
            pad_m = self.weather_im_size[0] - m
            pad_n = self.weather_im_size[1] - n

            if pad_n < 0:
                pad_n = 0
            if pad_m < 0:
                pad_m = 0

            arr_pad = np.zeros((l, m+pad_m, n+pad_n, d))
            for l_idx in range(l):
                for d_idx in range(d):
                    arr = np.pad(in_arr[l_idx, :, :, d_idx],
                                 ((0, pad_m), (0, pad_n)), mode='edge')
                    arr_pad[l_idx, :, :, d_idx] = arr

            if arr_pad.shape > self.weather_im_size:
                arr_pad = arr_pad[:, :self.weather_im_size[0]+1,
                                  :self.weather_im_size[1]+1, :]
            in_arr = arr_pad

        return in_arr

    @staticmethod
    def _check_in_range(r1, r2):
        if (r1[0][0] >= r2[0][0]) & (r1[0][1] <= r2[0][1]) & \
                (r1[1][0] >= r2[1][0]) & (r1[1][1] <= r2[1][1]):
            in_range = True
        else:
            in_range = False
        return in_range


if __name__ == '__main__':
    hurricane_path = 'data/ibtracs.NA.list.v04r00.csv'
    parameters = {
        'season_range': (1994, 2020)
    }
    data_creator = DataCreator(hurricane_path, **parameters)
