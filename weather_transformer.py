import os
import gc
import netCDF4
import numpy as np
import pandas as pd


class WeatherTransformer:
    def __init__(self, check=False):

        self.features = ['d', 'cc', 'z', 'u', 'v']
        self.file_dir = '/Volumes/data/dataset/ecmwf/atmosphere'
        self.index_date = pd.to_datetime('1900-01-01')
        self.freq = 3

        self.dates = self._get_file_dates()
        if check:
            self._check_filename_date_matching()

    def _get_file_dates(self):
        file_names = os.listdir(self.file_dir)
        file_names = list(map(lambda x: x.split('.')[0].replace('_', '-') + '-01', file_names))
        file_dates = pd.to_datetime(file_names)

        return file_dates

    def _check_filename_date_matching(self):
        for file_name in os.listdir(self.file_dir):
            file_path = os.path.join(self.file_dir, file_name)
            nc = netCDF4.Dataset(file_path, 'r')
            first_date = int(nc['time'][:][0])

            first_date = self.index_date + pd.DateOffset(hours=first_date)

            file_ = file_name.split('.')[0]
            file_y, file_m = file_.split('_')

            if first_date.year != int(file_y) or first_date.month != int(file_m):
                raise IndexError('{} does not match with inside date'.format(file_))
            gc.collect()

        print()

    def transform(self, date_range, spatial_range):
        """

        :param date_range: e.g pd.date_range(start='2019-01-01', end='2020-03-01', freq='3H')
        :param spatial_range: e.g [[40, 43], [-96, -89]
        :return:
        """
        if date_range.freq.n != self.freq:
            raise ValueError('Input date_range must have the '
                             'same frequency with netCDF files')

        file_dates = []
        for day in date_range:
            file_name = str(day.year) + '_' + str(day.month) + '.nc'
            file_dates.append(file_name)
        file_dates = np.array(file_dates)
        _, idx = np.unique(file_dates, return_index=True)
        file_dates = file_dates[np.sort(idx)]

        data_arr_list = []
        for file_name in file_dates:
            file_path = os.path.join(self.file_dir, file_name)
            nc = netCDF4.Dataset(file_path, 'r')
            data_arr = self._crop_spatial(data=nc, in_range=spatial_range)
            data_arr_list.append(data_arr)
            # since files are big, garbage collect the unref. files
            gc.collect()

        # combine all arr on time dimension
        data_combined = np.stack(data_arr_list, axis=0)

        # temporal crop
        total_time_stamp = len(date_range)
        data_cropped = data_combined[:total_time_stamp]

        return data_cropped

    def _crop_spatial(self, data, in_range):
        """
        :param data:
        :param in_range:[[40, 43], [-96, -89]
        :return:
        """
        lats = data['latitude'][:]
        lons = data['longitude'][:]
        lat_bnds, lon_bnds = in_range

        lat_inds = np.where((lats > lat_bnds[0]) & (lats < lat_bnds[1]))[0]
        lon_inds = np.where((lons > lon_bnds[0]) & (lons < lon_bnds[1]))[0]

        arr_list = []
        for key in self.features:
            subset_arr = data.variables[key][:, :, lat_inds, lon_inds]
            arr_list.append(np.array(subset_arr))

        # combine all features to create arr with shape of
        # (T, L, M, N, D) where L is the levels
        data_combined = np.stack(arr_list, axis=4)

        return data_combined
