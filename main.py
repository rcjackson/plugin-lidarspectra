import highiq
import xarray as xr
import os
import pywaggle
from waggle.plugin import Plugin

file_list = ['aet_RHI_240_20230815_060722.raw']
for lidar_file in file_list:
    homepoint = 'atmos.homepoint'
    print(lidar_file)
    raw_file = highiq.io.read_00_data(lidar_file, homepoint, start_sample=3, end_sample=120)
    my_list = []
    if len(raw_file.time.values) > 2000:
        for x in raw_file.groupby_bins('time', raw_file.time.values[::2000]):
            d = x[1]
            psd = highiq.calc.get_psd(d, nfft=128,
                    time_window=10.)
            my_list.append(highiq.calc.get_lidar_moments(psd))
            del psd
        ds_out = xr.concat(my_list, dim='time')
        # Improper noise floor at some
    else:
        ds_out = highiq.calc.get_psd(d, nfft=128, time_window=10.)
        ds_out = highiq.calc.get_lidar_moments(ds_out)
    ds.to_netcdf(lidar_file + '.nc', engine='h5netcdf')
    with Plugin() as plugin:
        plugin.upload_file(lidar_file + '.nc')
