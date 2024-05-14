import highiq
import xarray as xr
import os
import paramiko
import time
from waggle.plugin import Plugin
import logging
logging.basicConfig(level=logging.DEBUG)

file_list = ['aet_RHI_240_20230815_060722.raw']
for lidar_file in file_list:
    homepoint = 'atmos.homepoint'
    logging.debug(lidar_file)
    start_time = time.time()
    raw_file = highiq.io.read_00_data(lidar_file, homepoint, start_sample=3, end_sample=120)
    logging.debug(len(raw_file.time))
    my_list = []
    logging.debug("Processing spectra")
    ds_out = highiq.calc.get_psd(raw_file, nfft=256)
    logging.debug("Output spectra")
    ds_out = highiq.calc.get_lidar_moments(ds_out)
    ds_out.to_netcdf(lidar_file + '.nc', engine='h5netcdf')

    with Plugin() as plugin:
        plugin.upload_file(lidar_file + '.nc')
    logging.debug("Time taken: %f" % (time.time() - start_time))
