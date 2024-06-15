import highiq
import xarray as xr
import os
import paramiko
import time
from waggle.plugin import Plugin
import logging
import shutil
import gzip
import argparse
import datetime
import glob
import numpy as np
import cupy
import gc


logging.basicConfig(level=logging.DEBUG)

#file_list = ['aet_RHI_240_20230815_060722.raw']
num_lags = 20
num_samples = 4000
bytes_per_sample = num_lags * num_samples * 8 * 2 + 24

def get_file_portion(path, sftp, beam_start, beam_end, out_name):
    with sftp.file(path , mode='rb') as fi:
        with open(out_name, mode='wb') as out_fi:
            # Write background scan first
            background = fi.read(bytes_per_sample - 24)
            out_fi.write(background)
            fi.seek(bytes_per_sample * beam_start, whence=1)
            for i in range(beam_start, beam_end):
                if i % 50 == 0:
                    logging.debug("Sample %d downloaded" % i)
                data = fi.read(bytes_per_sample)
                out_fi.write(data)


if __name__ == "__main__":
    cur_time = datetime.datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument('--lidar_ip_addr', type=str, default='10.31.81.87',
            help='Lidar IP address')
    parser.add_argument('--lidar_uname', type=str, default='end user',
            help='Lidar username')
    parser.add_argument('--lidar_pwd', type=str, default='',
            help='Lidar password')
    parser.add_argument('--processing_time', type=str, default='',
            help='Process given time period [YYYYMMDD.HH]')
    parser.add_argument('--time_resolution', type=int, default=10,
            help='Process given time period [YYYYMMDD.HH]')
    parser.add_argument('--gate_resolution', type=int, default=60,
            help='Process given time period [YYYYMMDD.HH]')
    parser.add_argument('--processing_interval', type=int, default=200,
            help='Number of points to store in processing interval')
    parser.add_argument('--nfft', type=int, default=1024,
            help="Number of points in the FFT for spectral processing")
    parser.add_argument('--delete', default=False, action='store_true',
            help="Delete spectra for given time period then exit.")
    parser.add_argument('--prev_hour', default=False, action='store_true',
            help="Look at previous hour for default time as well.")
    args = parser.parse_args()
    lidar_ip_addr = args.lidar_ip_addr
    lidar_uname = args.lidar_uname
    lidar_pwd = args.lidar_pwd
    nfft = args.nfft
    if not args.processing_time == "":
        cur_time = datetime.datetime.strptime(args.processing_time, '%Y%m%d.%H')
            
    processing_interval = args.processing_interval
    homepoint = 'atmos.homepoint'
    i = 0
    finish = False
    with paramiko.SSHClient() as ssh:
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        logging.debug("Connecting to %s" % lidar_ip_addr)
        ssh.connect(lidar_ip_addr, username=lidar_uname, password=lidar_pwd)
        year = cur_time.year
        day = cur_time.day
        month = cur_time.month
        hour = cur_time.hour
        prev_hour = cur_time - datetime.timedelta(hours=1)
        pyear = prev_hour.year
        pday = prev_hour.day
        pmonth = prev_hour.month
        phour = prev_hour.hour
        file_path = "/C:/Lidar/Data/Raw/%d/%d%02d/%d%02d%02d/" % (year, year, month, year, month, day)
        file_pathp = "/C:/Lidar/Data/Raw/%d/%d%02d/%d%02d%02d/" % (pyear, pyear, pmonth, pyear, pmonth, pday)
        mempool = cupy.get_default_memory_pool()
        pinned_mempool = cupy.get_default_pinned_memory_pool()
        with ssh.open_sftp() as sftp:
            logging.debug("Connected to the Lidar!")
            file_list = sftp.listdir(file_path)
            if args.prev_hour is True and pday != day:
                file_list = file_list + (sftp.listdir(file_pathp))
            file_name = None
            
            for f in file_list:
                stat = sftp.stat(os.path.join(file_path, f))
                if time.time() - stat.st_mtime < 60:
                    logging.debug("%s modified recently, continuing." % f)
                    continue
                name = f
                for i in range(0, 5760, processing_interval):
                    start_time = time.time()
                    logging.debug("Downloading file samples %d to %d" % (i, i+processing_interval))
                    get_file_portion(os.path.join(file_path, f), sftp, i, i+processing_interval, name)

                    logging.debug("Loading file")
                    file_list = glob.glob('*.raw')
                    raw_file = highiq.io.read_00_data(
                        file_list[0], homepoint)
                    attrs = raw_file.attrs
                    raw_file.to_netcdf('temp.nc')
                    raw_file.close()
                    raw_file = None
                    raw_file = xr.open_dataset('temp.nc')
                
                    raw_file.attrs = attrs
                    logging.debug(raw_file)
                    logging.debug(len(raw_file.time))
                    my_list = []
                    logging.debug("Processing spectra")
                    logging.debug("%d" % mempool.used_bytes())
                    logging.debug("%d" % mempool.total_bytes())
                    logging.debug("%d" % pinned_mempool.n_free_blocks())
                    try:
                        ds_out = highiq.calc.get_psd(raw_file, nfft=nfft,
                                time_window=args.time_resolution, gate_resolution=args.gate_resolution)
                        raw_file = None
                    except TypeError:
                        raw_file.close()
                        raw_file = None 
                        break
                    logging.debug("Output spectra")
                    logging.debug("%d" % mempool.total_bytes())
                     
                    ds_out = highiq.calc.get_lidar_moments(
                           ds_out, intensity_thresh=1.008)
                    ds_out["power_spectral_density"] = ds_out["power_spectral_density"].where(
                    ds_out["intensity"] > 1.008)
                    time_range_shape = ds_out["intensity"].shape
                    spectrum_index = np.where(
                            ds_out["intensity"].values.flatten() > 1.008, 1, 0)
                    which = spectrum_index.cumsum() - 1
                    spectrum_index[spectrum_index > 0] = which[spectrum_index > 0]
                    spectrum_index = np.reshape(spectrum_index, time_range_shape)
                    ds_out["spectrum_index"] = (["time", "range"], spectrum_index)
                    ds_out["spectrum_index"].attrs["long_name"] = "Spectrum index"
                    ds_out["spectrum_index"].attrs["units"] = "1"
                    psd = ds_out["power_spectral_density"].values.reshape(
                        (time_range_shape[0] * time_range_shape[1], nfft))
                    psd = psd[spectrum_index.flatten()]
                    psd_attrs = ds_out["power_spectral_density"].attrs
                    ds_out["power_spectral_density"] = (["index", "sample"], psd)
                    ds_out["power_spectral_density"].attrs = psd_attrs
                    ds_out["doppler_velocity"] = ds_out["doppler_velocity_max_peak"]
                    ds_out = ds_out.drop("doppler_velocity_max_peak")
                    mempool.free_all_blocks()
                    pinned_mempool.free_all_blocks()
                    # Compress the data
                    encoding = {'power_spectral_density': {'dtype': 'float32',  '_FillValue': -9999., 'zlib': True},
                        'intensity': {'dtype': 'float32', '_FillValue': -9999, 'zlib': True},
                        'skewness': {'dtype': 'float32', '_FillValue': -9999, 'zlib': True},
                        'kurtosis': {'dtype': 'float32', '_FillValue': -9999, 'zlib': True},
                        'spectral_width': {'dtype': 'float32', '_FillValue': -9999, 'zlib': True}}
                    
                    lidar_file = file_list[0]
                    start_time_str = str(ds_out['time'][0].dt.strftime('%Y%m%d.%H%M%S').values)
                    out_file_name = 'nant.lidarspectra.z02.c1.%s.nc' % start_time_str
                    ds_out.to_netcdf(out_file_name, 
                            encoding=encoding)
                    ds_out.close()
                    ds_out = None
                    raw_file = None
                    psd = None
                    os.remove('temp.nc')
                    os.remove(lidar_file)
                    logging.debug("Spectra processed in %4.3f seconds" % (time.time() - start_time))
                    with Plugin() as plugin:
                        plugin.upload_file(out_file_name)
                    logging.debug("Time taken: %f" % (time.time() - start_time))
                    gc.collect()
                if args.delete == True:
                    logging.debug("Removing %s" % f)
                    sftp.remove(os.path.join(file_path, name))

