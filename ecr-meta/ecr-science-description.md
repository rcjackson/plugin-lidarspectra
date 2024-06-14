## Halo Photonics Doppler lidar spectral processing

This plugin will process raw Doppler lidar spectra from a connected Halo Photonics Doppler lidar then send the results to the beehive. The spectra are processed into netCDF files at a 10 second, 60 m resolution. In addition this plugin will provide the intensity, doppler velocity, spectral width, skewness and kurtosis of the spectra. 

## Arguments
    
    --lidar_ip_addr: Lidar's IP address
    --lidar_uname: Lidar's username
    --lidar_pwd: Lidar's password
    --nfft: Number of points in FFT (default=1024)
    --processing_interval: Number of samples per file (default 200)
    --processing_time: Time period to process in YYYYMMDD.HH (default latest)


# Data Query
To query the last hour of data, do:
```
df = sage_data_client.query(
            start="-1h",
            filter={"name": "upload", "vsn": "W08D",
                   "plugin": "10.31.81.1:5000/local/plugin-lidarspectra"},).set_index("timestamp")
```                   
The names of the available files are in the *value* key of the dataframe.
