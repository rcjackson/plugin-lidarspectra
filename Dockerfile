FROM waggle/plugin-base:1.1.1-base

WORKDIR /app
RUN apt-get update -y
RUN apt update -y
RUN apt-get install -y --no-install-recommends python3-netcdf4 gcc
RUN apt-get install -y libpython3-dev
COPY requirements.txt /app
RUN git config --global --add safe.directory /app
RUN pip3 install -r requirements.txt
RUN pip3 install git+https://github.com/rcjackson/HighIQ.git
RUN apt-get install -y libhdf5-serial-dev python3-h5py
RUN pip3 install h5netcdf
RUN pip3 install numpy==1.21
COPY . . 

ENTRYPOINT ["python3", "main.py"]
