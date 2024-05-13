FROM waggle/plugin-base:1.1.1-base

WORKDIR /app

RUN apt-get update -y
RUN apt update -y
RUN apt-get install -y --no-install-recommends python3-netcdf4 gcc
RUN apt-get install -y libpython3-dev
COPY . . 
RUN git config --global --add safe.directory /app
RUN pip3 install -r requirements.txt
RUN pip3 install git+https://github.com/rcjackson/HighIQ.git


ENTRYPOINT ["python3", "main.py"]
