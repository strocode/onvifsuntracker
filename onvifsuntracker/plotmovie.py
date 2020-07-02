#!/usr/bin/env python
"""
Template for making scripts to run from the command line

Copyright (C) Keith Bannister 2020
"""
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
import numpy as np
import os
import sys
import logging
import pandas as pd
from influxdb import DataFrameClient
import dateutil.parser
import datetime
import seaborn as sns
from PIL import Image
import pytz
import pysolar.solar as sun
import pysolar.radiation as radiation


__author__ = "Keith Bannister <keith.bannister@csiro.au>"

utcoff = datetime.timedelta(hours=-10)

latlong = (-33.474523, 151.345991)

def parse_file_date(f):
    return dateutil.parser.isoparse(f.replace('.jpg','')).astimezone().astimezone(datetime.timezone.utc)


class ImageFile:
    def __init__(self, filename):
        self.filename = filename

    @property
    def date(self):
        return parse_file_date(self.filename)

    def read_image(self):
        logging.info('Reading image %s', self.filename)
        #return mpimg.imread(self.filename)
        image = Image.open(self.filename)
        return image


def get_files(values):
    return [ImageFile(f) for f in values.files]

def get_altitude(dutc):
    return sun.get_altitude(latlong[0], latlong[1], dutc)

def get_radiation_direct(dutc):
    alt_deg = get_altitude(dutc)
    rad_wm2 = radiation.get_radiation_direct(dutc, alt_deg)
    return rad_wm2


class MoviePlotter:
    def __init__(self, fig, ax, weather, files):
        self.fig = fig
        self.ax = ax
        self.weather = weather
        self.files = files

    def init(self):
        weather = self.weather
        files = self.files
        self.img = self.ax[0].imshow(files[0].read_image())
        self.ax[1].plot(weather['time'], weather['solarradiation'], label='Measured')
        self.ax[1].set_xlabel('Time (UTC)')
        self.ax[1].set_ylabel('Insolation ($W/m^2$)')

        self.tline = self.ax[1].axvline(weather['time'][0])
        self.wline = self.ax[1].axhline(weather['solarradiation'][0])
        [self.solar_point] = self.ax[1].plot(weather['time'][0], weather['solarradiation'][0], 'ro', label='Current')

        direct_radiation = [get_radiation_direct(t.to_pydatetime()) for t in weather['time']]
        self.directline = self.ax[1].plot(weather['time'], direct_radiation, label='Direct radiation')
        self.ax[1].legend(loc='upper left')

    def __call__(self, file):
        self.img.set_data(file.read_image())
        d = pd.Timestamp(file.date.astimezone(datetime.timezone.utc).replace(tzinfo=None), tz='UTC')
        w = self.weather

        t = np.argmin(abs(w['time'] - d))
        dtime = w['time'][t]
        dsolar = w['solarradiation'][t]
        self.tline.set_xdata([dtime, dtime])
        self.wline.set_ydata([dsolar, dsolar])
        self.solar_point.set_data([dtime],[dsolar])

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Be verbose')
    parser.add_argument('-o','--output', help='Output file', default='movie.mp4')
    parser.add_argument('-s','--show', action='store_true', default=False, help='Show plot')
    parser.add_argument(dest='files', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


    os.makedirs(values.outdir, exist_ok=True)
    files = get_files(values)
    files = sorted(files, key=lambda f:f.date) # sort by date
    dates = [f.date for  f in files]
    fmt = '%Y-%m-%dT%H:%M:%S.%fZ'
    start = min(dates).strftime(fmt)
    end = max(dates).strftime(fmt)
    logging.info('Got dates %s %s - %s', min(dates).isoformat(), start, end)

    client = DataFrameClient('localhost', 8086, database='weather')
    weather = client.query(r"select * from weather where time >= $start and time <= $end",
            bind_params={'start':start, 'end':end})['weather']
    weather = weather.reset_index()
    weather.rename(columns={'index':'time'}, inplace=True)
    logging.info('Got %s data points for dates %s-%s', len(weather), start, end)

    fig, axs = pylab.subplots(2,1, figsize=[6,8])
    plotter = MoviePlotter(fig, axs, weather, files)
    ani = animation.FuncAnimation(fig, plotter, files,
        init_func=plotter.init, blit=False, interval=50,
        repeat=False)
    if values.output:
        ani.save(values.output)

    if values.show:
        pylab.show()



if __name__ == '__main__':
    _main()
