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
    rad_wm2 = radiation.get_radiation_direct(dutc, alt_deg)*np.sin(np.radians(alt_deg))
    return rad_wm2

def hide_axes(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)



class MoviePlotter:
    def __init__(self, fig, ax, weather, files):
        self.fig = fig
        self.ax = ax
        self.weather = weather
        self.files = files
        self.lookahead = datetime.timedelta(minutes=15)


    def init(self):
        weather = self.weather
        files = self.files
        self.direct_radiation = np.array([get_radiation_direct(t.to_pydatetime()) for t in weather['time']])
        self.coeff = weather['solarradiation'].max()/self.direct_radiation.max()
        self.diff = self.direct_radiation*self.coeff - weather['solarradiation']

        t0imgax = self.ax[0,0]
        t5imgax = self.ax[0,1]
        diffimgax = self.ax[0,2]
        pltax = self.ax[1,2]
        self.pltzoomax = self.ax[1,0]
        self.pltdiffax = self.ax[1,1]

        self.main_lines = self.plot_weather(pltax)
        self.zoom_lines = self.plot_weather(self.pltzoomax)
        self.diff_lines = self.plot_weather_diff(self.pltdiffax)

        times = weather['time']
        self.update_zoom(times[0])

        self.img = t0imgax.imshow(files[0].read_image())
        t0imgax.set_title('T=0')
        self.t5img = t5imgax.imshow(files[0].read_image())
        t5imgax.set_title('T=5min')
        self.diffimg = diffimgax.imshow(files[0].read_image())
        diffimgax.set_title('Difference T5min-T0min')
        hide_axes(t0imgax)
        hide_axes(t5imgax)
        hide_axes(diffimgax)
        self.update_imgs(0)


    def update_zoom(self, t):
        toff = self.lookahead
        print('Updating zoom', t, toff, t-toff, t+toff)
        self.pltzoomax.set_xlim(t - toff, t + toff)
        self.pltdiffax.set_xlim(t - toff, t + toff)

    def update_imgs(self, ifile):
        file = self.files[ifile]
        currimg = file.read_image()
        self.img.set_data(currimg)
        nextfile = ifile + 5 # TODO: actually calculate times

        if nextfile < len(self.files):
            nextimg = self.files[nextfile].read_image()
            self.t5img.set_data(nextimg)
            currb = np.asarray(currimg);
            nextb = np.asarray(nextimg);
            self.diffimg.set_data(nextb - currb)

    def plot_weather(self, pltax):
        weather = self.weather
        coeff = self.coeff
        time = weather['time']
        measuredline = pltax.plot(time, weather['solarradiation'], label='Measured')
        directline = pltax.plot(time, self.direct_radiation*coeff, label=f'Direct radiation x{coeff:0.2f}')
        #ax2 = pltax.twinx()
        #ax2.plot(time, self.diff, 'g', label='Difference')
        [solar_point] = pltax.plot(time[0], weather['solarradiation'][0], 'ro', label='Current')
        tline = pltax.axvline(time[0])
        wline = pltax.axhline(weather['solarradiation'][0])
        pltax.set_xlabel('Date (UTC)')
        pltax.set_ylabel('Insolation ($W/m^2$)')
        plt.xticks(rotation=45)
        pltax.legend(loc='lower left')

        return (directline, measuredline, solar_point, tline, wline)

    def plot_weather_diff(self, pltax):
        weather = self.weather
        diff = self.diff
        measuredline = pltax.plot(weather['time'], diff , label=f'Diffference')
        directline = None
        [solar_point] = pltax.plot(weather['time'][0], diff[0], 'ro', label='Current')
        tline = pltax.axvline(weather['time'][0])
        wline = pltax.axhline(diff[0])
        pltax.set_xlabel('Date (UTC)')
        pltax.set_ylabel('Insolation ($W/m^2$)')
        plt.xticks(rotation=45)
        pltax.legend(loc='lower left')
        return (directline, measuredline, solar_point, tline, wline)

    def tnow(self, d):
        w = self.weather

        return np.argmin(abs(w['time'] - d))

    def update_weather(self, lines, tnow):
        (directline, measuredline, solar_point, tline, wline) = lines
        w = self.weather

        dtime = measuredline[0].get_xdata()[tnow]
        dsolar = measuredline[0].get_ydata()[tnow]
        tline.set_xdata([dtime, dtime])
        wline.set_ydata([dsolar, dsolar])
        solar_point.set_data([dtime],[dsolar])

    def __call__(self, ifile):
        file = self.files[ifile]
        d = pd.Timestamp(file.date.astimezone(datetime.timezone.utc).replace(tzinfo=None), tz='UTC')
        tnow = self.tnow(d)
        w = self.weather

        self.update_weather(self.main_lines, tnow)
        self.update_weather(self.zoom_lines, tnow)
        self.update_weather(self.diff_lines, tnow)
        self.update_zoom(w['time'][tnow])
        self.update_imgs(ifile)


def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Be verbose')
    parser.add_argument('-o','--output', help='Output file')
    parser.add_argument('-s','--show', action='store_true', default=False, help='Show plot')
    parser.add_argument(dest='files', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

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

    fig, axs = pylab.subplots(2,3, figsize=[12,9])
    fig.subplots_adjust(left=0.06, right=0.98, top=0.98, hspace=0, wspace=0.22)
    #fig.tight_layout()
    plotter = MoviePlotter(fig, axs, weather, files)
    ani = animation.FuncAnimation(fig, plotter, range(len(files)),
        init_func=plotter.init, blit=False, interval=200,
        repeat=False)

    if values.output:
        logging.info('Writing output to %s', values.output)
        ani.save(values.output, fps=10)

    if values.show:
        pylab.show()



if __name__ == '__main__':
    _main()
