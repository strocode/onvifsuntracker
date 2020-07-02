#!/usr/bin/env python


__author__ = 'Keith Bannister <k.bannister@ieee.org>'

from onvif import ONVIFCamera
from dvrip import DVRIPCam
import time
import datetime
import requests
import logging

from urllib.request import urlopen



def move(x,y,zoom):
    move_params = mycam.ptz.create_type('RelativeMove')
    move_params.ProfileToken = '000'
    move_params.Translation = {'PanTilt':{'x':x,'y':y}, 'Zoom': zoom}
    move_params.Speed = {'PanTilt':{'x':1,'y':1}, 'Zoom': 1}
    mycam.ptz.RelativeMove(move_params) # no command errors, but camera doesn't move.

def move_with_onvif():
    mycam = ONVIFCamera('192.168.2.3', 8899, 'admin', 'Th3Cloud5', '../venv/lib/python3.7/site-packages/wsdl/')
    ptz_service = mycam.create_ptz_service()
    move(mycam, -0.01, 0.01, 3)

class Camera:
    def __init__(self, address, user, password):
        self.dvrip = DVRIPCam(address, user, password)
        loginok = self.dvrip.login()
        if not loginok:
            raise ValueError('Login failed')

        self.onvif = ONVIFCamera(address, 8899, user, password,  '../venv/lib/python3.7/site-packages/wsdl/')
        self.onvif_media = self.onvif.create_media_service();

    def step(self, direction, sleep=0.1):
        assert direction in ('DirectionUp','DirectionDown','DirectionLeft','DirectionRight')
        self.dvrip.ptz(direction, preset=65535) # move
        time.sleep(sleep)
        self.dvrip.ptz(direction, preset=-1) # Stop

    def get_snapshot_uri(self):
        for profile in ('000','001','002'):
            snapshot_uri = self.onvif.media.GetSnapshotUri('000').Uri
            logging.info("Snapshot uri is: %s for profile %s", snapshot_uri, profile)
        return snapshot_uri

    def acquire(self, filename=None):
        if filename is None:
            filename = datetime.datetime.now().isoformat() + '.jpg'

        logging.info('Saving file to %s', filename)

        hdr = {'Accept':'*/*'}
        with open(filename, 'wb') as fout:
            '''
            r = requests.get(self.get_snapshot_uri(), stream=True, headers=hdr)
            if r.status_code == 200:
                with open(path, 'wb') as f:
                    r.raw.decode_content = True
                    shutil.copyfileobj(r.raw, fout)
            '''
            response = urlopen(self.get_snapshot_uri())
            data = response.read()
            fout.write(data)


    def close(self):
        self.dvrip.close()

def move_with_dvrip():
    try:

        print(cam.get(0x03fc, "SystemInfo"))
        print(cam.get(0x0550, "SystemFunction"))
        print(cam.get(0x0412, "fVideo.AudioSupportType"))
        print(cam.get(0x0412, "Detect.MotionDetect.[0]"))

        #cam.goToPositition();
        cam.set_info("General.AppBindFlag", {"bebinded": True})
        print(cam.ptz('DirectionUp', step=4)) # MOves all the way to the end, even though step=4
        time.sleep(10)
    finally:
        if cam:
            cam.close()

if __name__ == '__main__':
    cam = Camera('192.168.20.25', 'admin', 'Th3Cloud5')
    while True:
        cam.acquire()
        time.sleep(60)
        
    for round in range(10):
        cam.step('DirectionUp', 5)
        dir = 'DirectionDown'
        for step in range(10):
            cam.acquire(f'{dir}_r{round}_s{step}.jpg')
            cam.step(dir)

    cam.close()
