import numpy as np

class BaseModel():
    def __init__(self,port_names, info={}, settings={}, wl = 1.55e-6):
        self.port_names = port_names
        self.info = info
        self.settings = settings
        self.wl = wl

    def info_handler(self):
        #A baseline layout information handler
        # How do we deal with self.info and self.settings
        pass

    def update_time(self):
        pass

    def update_wavelength(self,wl):
        self.wl = wl