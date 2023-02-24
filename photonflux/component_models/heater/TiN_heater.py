import numpy as np
from ..Base_model import BaseModel
import pickle
import os

class straight_heater_metal_model(BaseModel):
    def __init__(self,info,settings):
        super().__init__(port_names=["o1","o2"], info=info, settings=settings)
        self.info_handler()
        
        if self.with_undercut == False:
            filename = os.path.dirname(__file__) + os.sep + "TiN_heater_sweep.pkl"
            with open(filename,'rb') as f:
                storage_dict = pickle.load(f)
                self.Ppi = storage_dict['Ppi'][0][0]
                #TODO: Add in handling of waveguide width and heater width

        elif self.with_undercut == True:
            filename = os.path.dirname(__file__) + os.sep + "TiN_heater_undercut.pkl"
            with open(filename,'rb') as f:
                storage_dict = pickle.load(f)
                self.Ppi = storage_dict['Ppi']
        self.alpha_dB_m = 300
        self.alpha = self.alpha_dB_m/4.34
        
        self.voltage = 0

    def info_handler(self):
        if self.info['resistance'] != None:
            self.resistance = self.info['resistance']
        else:
            self.resistance = 100
        try:
            self.with_undercut = self.settings['with_undercut']
        except:
            self.with_undercut = True
        self.length = self.settings['length']*1e-6
        return super().info_handler()

    def update_voltage(self,voltage):
        self.voltage = voltage

    def __call__(self):
        power_dissipated = self.voltage**2/self.resistance
        deltaphi = np.pi*power_dissipated/self.Ppi
        return {
            ("o1","o2"): np.exp(-self.alpha*self.length/2 + 1j*deltaphi),
            ("o2","o1"): np.exp(-self.alpha*self.length/2 + 1j*deltaphi)
        }