import numpy as np
from ..Base_model import BaseModel
import pickle
import os

class straight_model(BaseModel):
    def __init__(self,info, settings):
        super().__init__(port_names=["o1","o2"], info=info, settings=settings)
        self.alpha_dB_m = 300 #3dB/cm Si waveguide loss
        self.alpha = self.alpha_dB_m/4.34 #alpha in units of 1/m
        self.length = 1e-6
        with open(os.path.dirname(__file__) + os.sep + "Si_strip_C_band.pkl",'rb') as f:
            storage_dict = pickle.load(f)
            width_sweep = np.array([i['width'] for i in storage_dict['TE0']])
            neff_sweep = np.array([np.real(i['neff']) for i in storage_dict['TE0']])
            ng_sweep = np.array([np.real(i['ng']) for i in storage_dict['TE0']])
        self.info_handler()
        self.neff = np.interp(self.width,width_sweep,neff_sweep)
        self.ng = np.interp(self.width,width_sweep,ng_sweep)
        self.dneff_dlambda = (self.neff - self.ng)/(1.55e-6)

    def info_handler(self):
        self.length = self.info['length']*1e-6
        self.width = self.info['width']*1e-6
        return super().info_handler()
    
    def __call__(self):
        phi = 2 * np.pi * (self.neff + (self.wl - 1.55e-6)*self.dneff_dlambda) * self.length/self.wl
        return {
            ("o1","o2"): np.exp(-self.alpha*self.length/2 + 1j*phi),
            ("o2","o1"): np.exp(-self.alpha*self.length/2 + 1j*phi),
        }