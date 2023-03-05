import numpy as np
from ..Base_model import BaseModel
import pickle
import os

class ring_resonator_model(BaseModel):
    def __init__(self,info, settings):
        super().__init__(port_names=["o1","o2"], info=info, settings=settings)
        self.alpha_dB_m = 300
        self.alpha_m = self.alpha_dB_m/4.34
        self.radius = 5e-6
        self.width = 500e-9 #TODO: Find waveguide width in the info handler
        self.Lrt = 2*np.pi*self.radius
        self.t = np.exp(-self.alpha_m*self.Lrt/2)
        self.deltaphi = 0 #On resonance
        self.heater_voltage = 0
        self.heater_resistance = 100
        self.Ppi = 20e-3
        self.info_handler()
        with open(os.path.dirname(__file__) + os.sep + "../waveguides/Si_strip_C_band.pkl",'rb') as f:
            storage_dict = pickle.load(f)
            width_sweep = np.array([i['width'] for i in storage_dict['TE0']])
            neff_sweep = np.array([np.real(i['neff']) for i in storage_dict['TE0']])
            ng_sweep = np.array([np.real(i['ng']) for i in storage_dict['TE0']])
        self.neff = np.interp(self.width,width_sweep,neff_sweep)
        self.ng = np.interp(self.width,width_sweep,ng_sweep)
        self.dneff_dlambda = (self.neff - self.ng)/(1.55e-6)

    def info_handler(self):
        return super().info_handler()

    def update_voltage(self,new_voltage):
        self.heater_voltage = new_voltage
        self.deltaphi = np.pi/self.Ppi * self.heater_voltage**2 / self.heater_resistance

    def __call__(self):
        phi = 2*np.pi*(self.neff + (self.wl - 1.55e-6)*self.dneff_dlambda)*self.Lrt/self.wl
        num = self.t-np.exp(-self.alpha_m*self.Lrt/2)*np.exp(1j*(phi+self.deltaphi))
        denom = 1-self.t*np.exp(-self.alpha_m*self.Lrt/2)*np.exp(1j*(phi+self.deltaphi))
        field_transmission = num/denom

        return {
            # ("o1","o2"): field_transmission,
            ("o2","o1"): field_transmission 
        }