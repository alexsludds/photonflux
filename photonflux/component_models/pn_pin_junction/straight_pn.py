import numpy as np
from ..Base_model import BaseModel
import pickle
import os

class straight_pn(BaseModel):
    def __init__(self, junction, doping, info, settings):
        super().__init__(port_names=["o1","o2"], info=info, settings=settings)
        self.info_handler()
        filename = os.path.dirname(__file__) + os.sep + junction + ".pkl"
        with open(filename,'rb') as f:
            storage_dict = pickle.load(f)
        self.doping = doping
        self.voltage_sweep = storage_dict[self.doping]['voltage']
        self.voltage = 0
        self.neff_fit = lambda v: np.interp(v,storage_dict[self.doping]['voltage'],np.real(storage_dict[self.doping]['neff']))
        self.alpha_fit = lambda v: np.interp(v,storage_dict[self.doping]['voltage'],np.imag(storage_dict[self.doping]['neff']))

        self.width = 500e-9
        with open(os.path.dirname(__file__) + os.sep + "../waveguides/Si_strip_C_band.pkl",'rb') as f:
            mode_storage_dict = pickle.load(f)
            width_sweep = np.array([i['width'] for i in mode_storage_dict['TE0']])
            neff_sweep = np.array([np.real(i['neff']) for i in mode_storage_dict['TE0']])
            ng_sweep = np.array([np.real(i['ng']) for i in mode_storage_dict['TE0']])
        self.neff = np.interp(self.width,width_sweep,neff_sweep)
        self.ng = np.interp(self.width,width_sweep,ng_sweep)
        self.dneff_dlambda = (self.neff - self.ng)/(1.55e-6)

    def update_voltage(self,voltage):
        if (voltage <= np.max(self.voltage_sweep)) and (voltage >= np.min(self.voltage_sweep)):
            self.voltage = voltage
        else:
            print(f"Straight pn voltage {voltage} out of range: {np.min(self.voltage_sweep)} - {np.max(self.voltage_sweep)}")

    def info_handler(self):
        self.length = self.settings['length']*1e-6
        return super().info_handler()

    def __call__(self):
        neff = self.neff_fit(self.voltage) + 1j*self.alpha_fit(self.voltage)
        phi = 2*np.pi*(neff + (self.wl - 1.55e-6)*self.dneff_dlambda)*self.length/self.wl
        # phi = 2*np.pi*neff*self.length/self.wl
        transmission = np.exp(1j*phi)
        return {
            ("o1","o2"): transmission,
            ("o2","o1"): transmission,
        }