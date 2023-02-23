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
        self.voltage_sweep = storage_dict[doping][0]
        self.voltage = 0
        self.neff_fit = lambda v: np.interp(v,storage_dict[self.doping][0],np.real(storage_dict[self.doping][1]))
        self.alpha_fit = lambda v: np.interp(v,storage_dict[self.doping][0],np.imag(storage_dict[self.doping][1]))

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
        transmission = np.exp(1j*2*np.pi*neff*self.length/self.wl)
        return {
            ("o1","o2"): transmission,
            ("o2","o1"): transmission,
        }