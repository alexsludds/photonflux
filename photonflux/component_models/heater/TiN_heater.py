import numpy as np
from ..Base_model import BaseModel

class straight_heater_metal_undercut_model(BaseModel):
    def __init__(self,info,settings):
        assert False 
        #TOOD, Create a default TiN heater
        super().__init__(port_names=["o1","o2"], info=info, settings=settings)
        self.alpha_dB_m = 300
        self.alpha = self.alpha_dB_m/4.34
        self.resistance = 100
        self.Ppi = 20e-3
        self.length = 100e-6
        self.voltage = 0
        self.info_handler()

    def info_handler(self):
        if self.info['resistance'] != None:
            self.resistance = self.info['resistance']
        print(self.resistance)
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

class straight_heater_metal_model(BaseModel):
    def __init__(self,info,settings):
        super().__init__(port_names=["o1","o2"], info=info, settings=settings)
        self.alpha_dB_m = 300
        self.alpha = self.alpha_dB_m/4.34
        self.resistance = 100
        self.Ppi = 20e-3
        self.length = 100e-6
        self.voltage = 0
        self.info_handler()

    def info_handler(self):
        if self.info['resistance'] != None:
            self.resistance = self.info['resistance']
        print(self.resistance)
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