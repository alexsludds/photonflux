import numpy as np
from ..Base_model import BaseModel

class straight_model(BaseModel):
    def __init__(self,info, settings):
        super().__init__(port_names=["o1","o2"], info=info, settings=settings)
        self.alpha_dB_m = 300
        self.alpha = self.alpha_dB_m/4.34 #alpha in units of 1/m
        self.length = 1e-6
        self.neff = 2.4
        self.info_handler()

    def info_handler(self):
        self.length = self.info['length']*1e-6
        self.width = self.info['width']*1e-6
        return super().info_handler()
    
    def __call__(self):
        phi = 2 * np.pi * self.neff * self.length/self.wl
        return {
            ("o1","o2"): np.exp(-self.alpha*self.length/2 + 1j*phi),
            ("o2","o1"): np.exp(-self.alpha*self.length/2 + 1j*phi),
        }