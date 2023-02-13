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

# Models return a dictionary with elements of (port from, port to): propagation value
class bend_euler_model(BaseModel):
    def __init__(self, info,settings):
        super().__init__(port_names=["o1","o2"], info=info, settings=settings)
        self.neff = 2.4
        self.alpha_dB_m = 300
        self.alpha = self.alpha_dB_m/4.34
        self.info_handler()
    
    def info_handler(self):
        self.length = self.info['length']*1e-6
        return super().info_handler()

    def __call__(self):
        #Bend euler has ports o1, o2
        phi = 2 * np.pi * self.neff * self.length/self.wl
        return {
            ("o1","o2"): np.exp(-self.alpha*self.length/2 + 1j*phi),
            ("o2","o1"): np.exp(-self.alpha*self.length/2 + 1j*phi),
        }

class mmi1x2_model(BaseModel):
    def __init__(self, info, settings):
        super().__init__(port_names=["o1","o2","o3"], info=info, settings=settings)
        self.info_handler()

    def info_handler(self):
        return super().info_handler()

    def __call__(self):
        #MMI1x2 has ports o1 as input, and  o2, o3 as output
        return {
            ("o1","o2"): 1/np.sqrt(2),
            ("o1","o3"): 1/np.sqrt(2),
            ("o2","o1"): 1/np.sqrt(2),
            ("o3","o1"): 1/np.sqrt(2)
        }

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

class straight_heater_metal_undercut_model(BaseModel):
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

class ring_resonator_model(BaseModel):
    def __init__(self,info, settings):
        super().__init__(port_names=["o1","o2"], info=info, settings=settings)
        self.alpha_dB_m = 300
        self.alpha_m = self.alpha_dB_m/4.34
        self.neff = 2.4
        self.radius = 5e-6
        self.Lrt = 2*np.pi*self.radius
        self.t = np.exp(-self.alpha_m*self.Lrt/2)
        self.deltaphi = 0 #On resonance
        self.heater_voltage = 0
        self.heater_resistance = 100
        self.Ppi = 20e-3
        self.info_handler()

    def info_handler(self):
        return super().info_handler()

    def update_voltage(self,new_voltage):
        self.heater_voltage = new_voltage
        self.deltaphi = np.pi/self.Ppi * self.heater_voltage**2 / self.heater_resistance

    def __call__(self):
        phi = 2*np.pi*self.neff*self.Lrt/self.wl
        num = self.t-np.exp(-self.alpha_m*self.Lrt/2)*np.exp(1j*(phi+self.deltaphi))
        denom = 1-self.t*np.exp(-self.alpha_m*self.Lrt/2)*np.exp(1j*(phi+self.deltaphi))
        field_transmission = num/denom

        return {
            ("o1","o2"): field_transmission,
            #TODO: If I add in the below line I 
            # ("o2","o1"): field_transmission 
        }

class laser_model(BaseModel):
    def __init__(self):
        super().__init__(port_names=["o1"])

    def __call__(self):
        return {
            ("o1"): 1
        }

class detector_model(BaseModel):
    def __init__(self):
        super().__init__(port_names=["o1"])

    def __call__(self):
        return {
            ("o1"): 1
        }

models_dict = {
    "bend_euler": bend_euler_model,
    "mmi1x2":mmi1x2_model,
    "straight":straight_model,
    "straight_heater_metal_undercut":straight_heater_metal_undercut_model,
    "laser": laser_model,
    "detector": detector_model,
    "ring_single_pn": ring_resonator_model,
}