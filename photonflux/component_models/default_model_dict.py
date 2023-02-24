import functools

from photonflux.component_models.bends.bend_circular import bend_circular_model
from photonflux.component_models.bends.bend_euler import bend_euler_model
from photonflux.component_models.splitters.mmi1x2 import mmi1x2_model
from photonflux.component_models.waveguides.strip_waveguide import straight_model
from photonflux.component_models.laser.laser import laser_model
from photonflux.component_models.photodetector.photodetector import detector_model
from photonflux.component_models.heater.TiN_heater import straight_heater_metal_model
from photonflux.component_models.ring_resonators.passive_rings import ring_resonator_model
from photonflux.component_models.ring_resonators.pn_ring import ring_resonator_pn_model
from photonflux.component_models.pn_pin_junction.straight_pn import straight_pn

straight_pn_u_shaped = functools.partial(straight_pn, junction="U_doping", doping=1E17)
straight_pn_lateral = functools.partial(straight_pn, junction="lateral_doping", doping=1E17)
straight_pn_vertical = functools.partial(straight_pn, junction="vertical_doping", doping=1E17)

models_dict = {
    "bend_euler": bend_euler_model,
    "bend_circular" : bend_circular_model,
    "mmi1x2":mmi1x2_model,
    "straight":straight_model,
    "straight_heater_metal_undercut":straight_heater_metal_model,
    "straight_pin":straight_pn_lateral,
    "laser": laser_model,
    "detector": detector_model,
    "ring_single": ring_resonator_model,
    "ring_single_pn": ring_resonator_pn_model,
}