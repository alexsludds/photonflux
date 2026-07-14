"""Auto-generated from Verilog-A MIR — do not edit by hand.

Regenerate with: ``python -m circulax.va <path/to/device.va>``
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from circulax.components.base_component import PhysicsReturn, Signals, States
from bosdi.circulax.va_component import va_component

def _SKY130_PFET_01V8_TT_setup(l: float=1.5e-07, w: float=2e-06, nf: float=1.0, _min: float=0.0, ad: float=0.0, ps: float=0.0, pd: float=0.0, sa: float=0.0, sb: float=0.0, sd: float=0.0, delvto: float=0.0, _ckt_gmin: float=1e-12, off: float=0.0, _temperature: float=300.15, _mfactor: float=1.0) -> jnp.ndarray:
    i_v5750 = jnp.where((4.23e-09 == 0.0) | ~jnp.isfinite(4.23e-09), 0.0, jnp.divide(3.4531302e-11, jnp.where((4.23e-09 == 0.0) | ~jnp.isfinite(4.23e-09), 1.0, 4.23e-09)))
    i_v5784 = jnp.divide(1.03594e-10, 3.4531302e-11) * 4.23e-09
    i_v5785 = jnp.sqrt(jnp.maximum(i_v5784, 1e-300))
    i_v5824 = 8.617342301212761e-05 * _temperature
    i_v5873 = _temperature - 303.15
    i_v5907 = 0.0020386 * i_v5873
    i_v5908 = jnp.where(0.6587 < 0.1, 0.1, 0.6587) - i_v5907
    i_v5911 = jnp.where(i_v5908 < 0.01, 0.01, i_v5908)
    i_v5912 = jnp.where(0.6587 < 0.1, 0.1, 0.6587) - i_v5907
    i_v5915 = jnp.where(i_v5912 < 0.01, 0.01, i_v5912)
    i_v5916 = 0.001246 * i_v5873
    i_v5917 = jnp.where(0.7418 < 0.1, 0.1, 0.7418) - i_v5916
    i_v5920 = jnp.where(i_v5917 <= 0.01, 0.01, i_v5917)
    i_v5921 = jnp.where(0.7418 < 0.1, 0.1, 0.7418) - i_v5916
    i_v5924 = jnp.where(i_v5921 <= 0.01, 0.01, i_v5921)
    i_v5925 = 0.0
    i_v5926 = jnp.where(0.7418 < 0.1, 0.1, 0.7418) - i_v5925
    i_v5929 = jnp.where(i_v5926 <= 0.01, 0.01, i_v5926)
    i_v5930 = jnp.where(0.7418 < 0.1, 0.1, 0.7418) - i_v5925
    i_v5933 = jnp.where(i_v5930 <= 0.01, 0.01, i_v5930)
    i_v5972 = l + 0.0
    i_v5974 = jnp.power(jnp.maximum(i_v5972, 0.0), 1.0)
    i_v5973 = jnp.where((nf == 0.0) | ~jnp.isfinite(nf), 0.0, jnp.divide(w, jnp.where((nf == 0.0) | ~jnp.isfinite(nf), 1.0, nf))) + 0.0
    i_v5975 = jnp.power(jnp.maximum(i_v5973, 0.0), 1.0)
    i_v5979 = i_v5974 * i_v5975
    i_v6006 = i_v5972 - 2.0 * (-1.3994e-08 + (jnp.where((i_v5974 == 0.0) | ~jnp.isfinite(i_v5974), 0.0, jnp.divide(0.0, jnp.where((i_v5974 == 0.0) | ~jnp.isfinite(i_v5974), 1.0, i_v5974))) + jnp.where((i_v5975 == 0.0) | ~jnp.isfinite(i_v5975), 0.0, jnp.divide(0.0, jnp.where((i_v5975 == 0.0) | ~jnp.isfinite(i_v5975), 1.0, i_v5975))) + jnp.where((i_v5979 == 0.0) | ~jnp.isfinite(i_v5979), 0.0, jnp.divide(0.0, jnp.where((i_v5979 == 0.0) | ~jnp.isfinite(i_v5979), 1.0, i_v5979)))))
    i_v5989 = jnp.power(jnp.maximum(i_v5972, 0.0), 1.0)
    i_v5990 = jnp.power(jnp.maximum(i_v5973, 0.0), 1.0)
    i_v5994 = i_v5989 * i_v5990
    i_v6010 = i_v5973 - 2.0 * (7.3039e-09 + (jnp.where((i_v5989 == 0.0) | ~jnp.isfinite(i_v5989), 0.0, jnp.divide(0.0, jnp.where((i_v5989 == 0.0) | ~jnp.isfinite(i_v5989), 1.0, i_v5989))) + jnp.where((i_v5990 == 0.0) | ~jnp.isfinite(i_v5990), 0.0, jnp.divide(0.0, jnp.where((i_v5990 == 0.0) | ~jnp.isfinite(i_v5990), 1.0, i_v5990))) + jnp.where((i_v5994 == 0.0) | ~jnp.isfinite(i_v5994), 0.0, jnp.divide(0.0, jnp.where((i_v5994 == 0.0) | ~jnp.isfinite(i_v5994), 1.0, i_v5994)))))
    i_v6014 = i_v5972 - 2.0 * (-1.3994e-08 + (jnp.where((i_v5974 == 0.0) | ~jnp.isfinite(i_v5974), 0.0, jnp.divide(0.0, jnp.where((i_v5974 == 0.0) | ~jnp.isfinite(i_v5974), 1.0, i_v5974))) + jnp.where((i_v5975 == 0.0) | ~jnp.isfinite(i_v5975), 0.0, jnp.divide(0.0, jnp.where((i_v5975 == 0.0) | ~jnp.isfinite(i_v5975), 1.0, i_v5975))) + jnp.where((i_v5979 == 0.0) | ~jnp.isfinite(i_v5979), 0.0, jnp.divide(0.0, jnp.where((i_v5979 == 0.0) | ~jnp.isfinite(i_v5979), 1.0, i_v5979)))))
    i_v6002 = jnp.where((i_v5989 == 0.0) | ~jnp.isfinite(i_v5989), 0.0, jnp.divide(0.0, jnp.where((i_v5989 == 0.0) | ~jnp.isfinite(i_v5989), 1.0, i_v5989))) + jnp.where((i_v5990 == 0.0) | ~jnp.isfinite(i_v5990), 0.0, jnp.divide(0.0, jnp.where((i_v5990 == 0.0) | ~jnp.isfinite(i_v5990), 1.0, i_v5990))) + jnp.where((i_v5994 == 0.0) | ~jnp.isfinite(i_v5994), 0.0, jnp.divide(0.0, jnp.where((i_v5994 == 0.0) | ~jnp.isfinite(i_v5994), 1.0, i_v5994)))
    i_v6018 = i_v5973 - 2.0 * (7.3039e-09 + i_v6002)
    i_v6022 = i_v5973 - 2.0 * (7.3039e-09 + i_v6002)
    i_v6023 = 2.0 == 1
    i_v6034, i_v6033, i_v6032 = jax.tree_util.tree_map(lambda _t, _f: jnp.where(i_v6023, _t, _f), (jnp.where((i_v6006 == 0.0) | ~jnp.isfinite(i_v6006), 0.0, jnp.divide(1e-06, jnp.where((i_v6006 == 0.0) | ~jnp.isfinite(i_v6006), 1.0, i_v6006))), jnp.where((i_v6010 == 0.0) | ~jnp.isfinite(i_v6010), 0.0, jnp.divide(1e-06, jnp.where((i_v6010 == 0.0) | ~jnp.isfinite(i_v6010), 1.0, i_v6010))), jnp.where((i_v6006 * i_v6010 == 0.0) | ~jnp.isfinite(i_v6006 * i_v6010), 0.0, jnp.divide(1e-12, jnp.where((i_v6006 * i_v6010 == 0.0) | ~jnp.isfinite(i_v6006 * i_v6010), 1.0, i_v6006 * i_v6010)))), (jnp.where((i_v6006 == 0.0) | ~jnp.isfinite(i_v6006), 0.0, jnp.divide(1.0, jnp.where((i_v6006 == 0.0) | ~jnp.isfinite(i_v6006), 1.0, i_v6006))), jnp.where((i_v6010 == 0.0) | ~jnp.isfinite(i_v6010), 0.0, jnp.divide(1.0, jnp.where((i_v6010 == 0.0) | ~jnp.isfinite(i_v6010), 1.0, i_v6010))), jnp.where((i_v6006 * i_v6010 == 0.0) | ~jnp.isfinite(i_v6006 * i_v6010), 0.0, jnp.divide(1.0, jnp.where((i_v6006 * i_v6010 == 0.0) | ~jnp.isfinite(i_v6006 * i_v6010), 1.0, i_v6006 * i_v6010)))))
    i_v6040 = 0.00013 + 0.0 + 0.0 + 0.0
    i_v6046 = 0.00078 + 0.0 + 0.0 + 0.0
    i_v6052 = 0.0 + 0.0 + 0.0 + 0.0
    i_v6058 = 1e-05 + 0.0 + 0.0 + 0.0
    i_v6070 = 1.5e-07 + 0.0 + 0.0 + 0.0
    i_v6088 = 10.1659 + -1.9291e-06 * i_v6034 + -1.46366e-05 * i_v6033 + 3.04425e-12 * i_v6032
    i_v6094 = 44.9044 + -9.07958e-06 * i_v6034 + -7.27016e-05 * i_v6033 + 1.51211e-11 * i_v6032
    i_v6112 = 2.4576 + -5.06181e-07 * i_v6034 + -4.09205e-06 * i_v6033 + 8.51098e-13 * i_v6032
    i_v6142 = 1e+23 + 0.0 + 0.0 + 0.0
    i_v6184 = 0.0650646 + -1.37909e-07 * i_v6034 + 0.0 + 0.0
    i_v6190 = 0.0 + 0.0 + 0.0 + 0.0
    i_v6202 = -0.133589 + 2.4186e-09 * i_v6034 + 0.0 + 0.0
    i_v6208 = -15.845 + 0.0 + 0.0 + 0.0
    i_v6214 = 2.0 + 0.0 + 0.0 + 0.0
    i_v6250 = 4.4955 + 0.0 + 0.0 + 0.0
    i_v6256 = 0.294 + 0.0 + 0.0 + 0.0
    i_v6262 = 0.015 + 0.0 + 0.0 + 0.0
    i_v6268 = -4.9772 + 0.0 + 0.0 + 0.0
    i_v6274 = 1147200.0 + 0.0 + 0.0 + 0.0
    i_v6280 = -0.00896 + 0.0 + 0.0 + 0.0
    i_v6388 = 0.43 + 0.0 + 0.0 + 0.0
    i_v6406 = 0.43 + 0.0 + 0.0 + 0.0
    i_v6424 = 5.36464e-09 + -1.05306e-15 * i_v6034 + -8.23977e-15 * i_v6033 + 1.71377e-21 * i_v6032
    i_v6430 = 1000000000.0 + 0.0 + 0.0 + 0.0
    i_v6436 = 300.0 + 0.0 + 0.0 + 0.0
    i_v6442 = 0.1 + 0.0 + 0.0 + 0.0
    i_v6580 = 0.0 + 0.0 + 0.0 + 0.0
    i_v6586 = 0.0 + 0.0 + 0.0 + 0.0
    i_v6592 = 0.0 + 0.0 + 0.0 + 0.0
    i_v6598 = 0.01 + 0.0 + 0.0 + 0.0
    i_v6628 = -0.32348 + 0.0 + 0.0 + 0.0
    i_v6646 = -1.63437 + 3.23633e-07 * i_v6034 + 2.16636e-06 * i_v6033 + -4.50576e-13 * i_v6032
    i_v6652 = -0.526166 + 2.40654e-07 * i_v6034 + 3.41794e-06 * i_v6033 + -7.10891e-13 * i_v6032
    i_v6670 = 0.918161 + -1.91007e-07 * i_v6034 + -1.53109e-06 * i_v6033 + 3.18449e-13 * i_v6032
    i_v6676 = 1484550000.0 + -142.377 * i_v6034 + -1140.04 * i_v6033 + 0.000237114 * i_v6032
    i_v6682 = 1.12801e-08 + -3.75035e-16 * i_v6034 + -1.47854e-15 * i_v6033 + 3.0752e-22 * i_v6032
    i_v6688 = 0.0 + 0.0 + 0.0 + 0.0
    i_v6700 = -5.722e-09 + 0.0 + 0.0 + 0.0
    i_v6706 = -1.7864e-08 + 0.0 + 0.0 + 0.0
    i_v6736 = 18.1615 + -1.99271e-06 * i_v6034 + -5.65065e-06 * i_v6033 + 1.17527e-12 * i_v6032
    i_v6742 = 5.36464e-09 + -1.05306e-15 * i_v6034 + -8.23977e-15 * i_v6033 + 1.71377e-21 * i_v6032
    i_v6748 = 1000000000.0 + 0.0 + 0.0 + 0.0
    i_v6754 = 300.0 + 0.0 + 0.0 + 0.0
    i_v6760 = 0.1 + 0.0 + 0.0 + 0.0
    i_v6766 = 0.43 + 0.0 + 0.0 + 0.0
    i_v6790 = 0.43 + 0.0 + 0.0 + 0.0
    i_v6808 = 0.35 + 0.0 + 0.0 + 0.0
    i_v6850 = 1.1 + 0.0 + 0.0 + 0.0
    i_v6856 = 1.0 + 0.0 + 0.0 + 0.0
    i_v6868 = 12.0 + 0.0 + 0.0 + 0.0
    i_v6880 = 0.0 + 0.0 + 0.0 + 0.0
    i_v6934 = -0.14469 + 0.0 + 0.0 + 0.0
    i_v5781 = jnp.divide(_temperature, 303.15)
    i_v6965 = i_v5781 - 1.0
    i_v6970 = 0.0 == 0
    i_v6316 = -3.40842e-18 + 1.13491e-24 * i_v6034 + 6.55112e-24 * i_v6033 + -1.36256e-30 * i_v6032
    i_v6322 = -9.4711e-21 + 8.36468e-26 * i_v6034 + 9.04834e-25 * i_v6033 + -1.88195e-31 * i_v6032
    i_v6995 = 0.0 == 3
    i_v7037 = jnp.where(i_v6970, i_v6316 + i_v6322 * i_v6965, jnp.where(i_v6995, i_v6316 * jnp.power(jnp.maximum(i_v5781, 0.0), i_v6322), i_v6316 * (1.0 + i_v6322 * i_v5873)))
    i_v6484 = 0.0 + 0.0 + 0.0 + 0.0
    i_v7038 = jnp.where(i_v6970, i_v6484, jnp.where(i_v6995, i_v6484 * jnp.power(jnp.maximum(i_v5781, 0.0), 0.0 + 0.0 + 0.0 + 0.0), i_v6484))
    i_v6328 = 7.50245e-13 + -1.03558e-19 * i_v6034 + 7.59079e-19 * i_v6033 + -1.57879e-25 * i_v6032
    i_v6334 = -3.4095e-11 + 7.21625e-18 * i_v6034 + 1.2326e-32 * i_v6033 + 5.87747e-39 * i_v6032
    i_v7039 = jnp.where(i_v6970, i_v6328 + i_v6334 * i_v6965, jnp.where(i_v6995, i_v6328 * jnp.power(jnp.maximum(i_v5781, 0.0), i_v6334), i_v6328 * (1.0 + i_v6334 * i_v5873)))
    i_v6304 = 3.21092e-09 + -1.13376e-15 * i_v6034 + -6.4009e-15 * i_v6033 + 1.33131e-21 * i_v6032
    i_v6310 = 1.0474e-09 + -1.89846e-16 * i_v6034 + -1.81764e-15 * i_v6033 + 3.78047e-22 * i_v6032
    i_v7040 = jnp.where(i_v6970, i_v6304 + i_v6310 * i_v6965, jnp.where(i_v6995, i_v6304 * jnp.power(jnp.maximum(i_v5781, 0.0), i_v6310), i_v6304 * (1.0 + i_v6310 * i_v5873)))
    i_v6076 = 185247.0 + -0.0183494 * i_v6034 + 0.227307 * i_v6033 + -4.72771e-08 * i_v6032
    i_v6082 = 731816.0 + -0.138 * i_v6034 + -0.817538 * i_v6033 + 1.70038e-07 * i_v6032
    i_v7043 = jnp.where(i_v6970, i_v6076 - i_v6082 * i_v6965, i_v6076 * (1.0 - i_v6082 * i_v5873))
    i_v6634 = 0.0 + 0.0 + 0.0 + 0.0
    i_v6979 = i_v6634 * i_v6965
    i_v7021 = 1.0 + i_v6634 * i_v5873
    i_v7046 = jnp.where(i_v6970, 0.0 + i_v6979, 0.0)
    i_v6969 = jnp.power(jnp.maximum(i_v6022 * 1000000.0, 0.0), 1.0 + 0.0 + 0.0 + 0.0) * nf
    i_v7055 = jnp.where((i_v6969 == 0.0) | ~jnp.isfinite(i_v6969), 0.0, jnp.divide(jnp.where(i_v7046 < 0.0, 0.0, i_v7046), jnp.where((i_v6969 == 0.0) | ~jnp.isfinite(i_v6969), 1.0, i_v6969)))
    i_v7044 = jnp.where(i_v6970, 0.0 + i_v6979, 0.0)
    i_v7063 = jnp.where((i_v6969 == 0.0) | ~jnp.isfinite(i_v6969), 0.0, jnp.divide(jnp.where(i_v7044 < 0.0, 0.0, i_v7044), jnp.where((i_v6969 == 0.0) | ~jnp.isfinite(i_v6969), 1.0, i_v6969)))
    i_v7084 = (0.0 + 0.0 + 0.0 + 0.0) * (1.0 + (0.0 + 0.0 + 0.0 + 0.0) * i_v5873)
    i_v7090 = -4.66791 + 1.4044e-06 * i_v6034 + 5.17521e-06 * i_v6033 + -1.07638e-12 * i_v6032 + jnp.divide((0.0 + 0.0 + 0.0 + 0.0) * i_v5873, 303.15)
    i_v7113 = 0.0
    i_v6124 = 1.7e+17 + 0.0 + 0.0 + 0.0
    i_v5678 = 0 == 0
    i_v5822 = jnp.where(i_v5678, 18512213434.84149, 14946424623.46526)
    i_v7125 = 0.026123473186126484 * jnp.log(jnp.maximum(jnp.where((i_v5822 == 0.0) | ~jnp.isfinite(i_v5822), 0.0, jnp.divide(i_v6124, jnp.where((i_v5822 == 0.0) | ~jnp.isfinite(i_v5822), 1.0, i_v5822))), 1e-300)) + (0.0 + 0.0 + 0.0 + 0.0) + 0.4
    i_v7127 = jnp.sqrt(jnp.maximum(i_v7125, 1e-300))
    i_v7134 = jnp.sqrt(jnp.maximum(jnp.where((1.602176462e-19 * i_v6124 * 1000000.0 == 0.0) | ~jnp.isfinite(1.602176462e-19 * i_v6124 * 1000000.0), 0.0, jnp.divide(2.0 * 1.03594e-10, jnp.where((1.602176462e-19 * i_v6124 * 1000000.0 == 0.0) | ~jnp.isfinite(1.602176462e-19 * i_v6124 * 1000000.0), 1.0, 1.602176462e-19 * i_v6124 * 1000000.0))), 1e-300)) * i_v7127
    i_v7144 = jnp.where(i_v5678, jnp.sqrt(jnp.maximum(3.0 * i_v6070 * 4.23e-09, 1e-300)), jnp.sqrt(jnp.maximum(3.0 * i_v6070 * 4.23e-09, 1e-300)))
    i_v6130 = 1e+20 + 0.0 + 0.0 + 0.0
    i_v7149 = 0.026123473186126484 * jnp.log(jnp.maximum(jnp.where((i_v5822 * i_v5822 == 0.0) | ~jnp.isfinite(i_v5822 * i_v5822), 0.0, jnp.divide(i_v6130 * i_v6124, jnp.where((i_v5822 * i_v5822 == 0.0) | ~jnp.isfinite(i_v5822 * i_v5822), 1.0, i_v5822 * i_v5822))), 1e-300))
    i_v7153 = 0.026123473186126484 * jnp.log(jnp.maximum(jnp.where((i_v6130 == 0.0) | ~jnp.isfinite(i_v6130), 0.0, jnp.divide(i_v6142, jnp.where((i_v6130 == 0.0) | ~jnp.isfinite(i_v6130), 1.0, i_v6130))), 1e-300))
    i_v7172 = jnp.sqrt(jnp.maximum(jnp.where((i_v7125 == 0.0) | ~jnp.isfinite(i_v7125), 0.0, jnp.divide(jnp.divide(1.602176462e-19 * 1.03594e-10 * i_v6124 * 1000000.0, 2.0), jnp.where((i_v7125 == 0.0) | ~jnp.isfinite(i_v7125), 1.0, i_v7125))), 1e-300))
    i_v7188 = -1.0 == 1
    i_v7189 = jnp.where(i_v7188, 4.97232e-07, 3.42537e-07)
    i_v7195 = i_v7189 * i_v6010
    i_v6844 = 1.0 + 0.0 + 0.0 + 0.0
    i_v6862 = 1.0 + 0.0 + 0.0 + 0.0
    i_v7187 = jnp.where((i_v6862 == 0.0) | ~jnp.isfinite(i_v6862), 0.0, jnp.divide(jnp.where((i_v6862 == 0.0) | ~jnp.isfinite(i_v6862), 0.0, jnp.divide(jnp.where((4.23e-09 == 0.0) | ~jnp.isfinite(4.23e-09), 0.0, jnp.divide(jnp.where((4.23e-09 == 0.0) | ~jnp.isfinite(4.23e-09), 0.0, jnp.divide(jnp.exp(jnp.clip(i_v6844 * jnp.log(jnp.maximum(jnp.where((4.23e-09 * i_v6862 == 0.0) | ~jnp.isfinite(4.23e-09 * i_v6862), 0.0, jnp.divide(4.23e-09, jnp.where((4.23e-09 * i_v6862 == 0.0) | ~jnp.isfinite(4.23e-09 * i_v6862), 1.0, 4.23e-09 * i_v6862))), 1e-300)), -709.0, 709.0)), jnp.where((4.23e-09 == 0.0) | ~jnp.isfinite(4.23e-09), 1.0, 4.23e-09))), jnp.where((4.23e-09 == 0.0) | ~jnp.isfinite(4.23e-09), 1.0, 4.23e-09))), jnp.where((i_v6862 == 0.0) | ~jnp.isfinite(i_v6862), 1.0, i_v6862))), jnp.where((i_v6862 == 0.0) | ~jnp.isfinite(i_v6862), 1.0, i_v6862)))
    i_v7197 = i_v7195 * -1.3994e-08 * i_v7187
    i_v7199 = i_v7195 * -1.3994e-08 * i_v7187
    i_v7192 = jnp.where(i_v7188, 745669000000.0, 1166450000000.0)
    i_v7202 = -i_v7192 * 4.23e-09 * i_v6862
    i_v7178 = jnp.where((4.23e-09 == 0.0) | ~jnp.isfinite(4.23e-09), 0.0, jnp.divide(jnp.where((4.23e-09 == 0.0) | ~jnp.isfinite(4.23e-09), 0.0, jnp.divide(jnp.exp(jnp.clip(i_v6844 * jnp.log(jnp.maximum(jnp.where((4.23e-09 == 0.0) | ~jnp.isfinite(4.23e-09), 0.0, jnp.divide(4.23e-09, jnp.where((4.23e-09 == 0.0) | ~jnp.isfinite(4.23e-09), 1.0, 4.23e-09))), 1e-300)), -709.0, 709.0)), jnp.where((4.23e-09 == 0.0) | ~jnp.isfinite(4.23e-09), 1.0, 4.23e-09))), jnp.where((4.23e-09 == 0.0) | ~jnp.isfinite(4.23e-09), 1.0, 4.23e-09)))
    i_v7205 = i_v7189 * (i_v6010 * i_v6006 * i_v7178)
    i_v7207 = i_v7192 * -4.23e-09
    i_v7210 = 0.5 + jnp.divide(jnp.arctan(0.0 + 0.0 + 0.0 + 0.0), 3.141592653589793)
    i_v7215 = (0.677645 + -2.13215e-07 * i_v6034 + -1.35393e-06 * i_v6033 + 2.816e-13 * i_v6032) * (1.0 + (0.0 + 0.0 + 0.0 + 0.0) * i_v5873) + jnp.where((i_v6006 == 0.0) | ~jnp.isfinite(i_v6006), 0.0, jnp.divide(0.0, jnp.where((i_v6006 == 0.0) | ~jnp.isfinite(i_v6006), 1.0, i_v6006)))
    i_v6178 = -5.83361 + 1.42914e-06 * i_v6034 + 8.59939e-06 * i_v6033 + -1.78857e-12 * i_v6032
    i_v6160 = -3.0 + 0.0 + 0.0 + 0.0
    i_v7322 = jnp.where((4.23e-09 == 0.0) | ~jnp.isfinite(4.23e-09), 0.0, jnp.divide(i_v6178 * 4.23e-09, jnp.where((4.23e-09 == 0.0) | ~jnp.isfinite(4.23e-09), 1.0, 4.23e-09)))
    i_v7324 = jnp.sqrt(jnp.maximum(i_v5784 * i_v7134, 1e-300))
    i_v7326 = jnp.divide((1.0 + -1.2156e-08 * i_v6034 + 1.83259e-08 * i_v6033 + -3.81158e-15 * i_v6032) * i_v6006, i_v7324)
    i_v7327 = i_v7326 < 34.0
    i_v7329 = jnp.exp(jnp.clip(i_v7326, -709.0, 709.0))
    i_v7330 = i_v7329 - 1.0
    i_v7331 = i_v7330 * i_v7330
    i_v7337 = jnp.where(i_v7327, jnp.where((i_v7331 + 2.0 * i_v7329 * 1.713908431e-15 == 0.0) | ~jnp.isfinite(i_v7331 + 2.0 * i_v7329 * 1.713908431e-15), 0.0, jnp.divide(i_v7329, jnp.where((i_v7331 + 2.0 * i_v7329 * 1.713908431e-15 == 0.0) | ~jnp.isfinite(i_v7331 + 2.0 * i_v7329 * 1.713908431e-15), 1.0, i_v7331 + 2.0 * i_v7329 * 1.713908431e-15))), 1.7139084316226671e-15)
    i_v7341 = jnp.divide((1.0 + 0.0 + 0.0 + 0.0) * i_v6006, i_v7324)
    i_v7342 = i_v7341 < 34.0
    i_v7343 = jnp.exp(jnp.clip(i_v7341, -709.0, 709.0))
    i_v7344 = i_v7343 - 1.0
    i_v7345 = i_v7344 * i_v7344
    i_v7353 = (1.13779 + -1.99042e-07 * i_v6034 + 0.0 + 0.0) * jnp.where(i_v7342, jnp.where((i_v7345 + 2.0 * i_v7343 * 1.713908431e-15 == 0.0) | ~jnp.isfinite(i_v7345 + 2.0 * i_v7343 * 1.713908431e-15), 0.0, jnp.divide(i_v7343, jnp.where((i_v7345 + 2.0 * i_v7343 * 1.713908431e-15 == 0.0) | ~jnp.isfinite(i_v7345 + 2.0 * i_v7343 * 1.713908431e-15), 1.0, i_v7345 + 2.0 * i_v7343 * 1.713908431e-15))), 1.7139084316226671e-15) + (0.025366 + -4.4383e-09 * i_v6034 + 0.0 + 0.0)
    i_v7354 = i_v7149 - i_v7125
    i_v7386 = i_v6010 + (0.0 + 0.0 + 0.0 + 0.0)
    i_v7387 = jnp.where((i_v7386 == 0.0) | ~jnp.isfinite(i_v7386), 0.0, jnp.divide(4.23e-09 * i_v7125, jnp.where((i_v7386 == 0.0) | ~jnp.isfinite(i_v7386), 1.0, i_v7386)))
    i_v7390 = jnp.sqrt(jnp.maximum(1.0 + jnp.where((i_v6006 == 0.0) | ~jnp.isfinite(i_v6006), 0.0, jnp.divide(0.0 + 0.0 + 0.0 + 0.0, jnp.where((i_v6006 == 0.0) | ~jnp.isfinite(i_v6006), 1.0, i_v6006))), 1e-300))
    i_v7355 = i_v5785 * jnp.sqrt(jnp.maximum(i_v7134, 1e-300))
    i_v7372 = jnp.divide(i_v6256 * i_v6006, i_v7355)
    i_v7373 = i_v7372 < 34.0
    i_v7096 = 0.21164 + -6.1622e-09 * i_v6034 + -1.2252e-06 * i_v6033 + 2.54826e-13 * i_v6032 + jnp.divide((0.0 + 0.0 + 0.0 + 0.0) * i_v5873, 303.15)
    i_v6346 = 0.0145039 + -2.27838e-09 * i_v6034 + -1.07207e-08 * i_v6033 + 2.22977e-15 * i_v6032
    i_v7075 = jnp.where(i_v6346 > 1.0, jnp.divide(i_v6346, 10000.0), i_v6346) * (1.0 - (0.0 + 0.0 + 0.0 + 0.0) * jnp.exp(jnp.clip(jnp.where((1.0 + 0.0 + 0.0 + 0.0 == 0.0) | ~jnp.isfinite(1.0 + 0.0 + 0.0 + 0.0), 0.0, jnp.divide(-i_v6006, jnp.where((1.0 + 0.0 + 0.0 + 0.0 == 0.0) | ~jnp.isfinite(1.0 + 0.0 + 0.0 + 0.0), 1.0, 1.0 + 0.0 + 0.0 + 0.0))), -709.0, 709.0))) * jnp.power(jnp.maximum(i_v5781, 0.0), -0.819547 + 1.0806e-07 * i_v6034 + 0.0 + 0.0)
    i_v7655 = -1.05499 + 9.41548e-09 * i_v6034 + 3.44952e-07 * i_v6033 + -7.1746e-14 * i_v6032 + delvto
    i_v7659 = -1.0 * i_v7655
    i_v7658 = 0.0 + 0.0 + 0.0 + 0.0 + -1.0 * delvto
    i_v7668 = 4.0 * (i_v7659 - i_v7658 - i_v7125)
    i_v7671 = jnp.where(i_v7668 < 0.0, 0.0, i_v7668)
    i_v6196 = 2.57412 + -5.74679e-07 * i_v6034 + -3.47e-06 * i_v6033 + 7.21718e-13 * i_v6032
    i_v7677 = 0.9 * (i_v7125 - jnp.where((i_v6196 == 0.0) | ~jnp.isfinite(i_v6196), 0.0, jnp.divide(0.5 * i_v6178, jnp.where((i_v6196 == 0.0) | ~jnp.isfinite(i_v6196), 1.0, i_v6196))) * jnp.where((i_v6196 == 0.0) | ~jnp.isfinite(i_v6196), 0.0, jnp.divide(0.5 * i_v6178, jnp.where((i_v6196 == 0.0) | ~jnp.isfinite(i_v6196), 1.0, i_v6196))))
    i_v7686 = jnp.where(i_v7677 > i_v6160, i_v6160, i_v7677)
    i_v7688 = jnp.where((4.23e-09 == 0.0) | ~jnp.isfinite(4.23e-09), 0.0, jnp.divide(i_v6196 * 4.23e-09, jnp.where((4.23e-09 == 0.0) | ~jnp.isfinite(4.23e-09), 1.0, 4.23e-09)))
    i_v7358 = jnp.divide(i_v6274 * i_v6010 * i_v6006, i_v7355)
    i_v7359 = i_v7358 < 34.0
    i_v7360 = jnp.exp(jnp.clip(i_v7358, -709.0, 709.0))
    i_v7361 = i_v7360 - 1.0
    i_v7362 = i_v7361 * i_v7361
    i_v7374 = jnp.exp(jnp.clip(i_v7372, -709.0, 709.0))
    i_v7375 = i_v7374 - 1.0
    i_v7376 = i_v7375 * i_v7375
    i_v6616 = 0.0 + 0.0 + 0.0 + 0.0
    i_v7045 = jnp.where(i_v6970, i_v6616 + i_v6979, i_v6616 * i_v7021)
    i_v7058 = jnp.where(i_v7045 < 0.0, 0.0, i_v7045)
    i_v7689 = -(i_v6268 * jnp.where(i_v7359, jnp.where((i_v7362 + 2.0 * i_v7360 * 1.713908431e-15 == 0.0) | ~jnp.isfinite(i_v7362 + 2.0 * i_v7360 * 1.713908431e-15), 0.0, jnp.divide(i_v7360, jnp.where((i_v7362 + 2.0 * i_v7360 * 1.713908431e-15 == 0.0) | ~jnp.isfinite(i_v7362 + 2.0 * i_v7360 * 1.713908431e-15), 1.0, i_v7362 + 2.0 * i_v7360 * 1.713908431e-15))), 1.7139084316226671e-15) * i_v7354) - i_v6250 * jnp.where(i_v7373, jnp.where((i_v7376 + 2.0 * i_v7374 * 1.713908431e-15 == 0.0) | ~jnp.isfinite(i_v7376 + 2.0 * i_v7374 * 1.713908431e-15), 0.0, jnp.divide(i_v7374, jnp.where((i_v7376 + 2.0 * i_v7374 * 1.713908431e-15 == 0.0) | ~jnp.isfinite(i_v7376 + 2.0 * i_v7374 * 1.713908431e-15), 1.0, i_v7376 + 2.0 * i_v7374 * 1.713908431e-15))), 1.7139084316226671e-15) * i_v7354 + i_v6208 * i_v7387 + (i_v7322 * (i_v7390 - 1.0) * i_v7127 + jnp.where(jnp.where(0.0 == 2, True, 0.0 == 3), -i_v6184 * i_v6965, jnp.where(jnp.where(0.0 == 1, True, i_v6970), (i_v6184 + jnp.where((i_v6006 == 0.0) | ~jnp.isfinite(i_v6006), 0.0, jnp.divide(i_v6190, jnp.where((i_v6006 == 0.0) | ~jnp.isfinite(i_v6006), 1.0, i_v6006)))) * i_v6965, jnp.where(i_v7373, i_v7376, jnp.where(i_v7359, i_v7362, jnp.where(i_v7342, i_v7345, jnp.where(i_v7327, i_v7331, i_v7058))))))) - i_v7125 - i_v6178 * i_v7127 + i_v7659
    i_v7791 = 1e-12 + jnp.where((50.0 == 0.0) | ~jnp.isfinite(50.0), 0.0, jnp.divide(1.0, jnp.where((50.0 == 0.0) | ~jnp.isfinite(50.0), 1.0, 50.0)))
    i_v7775, i_v7780, i_v7788, i_v7784 = jax.tree_util.tree_map(lambda _t, _f: jnp.where(50.0 < 0.001, _t, _f), (1000.0, 1000.0, 1000.0, 1000.0), (1e-12 + jnp.where((50.0 == 0.0) | ~jnp.isfinite(50.0), 0.0, jnp.divide(1.0, jnp.where((50.0 == 0.0) | ~jnp.isfinite(50.0), 1.0, 50.0))), 1e-12 + jnp.where((50.0 == 0.0) | ~jnp.isfinite(50.0), 0.0, jnp.divide(1.0, jnp.where((50.0 == 0.0) | ~jnp.isfinite(50.0), 1.0, 50.0))), 1e-12 + jnp.where((50.0 == 0.0) | ~jnp.isfinite(50.0), 0.0, jnp.divide(1.0, jnp.where((50.0 == 0.0) | ~jnp.isfinite(50.0), 1.0, 50.0))), 1e-12 + jnp.where((50.0 == 0.0) | ~jnp.isfinite(50.0), 0.0, jnp.divide(1.0, jnp.where((50.0 == 0.0) | ~jnp.isfinite(50.0), 1.0, 50.0)))))
    i_v7834 = 1.0 * nf * (i_v5972 - 0.0)
    i_v7835 = jnp.where((i_v7834 == 0.0) | ~jnp.isfinite(i_v7834), 0.0, jnp.divide(0.1 * (0.0 + jnp.where((1.0 == 0.0) | ~jnp.isfinite(1.0), 0.0, jnp.divide(jnp.divide(i_v6022, 3.0), jnp.where((1.0 == 0.0) | ~jnp.isfinite(1.0), 1.0, 1.0)))), jnp.where((i_v7834 == 0.0) | ~jnp.isfinite(i_v7834), 1.0, i_v7834)))
    i_v7837 = jnp.where((i_v7835 == 0.0) | ~jnp.isfinite(i_v7835), 0.0, jnp.divide(1.0, jnp.where((i_v7835 == 0.0) | ~jnp.isfinite(i_v7835), 1.0, i_v7835)))
    i_v8192 = 0.0
    i_v8453 = jnp.where((i_v8192 == 0.0) | ~jnp.isfinite(i_v8192), 0.0, jnp.divide(1.0, jnp.where((i_v8192 == 0.0) | ~jnp.isfinite(i_v8192), 1.0, i_v8192)))
    i_v8465 = 0.0
    i_v8697 = jnp.where((i_v8465 == 0.0) | ~jnp.isfinite(i_v8465), 0.0, jnp.divide(1.0, jnp.where((i_v8465 == 0.0) | ~jnp.isfinite(i_v8465), 1.0, i_v8465)))
    i_v8115 = jnp.where(0.0 < 0.0, 0.0, 0.0)
    i_v7848 = ps - i_v6022 * nf
    i_v7940 = jnp.where(i_v7848 < 0.0, 0.0, i_v7848)
    i_v8191 = jnp.where(ad < 0.0, 0.0, ad)
    i_v7946 = pd - i_v6022 * nf
    i_v8033 = jnp.where(i_v7946 < 0.0, 0.0, i_v7946)
    i_v8917 = jnp.where((i_v5824 == 0.0) | ~jnp.isfinite(i_v5824), 0.0, jnp.divide(jnp.where(i_v5678, 1.1142828575310917, 1.1142828575310917), jnp.where((i_v5824 == 0.0) | ~jnp.isfinite(i_v5824), 1.0, i_v5824))) * i_v6965
    i_v8980 = 0.0
    i_v8982 = 0.0
    i_v8984 = 0.0
    i_v8986 = 0.0
    i_v8978 = i_v6022 * nf
    i_v8977 = jnp.sqrt(jnp.maximum(jnp.where((i_v6022 == 0.0) | ~jnp.isfinite(i_v6022), 0.0, jnp.divide(0.0, jnp.where((i_v6022 == 0.0) | ~jnp.isfinite(i_v6022), 1.0, i_v6022))), 1e-300)) + 1.0
    i_v8989 = 0.0
    i_v8992 = 0.0
    i_v5755 = jnp.where(jnp.where(i_v5678, True, 0 != 0), jnp.where((4.23e-09 == 0.0) | ~jnp.isfinite(4.23e-09), 0.0, jnp.divide(3.4531302e-11, jnp.where((4.23e-09 == 0.0) | ~jnp.isfinite(4.23e-09), 1.0, 4.23e-09))), 0.0)
    i_v6904 = 0.6 + 0.0 + 0.0 + 0.0
    i_v9271 = jnp.where(i_v6904 < 0.02, 0.02, i_v6904)
    i_v6910 = 0.6 + 0.0 + 0.0 + 0.0
    i_v9274 = jnp.where(i_v6910 < 0.02, 0.02, i_v6910)
    i_v9299 = 1.0 == 1
    i_v6916 = 1.2e-11 + 0.0 + 0.0 + 0.0
    i_v7111 = (5.24893e-11 + i_v6916) * i_v6018
    i_v9512 = jnp.where(i_v9299, jnp.where(i_v7111 < 0.0, 0.0, i_v7111), i_v7111)
    i_v7109 = (5.24893e-11 + i_v6916) * i_v6018
    i_v9513, i_v9514, i_v9515, i_v9516, i_v9517, i_v9518, i_v9519 = jax.tree_util.tree_map(lambda _t, _f: jnp.where(i_v9299, _t, _f), (jnp.where(i_v7109 < 0.0, 0.0, i_v7109), jnp.where(0.29781 >= 0.99, 0.99, 0.29781), jnp.where(0.29781 >= 0.99, 0.99, 0.29781), jnp.where(0.34629 >= 0.99, 0.99, 0.34629), jnp.where(0.29781 >= 0.99, 0.99, 0.29781), jnp.where(0.29781 >= 0.99, 0.99, 0.29781), jnp.where(0.34629 >= 0.99, 0.99, 0.34629)), (i_v7109, 0.29781, 0.29781, 0.34629, 0.29781, 0.29781, 0.34629))
    i_v6106 = -1.27189 + 4.0428e-07 * i_v6034 + 0.0 + 0.0
    i_v9520 = jnp.where(i_v9299, i_v6106, i_v6106)
    i_v6100 = 0.0 + 0.0 + 0.0 + 0.0
    i_v9521 = jnp.where(i_v9299, i_v6100, i_v6100)
    i_v7041 = jnp.where(i_v6970, jnp.where((i_v6969 == 0.0) | ~jnp.isfinite(i_v6969), 0.0, jnp.divide((0.0 + i_v6979) * nf, jnp.where((i_v6969 == 0.0) | ~jnp.isfinite(i_v6969), 1.0, i_v6969))), jnp.where((i_v6969 == 0.0) | ~jnp.isfinite(i_v6969), 0.0, jnp.divide(0.0, jnp.where((i_v6969 == 0.0) | ~jnp.isfinite(i_v6969), 1.0, i_v6969))))
    i_v9522 = jnp.where(i_v9299, jnp.where(i_v7041 < 0.0, 0.0, i_v7041), i_v7041)
    i_v6622 = 0.1376 + 0.0 + 0.0 + 0.0
    i_v9524 = jnp.where(i_v9299, jnp.where(i_v6622 < 0.0, 0.0, i_v6622), i_v6622)
    i_v9588 = 0.026123473186126484 * (20.0 * (1.0 + 0.0))
    i_v9589 = 0.026123473186126484 * (20.0 * (1.0 + 0.0))
    i_v9590 = 0.026123473186126484 * (20.0 * (1.0 + 0.0))
    i_v9591 = 0.026123473186126484 * (20.0 * (1.0 + 0.0))
    i_v9592 = 0.026123473186126484 * (20.0 * (1.0 + 0.0))
    i_v9593 = 0.026123473186126484 * (20.0 * (1.0 + 0.0))
    i_v9607 = 0.95 * i_v7125
    i_v9619 = jnp.sqrt(jnp.maximum(1.0 + jnp.where((i_v6006 == 0.0) | ~jnp.isfinite(i_v6006), 0.0, jnp.divide(0.0 + 0.0 + 0.0 + 0.0, jnp.where((i_v6006 == 0.0) | ~jnp.isfinite(i_v6006), 1.0, i_v6006))), 1e-300))
    i_v9629 = i_v7658 + i_v7125
    i_v9631 = jnp.where(i_v5678, 1.03594e-10, 1.03593906e-10)
    i_v6604 = 547.88 + 0.0 + 0.0 + 0.0
    i_v7042 = jnp.where(i_v6970, jnp.where((i_v6969 == 0.0) | ~jnp.isfinite(i_v6969), 0.0, jnp.divide((i_v6604 + i_v6979) * nf, jnp.where((i_v6969 == 0.0) | ~jnp.isfinite(i_v6969), 1.0, i_v6969))), jnp.where((i_v6969 == 0.0) | ~jnp.isfinite(i_v6969), 0.0, jnp.divide(i_v6604 * i_v7021 * nf, jnp.where((i_v6969 == 0.0) | ~jnp.isfinite(i_v6969), 1.0, i_v6969))))
    i_v9366 = jnp.where(i_v6604 < 0.0, 0.0, i_v7042)
    i_v9640 = jnp.where(i_v9299, jnp.where(i_v9366 < 0.0, 0.0, i_v9366), i_v7042) * 0.5
    i_v9641 = 0.5 * i_v7322
    i_v9644 = i_v6010 + (0.0 + 0.0 + 0.0 + 0.0)
    i_v9645 = jnp.where((i_v9644 == 0.0) | ~jnp.isfinite(i_v9644), 0.0, jnp.divide(0.0 + 0.0 + 0.0 + 0.0, jnp.where((i_v9644 == 0.0) | ~jnp.isfinite(i_v9644), 1.0, i_v9644)))
    i_v9656 = jnp.where(0 == 0, 2.0 * -1.0 * -0.10714142876554583, 0.0)
    i_v9698 = 1.0 - i_v9520
    i_v9705 = 2.0 * i_v6598
    i_v9709 = 200000000.0 * 4.23e-09
    i_v9724 = 1.0 + 0.0
    i_v9732 = 1e-10 + 0.0 + 0.0 + 0.0 + (1e-10 + 0.0 + 0.0 + 0.0) * i_v6006
    i_v9736 = jnp.where((i_v6006 == 0.0) | ~jnp.isfinite(i_v6006), 0.0, jnp.divide(i_v9732, jnp.where((i_v6006 == 0.0) | ~jnp.isfinite(i_v6006), 1.0, i_v6006)))
    i_v9749 = (1.0 + 0.0 + 0.0 + 0.0) * i_v5824
    i_v9754 = jnp.where((i_v6969 == 0.0) | ~jnp.isfinite(i_v6969), 0.0, jnp.divide(i_v7058, jnp.where((i_v6969 == 0.0) | ~jnp.isfinite(i_v6969), 1.0, i_v6969))) * 0.5
    i_v6610 = 0.0 + 0.0 + 0.0 + 0.0
    i_v7047 = jnp.where(i_v6970, i_v6610 + i_v6979, i_v6610 * i_v7021)
    i_v9762 = jnp.where((i_v6969 == 0.0) | ~jnp.isfinite(i_v6969), 0.0, jnp.divide(jnp.where(i_v7047 < 0.0, 0.0, i_v7047), jnp.where((i_v6969 == 0.0) | ~jnp.isfinite(i_v6969), 1.0, i_v6969))) * 0.5
    i_v9766 = jnp.where(i_v5678, 3.0 * 4.23e-09, jnp.where((3.9 == 0.0) | ~jnp.isfinite(3.9), 0.0, jnp.divide(11.7 * 4.23e-09, jnp.where((3.9 == 0.0) | ~jnp.isfinite(3.9), 1.0, 3.9))))
    i_v9805 = jnp.where(jnp.where(0.0 != 0, True, 0.0 != 0), i_v7689, 0.0)
    i_v9809 = jnp.where(0.0 < 2, i_v5824, 0.026123473186126484) * (1.0 + 0.0 + 0.0 + 0.0)
    i_v6778 = 0.075 + 0.0 + 0.0 + 0.0
    i_v6772 = 0.054 + 0.0 + 0.0 + 0.0
    i_v9815 = i_v6766 * i_v6778 - i_v6772
    i_v9816 = i_v6772 * i_v6778
    i_v9817 = -i_v7207
    i_v6400 = 0.075 + 0.0 + 0.0 + 0.0
    i_v6394 = 0.054 + 0.0 + 0.0 + 0.0
    i_v9820 = i_v6388 * i_v6400 - i_v6394
    i_v9821 = i_v6394 * i_v6400
    i_v6418 = 0.075 + 0.0 + 0.0 + 0.0
    i_v6412 = 0.054 + 0.0 + 0.0 + 0.0
    i_v9823 = i_v6406 * i_v6418 - i_v6412
    i_v9824 = i_v6412 * i_v6418
    i_v9830 = 4.97232e-07 * i_v6010 * i_v6006 * i_v7178
    i_v9831 = -745669000000.0 * 4.23e-09
    i_v6802 = 0.075 + 0.0 + 0.0 + 0.0
    i_v6796 = 0.054 + 0.0 + 0.0 + 0.0
    i_v9834 = i_v6790 * i_v6802 - i_v6796
    i_v9835 = i_v6796 * i_v6802
    i_v9839 = i_v9830 * 0.7561
    i_v9841 = i_v9831 * 1.31724
    i_v6820 = 0.006 + 0.0 + 0.0 + 0.0
    i_v6814 = 0.03 + 0.0 + 0.0 + 0.0
    i_v9844 = i_v6808 * i_v6820 - i_v6814
    i_v9845 = i_v6814 * i_v6820
    i_v9853 = i_v5750 * i_v6018 * i_v6014 * nf
    i_v5876 = 1.0 + 0.0012407 * i_v5873
    i_v9901 = 0.000738019 * i_v5876 * i_v8191
    i_v9902 = 0.000738019 * i_v5876 * i_v8115
    i_v5887 = 1.0 + 0.00037357 * i_v5873
    i_v9903 = 9.88889e-11 * i_v5887 * i_v8033
    i_v5898 = 1.0 + 2e-12 * i_v5873
    i_v9905 = 9.88889e-11 * i_v5898 * i_v6022 * nf
    i_v9906 = 9.88889e-11 * i_v5887 * i_v7940
    i_v9908 = 9.88889e-11 * i_v5898 * i_v6022 * nf
    i_v9918 = i_v6018 * (9.54827e-12 + 0.0 + 0.0 + 0.0)
    i_v9922 = i_v6018 * (9.54827e-12 + 0.0 + 0.0 + 0.0)
    i_v10077 = _mfactor * i_v7837
    i_v10079 = _mfactor * -1.0
    i_v10081 = _mfactor * -i_v7837
    return jnp.array([0, 1.03594e-10, i_v5750, i_v5785, i_v5824, i_v5911, i_v5915, i_v5920, i_v5924, i_v5929, i_v5933, i_v6006, i_v6010, i_v6014, i_v6018, i_v6022, i_v6040, i_v6046, i_v6052, i_v6058, i_v6070, i_v6088, i_v6094, i_v6112, i_v6142, i_v6184, i_v6190, i_v6202, i_v6208, i_v6214, i_v6250, i_v6256, i_v6262, i_v6268, i_v6274, i_v6280, i_v6388, i_v6406, i_v6424, i_v6430, i_v6436, i_v6442, i_v6580, i_v6586, i_v6592, i_v6598, i_v6628, i_v6646, i_v6652, i_v6670, i_v6676, i_v6682, i_v6688, i_v6700, i_v6706, i_v6736, i_v6742, i_v6748, i_v6754, i_v6760, i_v6766, i_v6790, i_v6808, i_v6850, i_v6856, i_v6868, i_v6880, i_v6934, i_v6965, i_v7037, i_v7038, i_v7039, i_v7040, i_v7043, i_v7055, i_v7063, i_v7084, i_v7090, i_v7113, i_v7125, i_v7127, i_v7134, i_v7144, i_v7149, i_v7153, i_v7172, i_v7197, i_v7199, i_v7202, i_v7205, i_v7207, i_v7210, i_v7215, i_v6178, i_v7322, i_v7337, i_v7353, i_v7354, i_v7387, i_v7390, i_v7096, i_v7075, i_v7655, i_v7671, i_v7686, i_v7688, i_v7689, i_v7791, i_v7775, i_v7780, i_v7788, i_v7784, i_v7837, i_v8453, i_v8697, i_v8980, i_v8982, i_v8984, i_v8986, i_v8989, i_v8992, i_v5755, i_v9271, i_v9274, i_v9512, i_v9513, i_v9514, i_v9515, i_v9516, i_v9517, i_v9518, i_v9519, i_v9520, i_v9521, i_v9522, i_v9524, i_v9588, i_v9589, i_v9590, i_v9591, i_v9592, i_v9593, i_v9607, i_v9619, i_v9629, i_v9631, i_v9640, i_v9641, i_v9645, i_v9656, i_v9698, i_v9705, i_v9709, i_v9724, i_v9732, i_v9736, i_v9749, i_v9754, i_v9762, i_v9766, i_v9805, i_v9809, i_v9815, i_v9816, i_v9817, i_v9820, i_v9821, i_v9823, i_v9824, i_v9830, i_v9831, i_v9834, i_v9835, i_v9839, i_v9841, i_v9844, i_v9845, i_v9853, i_v9901, i_v9902, i_v9903, i_v9905, i_v9906, i_v9908, i_v9918, i_v9922, i_v10077, i_v10079, i_v10081])

def _SKY130_PFET_01V8_TT_combined(signals: Signals, s: States, init, l: float=1.5e-07, w: float=2e-06, nf: float=1.0, _min: float=0.0, ad: float=0.0, ps: float=0.0, pd: float=0.0, sa: float=0.0, sb: float=0.0, sd: float=0.0, delvto: float=0.0, _ckt_gmin: float=1e-12, off: float=0.0, _temperature: float=300.15, _mfactor: float=1.0) -> tuple:
    """Combined physics + Jacobian — single hoist block, auto-generated from VA MIR."""
    i_v5750 = init[2]
    i_v5785 = init[3]
    i_v5824 = init[4]
    i_v6006 = init[11]
    i_v6010 = init[12]
    i_v6022 = init[15]
    i_v6046 = init[17]
    i_v6052 = init[18]
    i_v6070 = init[20]
    i_v6088 = init[21]
    i_v6112 = init[23]
    i_v6142 = init[24]
    i_v6202 = init[27]
    i_v6214 = init[29]
    i_v6250 = init[30]
    i_v6262 = init[32]
    i_v6268 = init[33]
    i_v6280 = init[35]
    i_v6424 = init[38]
    i_v6430 = init[39]
    i_v6580 = init[42]
    i_v6586 = init[43]
    i_v6592 = init[44]
    i_v6598 = init[45]
    i_v6628 = init[46]
    i_v6646 = init[47]
    i_v6652 = init[48]
    i_v6670 = init[49]
    i_v6682 = init[51]
    i_v6688 = init[52]
    i_v6700 = init[53]
    i_v6706 = init[54]
    i_v6736 = init[55]
    i_v6742 = init[56]
    i_v6748 = init[57]
    i_v6868 = init[65]
    i_v6880 = init[66]
    i_v6934 = init[67]
    i_v6965 = init[68]
    i_v7037 = init[69]
    i_v7038 = init[70]
    i_v7039 = init[71]
    i_v7043 = init[73]
    i_v7090 = init[77]
    i_v7113 = init[78]
    i_v7125 = init[79]
    i_v7127 = init[80]
    i_v7134 = init[81]
    i_v7144 = init[82]
    i_v7153 = init[84]
    i_v7172 = init[85]
    i_v7197 = init[86]
    i_v7199 = init[87]
    i_v7202 = init[88]
    i_v7205 = init[89]
    i_v7207 = init[90]
    i_v7210 = init[91]
    i_v7322 = init[94]
    i_v7337 = init[95]
    i_v7353 = init[96]
    i_v7354 = init[97]
    i_v7387 = init[98]
    i_v7075 = init[101]
    i_v7686 = init[104]
    i_v7688 = init[105]
    i_v7689 = init[106]
    i_v7837 = init[112]
    i_v8453 = init[113]
    i_v8697 = init[114]
    i_v8980 = init[115]
    i_v8982 = init[116]
    i_v8984 = init[117]
    i_v8986 = init[118]
    i_v8989 = init[119]
    i_v8992 = init[120]
    i_v5755 = init[121]
    i_v9271 = init[122]
    i_v9274 = init[123]
    i_v9512 = init[124]
    i_v9513 = init[125]
    i_v9520 = init[132]
    i_v9521 = init[133]
    i_v9524 = init[135]
    i_v9588 = init[136]
    i_v9589 = init[137]
    i_v9590 = init[138]
    i_v9591 = init[139]
    i_v9592 = init[140]
    i_v9593 = init[141]
    i_v9607 = init[142]
    i_v9619 = init[143]
    i_v9629 = init[144]
    i_v9631 = init[145]
    i_v9640 = init[146]
    i_v9698 = init[150]
    i_v9705 = init[151]
    i_v9709 = init[152]
    i_v9724 = init[153]
    i_v9736 = init[155]
    i_v9749 = init[156]
    i_v9754 = init[157]
    i_v9762 = init[158]
    i_v9766 = init[159]
    i_v9815 = init[162]
    i_v9817 = init[164]
    i_v9820 = init[165]
    i_v9821 = init[166]
    i_v9823 = init[167]
    i_v9824 = init[168]
    i_v9830 = init[169]
    i_v9831 = init[170]
    i_v9834 = init[171]
    i_v9839 = init[173]
    i_v9841 = init[174]
    i_v9844 = init[175]
    i_v9853 = init[177]
    i_v9901 = init[178]
    i_v9902 = init[179]
    i_v9903 = init[180]
    i_v9905 = init[181]
    i_v9906 = init[182]
    i_v9908 = init[183]
    i_v9918 = init[184]
    i_v9922 = init[185]
    i_v10079 = init[187]
    i_v10081 = init[188]
    v8471935 = -1.0 * (signals.g - signals.s)
    v8471931 = -1.0 * (signals.d - signals.s)
    v8471984 = v8471935 - v8471931
    v8904479 = v8471984 - i_v7153
    v8904482 = jnp.sqrt(jnp.maximum(v8904479 * v8904479 + 0.0001, 1e-300))
    v8904488 = 1.0 + i_v9524 * (0.5 * (v8904479 + v8904482))
    v8897760 = -i_v6628
    v8471939 = -1.0 * (s.v_bi - signals.s)
    v8471980 = v8471939 - v8471931
    v8904830 = jnp.where((v8904488 == 0.0) | ~jnp.isfinite(v8904488), 0.0, jnp.divide(1.0, jnp.where((v8904488 == 0.0) | ~jnp.isfinite(v8904488), 1.0, v8904488))) + v8897760 * v8471980
    v8904833 = jnp.sqrt(jnp.maximum(v8904830 * v8904830 + 0.01, 1e-300))
    v9418325 = 1.0 + (init[74] + (v8904830 + v8904833) * i_v9762) * i_v8697
    v9418326 = jnp.where((v9418325 == 0.0) | ~jnp.isfinite(v9418325), 0.0, jnp.divide(i_v8697, jnp.where((v9418325 == 0.0) | ~jnp.isfinite(v9418325), 1.0, v9418325)))
    v9418327 = 0.0
    v9437770 = 0.0
    v8897332 = v8471935 - i_v7153
    v8897335 = jnp.sqrt(jnp.maximum(v8897332 * v8897332 + 0.0001, 1e-300))
    v8897549 = 1.0 + i_v9524 * (0.5 * (v8897332 + v8897335))
    v8898095 = jnp.where((v8897549 == 0.0) | ~jnp.isfinite(v8897549), 0.0, jnp.divide(1.0, jnp.where((v8897549 == 0.0) | ~jnp.isfinite(v8897549), 1.0, v8897549))) + v8897760 * v8471939
    v8898098 = jnp.sqrt(jnp.maximum(v8898095 * v8898095 + 0.01, 1e-300))
    v9407273 = 1.0 + (init[75] + (v8898095 + v8898098) * i_v9754) * i_v8453
    v9407274 = jnp.where((v9407273 == 0.0) | ~jnp.isfinite(v9407273), 0.0, jnp.divide(i_v8453, jnp.where((v9407273 == 0.0) | ~jnp.isfinite(v9407273), 1.0, v9407273)))
    v9407275 = 0.0
    v8650221 = v8471931 >= 0.0
    v8725815 = jnp.where(v8650221, 1, -1)
    v9336400 = v8725815 < 0
    v9123179 = nf != 1.0
    v8725855 = v8725815 > 0
    v8725375 = i_v6142 > 1e+18
    v8725473 = jnp.where(jnp.where(jnp.where(v8725375, i_v6142 < 1e+25, False), v8471935 > i_v9629, False), i_v9631 != 0.0, False)
    v8725550 = 2.0 * (v8471935 - i_v9629)
    v8725536 = jnp.divide(1.6597586840442798e-23 * i_v6142, 6.664156273243601e-05)
    v8725553 = jnp.sqrt(jnp.maximum(1.0 + jnp.where((v8725536 == 0.0) | ~jnp.isfinite(v8725536), 0.0, jnp.divide(v8725550, jnp.where((v8725536 == 0.0) | ~jnp.isfinite(v8725536), 1.0, v8725536))), 1e-300))
    v8725555 = v8725553 + 1.0
    v8725556 = jnp.where((v8725555 == 0.0) | ~jnp.isfinite(v8725555), 0.0, jnp.divide(v8725550, jnp.where((v8725555 == 0.0) | ~jnp.isfinite(v8725555), 1.0, v8725555)))
    v8725557 = 0.5 * v8725556
    v8725561 = 1.12 - jnp.where((v8725536 == 0.0) | ~jnp.isfinite(v8725536), 0.0, jnp.divide(v8725557 * v8725556, jnp.where((v8725536 == 0.0) | ~jnp.isfinite(v8725536), 1.0, v8725536))) - 0.05
    v8725564 = jnp.sqrt(jnp.maximum(v8725561 * v8725561 + 0.224, 1e-300))
    v8725865 = jnp.where(v8725473, v8471935 - (1.12 - 0.5 * (v8725561 + v8725564)), v8471935)
    v8725690 = jnp.where(jnp.where(jnp.where(v8725375, i_v6142 < 1e+25, False), v8471984 > i_v9629, False), i_v9631 != 0.0, False)
    v8725733 = 2.0 * (v8471984 - i_v9629)
    v8725719 = jnp.divide(1.6597586840442798e-23 * i_v6142, 6.664156273243601e-05)
    v8725736 = jnp.sqrt(jnp.maximum(1.0 + jnp.where((v8725719 == 0.0) | ~jnp.isfinite(v8725719), 0.0, jnp.divide(v8725733, jnp.where((v8725719 == 0.0) | ~jnp.isfinite(v8725719), 1.0, v8725719))), 1e-300))
    v8725738 = v8725736 + 1.0
    v8725739 = jnp.where((v8725738 == 0.0) | ~jnp.isfinite(v8725738), 0.0, jnp.divide(v8725733, jnp.where((v8725738 == 0.0) | ~jnp.isfinite(v8725738), 1.0, v8725738)))
    v8725740 = 0.5 * v8725739
    v8725744 = 1.12 - jnp.where((v8725719 == 0.0) | ~jnp.isfinite(v8725719), 0.0, jnp.divide(v8725740 * v8725739, jnp.where((v8725719 == 0.0) | ~jnp.isfinite(v8725719), 1.0, v8725719))) - 0.05
    v8725747 = jnp.sqrt(jnp.maximum(v8725744 * v8725744 + 0.224, 1e-300))
    v8725871 = jnp.where(v8725690, v8471984 - (1.12 - 0.5 * (v8725744 + v8725747)), v8471984)
    v8725872 = jnp.where(v8725855, v8725865, v8725871)
    v8650925 = jnp.where(v8650221, v8471939, v8471980)
    v8653420 = v8650925 - i_v7686 - 0.001
    v8653426 = v8653420 >= 0.0
    v8653425 = jnp.sqrt(jnp.maximum(v8653420 * v8653420 - 0.004 * i_v7686, 1e-300))
    v8653438 = v8653425 - v8653420
    v8654111 = i_v9607 - jnp.where(v8653426, i_v7686 + 0.5 * (v8653420 + v8653425), i_v7686 * (1.0 + jnp.where((v8653438 == 0.0) | ~jnp.isfinite(v8653438), 0.0, jnp.divide(-0.002, jnp.where((v8653438 == 0.0) | ~jnp.isfinite(v8653438), 1.0, v8653438))))) - 0.001
    v8654115 = jnp.sqrt(jnp.maximum(v8654111 * v8654111 + 0.004 * i_v9607, 1e-300))
    v8654118 = i_v9607 - 0.5 * (v8654111 + v8654115)
    v8654119 = i_v7125 - v8654118
    v8654121 = jnp.sqrt(jnp.maximum(v8654119, 1e-300))
    v8689484 = i_v7322 * v8654121
    v7319294 = init[31] * i_v6006
    v8657438 = jnp.divide(i_v7134 * v8654121, i_v7127)
    v8658363 = jnp.sqrt(jnp.maximum(v8657438, 1e-300))
    v8664496 = i_v5785 * v8658363
    v8663759 = i_v6262 * v8654118
    v8663761 = v8663759 >= -0.5
    v8663770 = 1.0 + 3.0 * v8663759
    v8663767 = 3.0 + 8.0 * v8663759
    v8663768 = jnp.where((v8663767 == 0.0) | ~jnp.isfinite(v8663767), 0.0, jnp.divide(1.0, jnp.where((v8663767 == 0.0) | ~jnp.isfinite(v8663767), 1.0, v8663767)))
    v8664497 = jnp.where(v8663761, 1.0 + v8663759, v8663770 * v8663768)
    v8664498 = v8664496 * v8664497
    v8667922 = jnp.where((v8664498 == 0.0) | ~jnp.isfinite(v8664498), 0.0, jnp.divide(v7319294, jnp.where((v8664498 == 0.0) | ~jnp.isfinite(v8664498), 1.0, v8664498)))
    v8667923 = v8667922 < 34.0
    v8667925 = jnp.exp(jnp.clip(v8667922, -709.0, 709.0))
    v8667926 = v8667925 - 1.0
    v8667929 = 2.0 * v8667925 * 1.713908431e-15
    v8667930 = v8667926 * v8667926 + v8667929
    v8667931 = jnp.where((v8667930 == 0.0) | ~jnp.isfinite(v8667930), 0.0, jnp.divide(v8667925, jnp.where((v8667930 == 0.0) | ~jnp.isfinite(v8667930), 1.0, v8667930)))
    v8668387 = jnp.where(v8667923, v8667931, 1.7139084316226671e-15)
    v7318983 = init[34] * i_v6010 * i_v6006
    v8667269 = i_v6280 * v8654118
    v8667271 = v8667269 >= -0.5
    v8667280 = 1.0 + 3.0 * v8667269
    v8667277 = 3.0 + 8.0 * v8667269
    v8667278 = jnp.where((v8667277 == 0.0) | ~jnp.isfinite(v8667277), 0.0, jnp.divide(1.0, jnp.where((v8667277 == 0.0) | ~jnp.isfinite(v8667277), 1.0, v8667277)))
    v8667292 = jnp.where(v8667271, 1.0 + v8667269, v8667280 * v8667278)
    v8667293 = v8664496 * v8667292
    v8669491 = jnp.where((v8667293 == 0.0) | ~jnp.isfinite(v8667293), 0.0, jnp.divide(v7318983, jnp.where((v8667293 == 0.0) | ~jnp.isfinite(v8667293), 1.0, v8667293)))
    v8669492 = v8669491 < 34.0
    v8669494 = jnp.exp(jnp.clip(v8669491, -709.0, 709.0))
    v8669495 = v8669494 - 1.0
    v8669498 = 2.0 * v8669494 * 1.713908431e-15
    v8669499 = v8669495 * v8669495 + v8669498
    v8669500 = jnp.where((v8669499 == 0.0) | ~jnp.isfinite(v8669499), 0.0, jnp.divide(v8669494, jnp.where((v8669499 == 0.0) | ~jnp.isfinite(v8669499), 1.0, v8669499)))
    v8683383 = init[100] + i_v6646 * v8654118
    v8683384 = v8683383 < 0.0001
    v8683391 = 0.0002 - v8683383
    v8683388 = 3.0 - 20000.0 * v8683383
    v8683389 = jnp.where((v8683388 == 0.0) | ~jnp.isfinite(v8683388), 0.0, jnp.divide(1.0, jnp.where((v8683388 == 0.0) | ~jnp.isfinite(v8683388), 1.0, v8683388)))
    v8686040 = jnp.where(v8683384, v8683391 * v8683389, v8683383) * i_v7337
    v8653449 = jnp.where(v8650221, v8471931, -v8471931)
    v8696278 = -1.0 * init[102] + (v8689484 - init[93] * i_v7127) * i_v9619 - i_v7688 * v8654118 - i_v6250 * v8668387 * i_v7354 - i_v6268 * jnp.where(v8669492, v8669500, 1.7139084316226671e-15) * i_v7354 + (init[28] + i_v6214 * v8654118) * i_v7387 + (i_v7322 * (init[99] - 1.0) * i_v7127 + (init[25] + jnp.where((i_v6006 == 0.0) | ~jnp.isfinite(i_v6006), 0.0, jnp.divide(init[26], jnp.where((i_v6006 == 0.0) | ~jnp.isfinite(i_v6006), 1.0, i_v6006))) + i_v6202 * v8654118) * i_v6965) - v8686040 * v8653449
    v8701956 = init[16] + i_v6046 * v8654118 + i_v6052 * v8653449
    v8701972 = jnp.divide(i_v7090 * jnp.divide(1.03594e-10, v8657438) + v8701956 * v8668387 + init[19], i_v5750)
    v8701975 = v8701972 >= -0.5
    v8701983 = 1.0 + 3.0 * v8701972
    v8701980 = 3.0 + 8.0 * v8701972
    v8701981 = jnp.where((v8701980 == 0.0) | ~jnp.isfinite(v8701980), 0.0, jnp.divide(1.0, jnp.where((v8701980 == 0.0) | ~jnp.isfinite(v8701980), 1.0, v8701980)))
    v8701987 = jnp.where(v8701975, 1.0 + v8701972, v8701983 * v8701981)
    v8725908 = v8725872 - v8696278
    v8728744 = i_v7210 * v8725908
    v8726063 = v8701987 * i_v5824
    v8731462 = 1.0 - i_v7210
    v8731470 = init[92] - v8731462 * v8725908
    v8731477 = jnp.where((v8726063 == 0.0) | ~jnp.isfinite(v8726063), 0.0, jnp.divide(v8731470, jnp.where((v8726063 == 0.0) | ~jnp.isfinite(v8726063), 1.0, v8726063)))
    v8731479 = v8731477 < -34.0
    v8734190 = v8731477 > 34.0
    v8734181 = jnp.divide(1.3991368707141175e-17, i_v7172)
    v8734196 = jnp.divide(4763048140830.671, i_v7172)
    v8734207 = jnp.divide(i_v5750, i_v7172)
    v8734204 = jnp.exp(jnp.clip(v8731477, -709.0, 709.0))
    v8734208 = v8734207 * v8734204
    v8734227 = jnp.where(v8731479, i_v7210 + v8734181 * v8701987, jnp.where(v8734190, i_v7210 + v8734196 * v8701987, i_v7210 + v8701987 * v8734208))
    v8734229 = jnp.where((v8734227 == 0.0) | ~jnp.isfinite(v8734227), 0.0, jnp.divide(v8728744, jnp.where((v8734227 == 0.0) | ~jnp.isfinite(v8734227), 1.0, v8734227)))
    v8775822 = jnp.where((4.23e-09 == 0.0) | ~jnp.isfinite(4.23e-09), 0.0, jnp.divide(v8734229 + v8696278 + v8696278 - init[149], jnp.where((4.23e-09 == 0.0) | ~jnp.isfinite(4.23e-09), 1.0, 4.23e-09)))
    v8781625 = init[72] + i_v7039 * v8654118 + i_v7037 * v8775822
    v10044046 = 2.0 * jnp.sqrt(jnp.maximum(v8696278 * v8696278 + 0.0001, 1e-300))
    v8775829 = v8734229 + v10044046
    v8775831 = jnp.where((v8775829 == 0.0) | ~jnp.isfinite(v8775829), 0.0, jnp.divide(1.0, jnp.where((v8775829 == 0.0) | ~jnp.isfinite(v8775829), 1.0, v8775829))) * 4.23e-09
    v8778727 = i_v7038 * v8775831
    v8778728 = v8778727 * v8775831
    v8778729 = v8778728 * v8696278
    v8781627 = v8775822 * v8781625 + v8778729 * v8696278
    v8799108 = v8781627 >= -0.8
    v8799117 = 0.6 + v8781627
    v8799115 = 7.0 + 10.0 * v8781627
    v8799116 = jnp.where((v8799115 == 0.0) | ~jnp.isfinite(v8799115), 0.0, jnp.divide(1.0, jnp.where((v8799115 == 0.0) | ~jnp.isfinite(v8799115), 1.0, v8799115)))
    v8809059 = jnp.where(v8799108, 1.0 + v8781627, v8799117 * v8799116)
    v8809061 = jnp.where((v8809059 == 0.0) | ~jnp.isfinite(v8809059), 0.0, jnp.divide(i_v7075, jnp.where((v8809059 == 0.0) | ~jnp.isfinite(v8809059), 1.0, v8809059)))
    v8828641 = jnp.where((i_v9709 == 0.0) | ~jnp.isfinite(i_v9709), 0.0, jnp.divide(v8734229 + init[103], jnp.where((i_v9709 == 0.0) | ~jnp.isfinite(i_v9709), 1.0, i_v9709)))
    v8829538 = jnp.exp(jnp.clip(0.7 * jnp.log(jnp.maximum(v8828641, 1e-300)), -709.0, 709.0))
    v8829539 = 1.0 + v8829538
    v8840232 = init[1] + i_v5755 * jnp.where((v8829539 == 0.0) | ~jnp.isfinite(v8829539), 0.0, jnp.divide(1.9e-09, jnp.where((v8829539 == 0.0) | ~jnp.isfinite(v8829539), 1.0, v8829539)))
    v8840233 = jnp.where((v8840232 == 0.0) | ~jnp.isfinite(v8840232), 0.0, jnp.divide(8.456821984368795e-13, jnp.where((v8840232 == 0.0) | ~jnp.isfinite(v8840232), 1.0, v8840232)))
    v8734401 = v8654121 - i_v7127
    v8740227 = i_v6010 - 2.0 * (i_v6700 * v8734229 + i_v6706 * v8734401)
    v8740230 = v8740227 < 2e-08
    v8740238 = 2e-08 * (4e-08 - v8740227)
    v8740234 = 6e-08 - 2.0 * v8740227
    v8740235 = jnp.where((v8740234 == 0.0) | ~jnp.isfinite(v8740234), 0.0, jnp.divide(1.0, jnp.where((v8740234 == 0.0) | ~jnp.isfinite(v8740234), 1.0, v8740234)))
    v8740245 = jnp.where(v8740230, v8740238 * v8740235, v8740227)
    v8840286 = jnp.where((i_v6006 == 0.0) | ~jnp.isfinite(i_v6006), 0.0, jnp.divide(v8840233 * v8740245, jnp.where((i_v6006 == 0.0) | ~jnp.isfinite(i_v6006), 1.0, i_v6006)))
    v8840293 = v8809061 * v8840286
    v8814810 = v8653449 == 0.0
    v8741568 = 1.0 + i_v9524 * v8734229
    v8744451 = jnp.where((v8741568 == 0.0) | ~jnp.isfinite(v8741568), 0.0, jnp.divide(1.0, jnp.where((v8741568 == 0.0) | ~jnp.isfinite(v8741568), 1.0, v8741568))) + i_v6628 * v8734401
    v8744454 = jnp.sqrt(jnp.maximum(v8744451 * v8744451 + 0.01, 1e-300))
    v8745483 = init[134] + (v8744451 + v8744454) * i_v9640
    v8813590 = i_v9698 - i_v9521 * v8734229 - 0.0001
    v8813727 = jnp.where(v8745483 == 0.0, i_v9520 == 1.0, False)
    v8809993 = 2.0 * i_v7043
    v8809996 = jnp.where((v8809061 == 0.0) | ~jnp.isfinite(v8809061), 0.0, jnp.divide(v8809993, jnp.where((v8809061 == 0.0) | ~jnp.isfinite(v8809061), 1.0, v8809061)))
    v8810043 = v8809996 * i_v6006
    v8813708 = v8734229 + 2.0 * i_v5824
    v8813803 = v8810043 * v8813708
    v8745944 = init[147] * i_v9619
    v8746258 = jnp.divide(v8745944, v8654121) + i_v7688 - i_v6214 * i_v7387
    v10043489 = 2.0 * jnp.sqrt(jnp.maximum(i_v6070 * v8657438, 1e-300))
    v8747172 = i_v6006 + v10043489
    v8747173 = jnp.where((v8747172 == 0.0) | ~jnp.isfinite(v8747172), 0.0, jnp.divide(i_v6006, jnp.where((v8747172 == 0.0) | ~jnp.isfinite(v8747172), 1.0, v8747172)))
    v8753495 = i_v6088 * v8747173 + init[148]
    v8756384 = -v8746258
    v8756382 = init[22] * i_v6088
    v8753496 = v8747173 * v8747173
    v8756383 = v8756382 * (v8753496 * v8747173)
    v8756385 = v8756384 * v8756383
    v8756396 = 1.0 + v8746258 * v8753495 + v8756385 * v8734229
    v8756409 = v8756396 < 0.1
    v8756414 = 0.2 - v8756396
    v8756412 = 3.0 - 20.0 * v8756396
    v8756413 = jnp.where((v8756412 == 0.0) | ~jnp.isfinite(v8756412), 0.0, jnp.divide(1.0, jnp.where((v8756412 == 0.0) | ~jnp.isfinite(v8756412), 1.0, v8756412)))
    v8756422 = jnp.where(v8756409, v8756414 * v8756413, v8756396)
    v8759414 = i_v6112 * v8654118
    v8759416 = v8759414 >= -0.9
    v8759418 = 1.0 + v8759414
    v8759425 = 17.0 + 20.0 * v8759414
    v8759421 = 0.8 + v8759414
    v8759422 = jnp.where((v8759421 == 0.0) | ~jnp.isfinite(v8759421), 0.0, jnp.divide(1.0, jnp.where((v8759421 == 0.0) | ~jnp.isfinite(v8759421), 1.0, v8759421)))
    v8759430 = jnp.where(v8759416, jnp.where((v8759418 == 0.0) | ~jnp.isfinite(v8759418), 0.0, jnp.divide(1.0, jnp.where((v8759418 == 0.0) | ~jnp.isfinite(v8759418), 1.0, v8759418))), v8759425 * v8759422)
    v8759431 = v8756422 * v8759430
    v8813799 = v8759431 * v8810043 + v8813708
    v8813800 = jnp.where((v8813799 == 0.0) | ~jnp.isfinite(v8813799), 0.0, jnp.divide(1.0, jnp.where((v8813799 == 0.0) | ~jnp.isfinite(v8813799), 1.0, v8813799)))
    v8813837 = jnp.where((i_v9520 == 0.0) | ~jnp.isfinite(i_v9520), 0.0, jnp.divide(2.0, jnp.where((i_v9520 == 0.0) | ~jnp.isfinite(i_v9520), 1.0, i_v9520))) - 1.0
    v8809942 = v8740245 * i_v7043 * i_v5750
    v8809991 = v8809942 * v8745483
    v8813823 = v8759431 * v8809991
    v8813843 = v8813708 * v8813837 + v8759431 * v8810043 + 3.0 * (v8813708 * v8813823)
    v8813828 = 2.0 * v8759431
    v8813834 = v8813823 - 1.0 + jnp.where((i_v9520 == 0.0) | ~jnp.isfinite(i_v9520), 0.0, jnp.divide(1.0, jnp.where((i_v9520 == 0.0) | ~jnp.isfinite(i_v9520), 1.0, i_v9520)))
    v8813835 = v8813828 * v8813834
    v8813848 = 2.0 * v8813835
    v8813845 = v8810043 + 2.0 * (v8813708 * v8809991)
    v8813846 = v8813708 * v8813845
    v8813851 = jnp.sqrt(jnp.maximum(v8813843 * v8813843 - v8813848 * v8813846, 1e-300))
    v8813852 = v8813843 - v8813851
    v8813856 = jnp.where(v8813727, v8813803 * v8813800, jnp.where((v8813835 == 0.0) | ~jnp.isfinite(v8813835), 0.0, jnp.divide(v8813852, jnp.where((v8813835 == 0.0) | ~jnp.isfinite(v8813835), 1.0, v8813835))))
    v8814779 = v8813856 - v8653449 - i_v6598
    v8814788 = v8814779 >= 0.0
    v8814781 = 4.0 * i_v6598
    v8814784 = jnp.sqrt(jnp.maximum(v8814779 * v8814779 + v8814781 * v8813856, 1e-300))
    v8814799 = v8814784 - v8814779
    v8814801 = 1.0 - jnp.where((v8814799 == 0.0) | ~jnp.isfinite(v8814799), 0.0, jnp.divide(i_v9705, jnp.where((v8814799 == 0.0) | ~jnp.isfinite(v8814799), 1.0, v8814799)))
    v8814811 = jnp.where(v8814810, 0.0, jnp.where(v8814788, v8813856 - 0.5 * (v8814779 + v8814784), v8813856 * v8814801))
    v8814817 = v8814811 > v8653449
    v8814821 = jnp.where(v8814817, v8653449, v8814811)
    v8840315 = 0.5 * v8814821
    v8840302 = jnp.where((v8813708 == 0.0) | ~jnp.isfinite(v8813708), 0.0, jnp.divide(v8759431, jnp.where((v8813708 == 0.0) | ~jnp.isfinite(v8813708), 1.0, v8813708)))
    v8840317 = 1.0 - v8840315 * v8840302
    v8840318 = v8734229 * v8840317
    v8840326 = v8840293 * v8840318
    v8821979 = i_v6880 > 0.0
    v8822061 = i_v6006 * v8809061
    v8822065 = jnp.where((v8822061 == 0.0) | ~jnp.isfinite(v8822061), 0.0, jnp.divide(i_v6880, jnp.where((v8822061 == 0.0) | ~jnp.isfinite(v8822061), 1.0, v8822061)))
    v8814823 = v8653449 - v8814821
    v8825021 = v8809996 * i_v7144
    v8825022 = jnp.where((v8825021 == 0.0) | ~jnp.isfinite(v8825021), 0.0, jnp.divide(1.0, jnp.where((v8825021 == 0.0) | ~jnp.isfinite(v8825021), 1.0, v8825021)))
    v8825033 = 1.0 + v8814823 * v8825022
    v8825035 = v8825033 * v8825033 + 1.0
    v8825037 = 1.0 - jnp.where((v8825035 == 0.0) | ~jnp.isfinite(v8825035), 0.0, jnp.divide(2.0, jnp.where((v8825035 == 0.0) | ~jnp.isfinite(v8825035), 1.0, v8825035)))
    v8825039 = 1.0 + v8822065 * v8825037
    v8825063 = v8810043 * v8825039
    v8825196 = jnp.where(v8821979, v8825063, v8810043)
    v8840324 = 1.0 + jnp.where((v8825196 == 0.0) | ~jnp.isfinite(v8825196), 0.0, jnp.divide(v8814821, jnp.where((v8825196 == 0.0) | ~jnp.isfinite(v8825196), 1.0, v8825196)))
    v8840327 = jnp.where((v8840324 == 0.0) | ~jnp.isfinite(v8840324), 0.0, jnp.divide(v8840326, jnp.where((v8840324 == 0.0) | ~jnp.isfinite(v8840324), 1.0, v8840324)))
    v8840363 = 1.0 + v8840327 * v8745483
    v8840364 = jnp.where((v8840363 == 0.0) | ~jnp.isfinite(v8840363), 0.0, jnp.divide(v8840327, jnp.where((v8840363 == 0.0) | ~jnp.isfinite(v8840363), 1.0, v8840363)))
    v8850955 = i_v6670 * v8654118
    v8850957 = v8850955 >= -0.9
    v8847853 = v8759431 * v8813856
    v8847872 = v8813708 * v8847853
    v8847873 = v8813708 + v8847853
    v8847878 = jnp.where((i_v7353 == 0.0) | ~jnp.isfinite(i_v7353), 0.0, jnp.divide(v8813708 - jnp.where((v8847873 == 0.0) | ~jnp.isfinite(v8847873), 0.0, jnp.divide(v8847872, jnp.where((v8847873 == 0.0) | ~jnp.isfinite(v8847873), 1.0, v8847873))), jnp.where((i_v7353 == 0.0) | ~jnp.isfinite(i_v7353), 1.0, i_v7353)))
    v8850959 = 1.0 + v8850955
    v8850960 = jnp.where((v8850959 == 0.0) | ~jnp.isfinite(v8850959), 0.0, jnp.divide(1.0, jnp.where((v8850959 == 0.0) | ~jnp.isfinite(v8850959), 1.0, v8850959)))
    v8850967 = 17.0 + 20.0 * v8850955
    v8850964 = 0.8 + v8850955
    v8850965 = jnp.where((v8850964 == 0.0) | ~jnp.isfinite(v8850964), 0.0, jnp.divide(1.0, jnp.where((v8850964 == 0.0) | ~jnp.isfinite(v8850964), 1.0, v8850964)))
    v8850968 = v8850967 * v8850965
    v8850971 = jnp.where(v8850957, v8847878 * v8850960, v8847878 * v8850968)
    v8843984 = jnp.where((v8825196 == 0.0) | ~jnp.isfinite(v8825196), 0.0, jnp.divide(i_v6688, jnp.where((v8825196 == 0.0) | ~jnp.isfinite(v8825196), 1.0, v8825196)))
    v8843994 = v8843984 * v8734229
    v8843996 = v8843994 > -0.9
    v8844004 = 0.8 + v8843994
    v8844002 = 17.0 + 20.0 * v8843994
    v8844003 = jnp.where((v8844002 == 0.0) | ~jnp.isfinite(v8844002), 0.0, jnp.divide(1.0, jnp.where((v8844002 == 0.0) | ~jnp.isfinite(v8844002), 1.0, v8844002)))
    v8844931 = jnp.where(v8843996, 1.0 + v8843994, v8844004 * v8844003)
    v8850981 = v8850971 * v8844931
    v8859396 = 1.0 + jnp.where((v8850981 == 0.0) | ~jnp.isfinite(v8850981), 0.0, jnp.divide(v8814823, jnp.where((v8850981 == 0.0) | ~jnp.isfinite(v8850981), 1.0, v8850981)))
    v8859447 = v8840364 * v8859396
    v8854726 = i_v6586 > 1.713908431e-15
    v8854086 = i_v6592 * v8653449
    v8854087 = v8854086 > 34.0
    v8854089 = jnp.exp(jnp.clip(v8854086, -709.0, 709.0))
    v8855389 = jnp.where((i_v6586 == 0.0) | ~jnp.isfinite(i_v6586), 0.0, jnp.divide(1.0 + i_v9724 * jnp.where(v8854087, 583461742500000.0, v8854089), jnp.where((i_v6586 == 0.0) | ~jnp.isfinite(i_v6586), 1.0, i_v6586)))
    v8840975 = i_v6580 <= 0.0
    v8840983 = i_v6580 * jnp.sqrt(jnp.maximum(i_v6006, 1e-300))
    v8840989 = 1.0 + jnp.where((v8813708 == 0.0) | ~jnp.isfinite(v8813708), 0.0, jnp.divide(v8840983, jnp.where((v8813708 == 0.0) | ~jnp.isfinite(v8813708), 1.0, v8813708)))
    v8840993 = jnp.where(v8840975, 1.0, jnp.where((v8840989 == 0.0) | ~jnp.isfinite(v8840989), 0.0, jnp.divide(1.0, jnp.where((v8840989 == 0.0) | ~jnp.isfinite(v8840989), 1.0, v8840989))))
    v8855415 = jnp.where(v8854726, v8855389 * v8840993, 583461742500000.0)
    v8859473 = 1.0 + jnp.where((v8855415 == 0.0) | ~jnp.isfinite(v8855415), 0.0, jnp.divide(v8814823, jnp.where((v8855415 == 0.0) | ~jnp.isfinite(v8855415), 1.0, v8855415)))
    v8859477 = v8859447 * v8859473
    v8825199 = 2.0 * (v8809991 * v8734229)
    v8825090 = 0.5 * v8759431
    v8825110 = v8825090 * v8813856
    v8825137 = 1.0 - jnp.where((v8813708 == 0.0) | ~jnp.isfinite(v8813708), 0.0, jnp.divide(v8825110, jnp.where((v8813708 == 0.0) | ~jnp.isfinite(v8813708), 1.0, v8813708)))
    v8825201 = v8825196 + v8813856 + v8825199 * v8825137
    v8825231 = jnp.where((i_v9520 == 0.0) | ~jnp.isfinite(i_v9520), 0.0, jnp.divide(2.0, jnp.where((i_v9520 == 0.0) | ~jnp.isfinite(i_v9520), 1.0, i_v9520))) - 1.0 + v8809991 * v8759431
    v8825232 = jnp.where((v8825231 == 0.0) | ~jnp.isfinite(v8825231), 0.0, jnp.divide(v8825201, jnp.where((v8825231 == 0.0) | ~jnp.isfinite(v8825231), 1.0, v8825231)))
    v8844831 = jnp.where(i_v6652 > 1.713908431e-15, v8814823 > 1e-10, False)
    v8844933 = v8840993 * v8844931
    v8844864 = 1.0 + v8745483 * v8840364
    v8844934 = v8844933 * v8844864
    v8844897 = jnp.where(v8821979, jnp.where((i_v6006 == 0.0) | ~jnp.isfinite(i_v6006), 0.0, jnp.divide(v8825063, jnp.where((i_v6006 == 0.0) | ~jnp.isfinite(i_v6006), 1.0, i_v6006))), v8809996)
    v8844918 = i_v6006 + jnp.where((v8844897 == 0.0) | ~jnp.isfinite(v8844897), 0.0, jnp.divide(v8813856, jnp.where((v8844897 == 0.0) | ~jnp.isfinite(v8844897), 1.0, v8844897)))
    v8844961 = i_v6652 * i_v7144
    v8844962 = jnp.where((v8844961 == 0.0) | ~jnp.isfinite(v8844961), 0.0, jnp.divide(v8844934 * v8844918, jnp.where((v8844961 == 0.0) | ~jnp.isfinite(v8844961), 1.0, v8844961)))
    v8851029 = v8825232 + jnp.where(v8844831, v8844962 * v8814823, 583461742500000.0)
    v8859546 = jnp.where((v8825232 == 0.0) | ~jnp.isfinite(v8825232), 0.0, jnp.divide(v8851029, jnp.where((v8825232 == 0.0) | ~jnp.isfinite(v8825232), 1.0, v8825232)))
    v8859547 = jnp.log(jnp.maximum(v8859546, 1e-300))
    v8859574 = jnp.where(v8844831, v8844962, 583461742500000.0)
    v8859595 = 1.0 + jnp.where((v8859574 == 0.0) | ~jnp.isfinite(v8859574), 0.0, jnp.divide(v8859547, jnp.where((v8859574 == 0.0) | ~jnp.isfinite(v8859574), 1.0, v8859574)))
    v8859598 = v8859477 * v8859595
    v8859326 = init[50] * i_v7144
    v8859344 = jnp.exp(jnp.clip(jnp.where((v8814823 == 0.0) | ~jnp.isfinite(v8814823), 0.0, jnp.divide(v8859326, jnp.where((v8814823 == 0.0) | ~jnp.isfinite(v8814823), 1.0, v8814823))), -709.0, 709.0))
    v8859351 = jnp.where((i_v6682 == 0.0) | ~jnp.isfinite(i_v6682), 0.0, jnp.divide(i_v6006 * v8859344, jnp.where((i_v6682 == 0.0) | ~jnp.isfinite(i_v6682), 1.0, i_v6682)))
    v8868904 = 1.0 + jnp.where((v8859351 == 0.0) | ~jnp.isfinite(v8859351), 0.0, jnp.divide(v8814823, jnp.where((v8859351 == 0.0) | ~jnp.isfinite(v8859351), 1.0, v8859351)))
    v8868910 = v8859598 * v8868904
    v8868917 = v8868910 * v8814821
    v9336426 = jnp.where(v9123179, v8868917 * nf, v8868917)
    v9336922 = -1.0 * jnp.where(v9336400, -v9336426, v9336426)
    v9042302 = i_v7153 + init[76]
    v9061923 = v8471984 - v9042302
    v9061926 = jnp.sqrt(jnp.maximum(v9061923 * v9061923 + 0.0001, 1e-300))
    v9074915 = i_v7199 * (v8471984 * v9061926)
    v9074893 = i_v9824 * v9061926
    v9074902 = jnp.exp(jnp.clip(i_v7202 * (init[37] + i_v9823 * v9061926 - v9074893 * v9061926), -709.0, 709.0))
    v9074918 = v9074915 * v9074902
    v9356292 = -1.0 * jnp.where(v9123179, v9074918 * nf, v9074918)
    v9006450 = init[161] * 1.7763568394002489e-15
    v9031743 = i_v7205 * (v8725872 * v9006450)
    v8984545 = i_v7689 - v8725872 + v8654118 - 0.02
    v8984547 = i_v7689 <= 0.0
    v8984554 = jnp.sqrt(jnp.maximum(v8984545 * v8984545 - 0.08 * i_v7689, 1e-300))
    v8984561 = jnp.sqrt(jnp.maximum(v8984545 * v8984545 + 0.08 * i_v7689, 1e-300))
    v8984569 = 0.5 * (v8984545 + jnp.where(v8984547, v8984554, v8984561))
    v8985072 = -(v8725872 - (i_v7689 - v8984569) - v8654118 - v8734229) + v8734229
    v9031721 = init[163] * v8985072
    v9031730 = jnp.exp(jnp.clip(i_v7207 * (init[60] + i_v9815 * v8985072 - v9031721 * v8985072), -709.0, 709.0))
    v9031746 = v9031743 * v9031730
    v9038568 = v8734229 + 1e-20
    v9038569 = jnp.where((v9038568 == 0.0) | ~jnp.isfinite(v9038568), 0.0, jnp.divide(i_v9817, jnp.where((v9038568 == 0.0) | ~jnp.isfinite(v9038568), 1.0, v9038568)))
    v9038570 = jnp.where((v9038568 == 0.0) | ~jnp.isfinite(v9038568), 0.0, jnp.divide(v9038569, jnp.where((v9038568 == 0.0) | ~jnp.isfinite(v9038568), 1.0, v9038568)))
    v9038767 = 1.0 - jnp.where((v9038568 == 0.0) | ~jnp.isfinite(v9038568), 0.0, jnp.divide(v8840315, jnp.where((v9038568 == 0.0) | ~jnp.isfinite(v9038568), 1.0, v9038568)))
    v9038770 = -init[64]
    v9038773 = v9038770 * v8814821
    v9038781 = jnp.exp(jnp.clip(v9038773, -709.0, 709.0))
    v9038791 = v9038781 - 1.0
    v9038813 = v9038773 * v9038781 - (v9038791 - 0.0001)
    v9038775 = v9038773 * v9038773 + 0.0002
    v8868766 = v8814823 > jnp.divide(i_v6736, 34.0)
    v8868772 = i_v9736 * v8814823
    v8868768 = -i_v6736
    v8868773 = jnp.exp(jnp.clip(jnp.where((v8814823 == 0.0) | ~jnp.isfinite(v8814823), 0.0, jnp.divide(v8868768, jnp.where((v8814823 == 0.0) | ~jnp.isfinite(v8814823), 1.0, v8814823))), -709.0, 709.0))
    v8868779 = i_v9736 * 1.713908431e-15
    v8868865 = jnp.where(v8868766, v8868772 * v8868773, v8868779 * v8814823)
    v8868864 = v8859598 * v8814821
    v8868866 = v8868865 * v8868864
    v8925988 = jnp.where((i_v9766 == 0.0) | ~jnp.isfinite(i_v9766), 0.0, jnp.divide(v8471931 - v8725865 - init[59], jnp.where((i_v9766 == 0.0) | ~jnp.isfinite(i_v9766), 1.0, i_v9766)))
    v8935352 = jnp.where((v8925988 == 0.0) | ~jnp.isfinite(v8925988), 0.0, jnp.divide(i_v6748, jnp.where((v8925988 == 0.0) | ~jnp.isfinite(v8925988), 1.0, v8925988)))
    v8935353 = v8935352 < 100.0
    v8935854 = i_v6742 * i_v6022
    v8935856 = v8935854 * v8925988
    v8935859 = jnp.exp(jnp.clip(-v8935352, -709.0, 709.0))
    v8935868 = i_v6742 * i_v6022 * 3.720075976e-44
    v8935891 = jnp.where(v8935353, v8935856 * v8935859, v8935868 * v8925988)
    v8935878 = -v8471980
    v8935877 = v8471980 * v8471980
    v8935879 = v8935878 * v8935877
    v8935889 = init[58] + v8935879
    v8935890 = jnp.where((v8935889 == 0.0) | ~jnp.isfinite(v8935889), 0.0, jnp.divide(v8935879, jnp.where((v8935889 == 0.0) | ~jnp.isfinite(v8935889), 1.0, v8935889)))
    v8935892 = v8935891 * v8935890
    v9366193 = -1.0 * (jnp.where(v9123179, v8868866 * nf, v8868866) + jnp.where(v9123179, v8935892 * nf, v8935892))
    v8608056 = 10.0 - v8471980
    v8608058 = v8608056 < 0.01
    v8608114 = jnp.where((i_v9593 == 0.0) | ~jnp.isfinite(i_v9593), 0.0, jnp.divide(-v8471980, jnp.where((i_v9593 == 0.0) | ~jnp.isfinite(i_v9593), 1.0, i_v9593))) * 10.0
    v8608110 = jnp.where((v8608056 == 0.0) | ~jnp.isfinite(v8608056), 0.0, jnp.divide(1.0, jnp.where((v8608056 == 0.0) | ~jnp.isfinite(v8608056), 1.0, v8608056)))
    v8609175 = 10.0 - v8471980
    v8609177 = v8609175 < 0.01
    v8609263 = jnp.where((i_v9591 == 0.0) | ~jnp.isfinite(i_v9591), 0.0, jnp.divide(-v8471980, jnp.where((i_v9591 == 0.0) | ~jnp.isfinite(i_v9591), 1.0, i_v9591))) * 10.0
    v8609259 = jnp.where((v8609175 == 0.0) | ~jnp.isfinite(v8609175), 0.0, jnp.divide(1.0, jnp.where((v8609175 == 0.0) | ~jnp.isfinite(v8609175), 1.0, v8609175)))
    v8610402 = 10.0 - v8471980
    v8610404 = v8610402 < 0.01
    v8610520 = jnp.where((i_v9592 == 0.0) | ~jnp.isfinite(i_v9592), 0.0, jnp.divide(-v8471980, jnp.where((i_v9592 == 0.0) | ~jnp.isfinite(i_v9592), 1.0, i_v9592))) * 10.0
    v8610516 = jnp.where((v8610402 == 0.0) | ~jnp.isfinite(v8610402), 0.0, jnp.divide(1.0, jnp.where((v8610402 == 0.0) | ~jnp.isfinite(v8610402), 1.0, v8610402)))
    v9563754 = -1.0 * (_ckt_gmin * v8471980 - (i_v8982 * (jnp.where(v8608058, 583461742500000.0 * (1.0 + jnp.where((i_v9593 == 0.0) | ~jnp.isfinite(i_v9593), 0.0, jnp.divide(-v8471980, jnp.where((i_v9593 == 0.0) | ~jnp.isfinite(i_v9593), 1.0, i_v9593))) * 1000.0 - 34.0), 583461742500000.0 * (1.0 + v8608114 * v8608110 - 34.0)) - 1.0) + i_v8986 * (jnp.where(v8609177, 583461742500000.0 * (1.0 + jnp.where((i_v9591 == 0.0) | ~jnp.isfinite(i_v9591), 0.0, jnp.divide(-v8471980, jnp.where((i_v9591 == 0.0) | ~jnp.isfinite(i_v9591), 1.0, i_v9591))) * 1000.0 - 34.0), 583461742500000.0 * (1.0 + v8609263 * v8609259 - 34.0)) - 1.0) + i_v8992 * (jnp.where(v8610404, 583461742500000.0 * (1.0 + jnp.where((i_v9592 == 0.0) | ~jnp.isfinite(i_v9592), 0.0, jnp.divide(-v8471980, jnp.where((i_v9592 == 0.0) | ~jnp.isfinite(i_v9592), 1.0, i_v9592))) * 1000.0 - 34.0), 583461742500000.0 * (1.0 + v8610520 * v8610516 - 34.0)) - 1.0)))
    v9151211 = -(2.0 * jnp.divide(i_v9853, 3.0))
    v9299631 = 2.0 == 0
    v9297198 = 0.0 == 3
    v8471947 = -1.0 * (signals.g - signals.s)
    v9300633 = jnp.where(v9297198, v8471947 - v8471931, v8471984)
    v9306440 = i_v9513 + i_v9918
    v9301638 = v9300633 + 0.02
    v9301642 = jnp.sqrt(jnp.maximum(v9301638 * v9301638 + 0.08, 1e-300))
    v9301644 = 0.5 * (v9301638 - v9301642)
    v9306442 = 0.5 * i_v9274
    v9306438 = jnp.sqrt(jnp.maximum(1.0 - jnp.where((i_v9274 == 0.0) | ~jnp.isfinite(i_v9274), 0.0, jnp.divide(4.0 * v9301644, jnp.where((i_v9274 == 0.0) | ~jnp.isfinite(i_v9274), 1.0, i_v9274))), 1e-300))
    v9311262 = jnp.where(v9299631, i_v9513 * v9300633, v9306440 * v9300633 - i_v9918 * (v9301644 + v9306442 * (v9306438 - 1.0)))
    v9145086 = v8654118 < 0.0
    v9151041 = i_v9853 * (v8725872 - jnp.where(v9145086, v8650925, v8654118) - i_v6934)
    v9306449 = jnp.where(v9297198, v8471947, v8471935) + 0.02
    v9306453 = jnp.sqrt(jnp.maximum(v9306449 * v9306449 + 0.08, 1e-300))
    v9947397 = -1.0 * (0.4 * (v9151211 * (v8725872 - (i_v6934 + i_v7125 + v8689484))) - jnp.where(v9123179, v9311262 * nf, v9311262))
    v9296786 = jnp.where((init[6] == 0.0) | ~jnp.isfinite(init[6]), 0.0, jnp.divide(i_v9901 * init[128], jnp.where((init[6] == 0.0) | ~jnp.isfinite(init[6]), 1.0, init[6]))) + jnp.where((init[8] == 0.0) | ~jnp.isfinite(init[8]), 0.0, jnp.divide(i_v9903 * init[127], jnp.where((init[8] == 0.0) | ~jnp.isfinite(init[8]), 1.0, init[8]))) + jnp.where((init[10] == 0.0) | ~jnp.isfinite(init[10]), 0.0, jnp.divide(i_v9905 * init[126], jnp.where((init[10] == 0.0) | ~jnp.isfinite(init[10]), 1.0, init[10])))
    v9296789 = i_v9901 + i_v9903 + i_v9905 + 0.5 * (v8471980 * v9296786)
    v9947401 = -1.0 * (v8471980 * v9296789)
    v9042303 = v8471935 - v9042302
    v9042306 = jnp.sqrt(jnp.maximum(v9042303 * v9042303 + 0.0001, 1e-300))
    v9058435 = i_v7197 * (v8471935 * v9042306)
    v9058413 = i_v9821 * v9042306
    v9058422 = jnp.exp(jnp.clip(i_v7202 * (init[36] + i_v9820 * v9042306 - v9058413 * v9042306), -709.0, 709.0))
    v9058438 = v9058435 * v9058422
    v9346607 = -1.0 * jnp.where(v9123179, v9058438 * nf, v9058438)
    v9038797 = v9038791 + 0.0001 - v9038773
    v9038798 = jnp.where((v9038775 == 0.0) | ~jnp.isfinite(v9038775), 0.0, jnp.divide(v9038797, jnp.where((v9038775 == 0.0) | ~jnp.isfinite(v9038775), 1.0, v9038775)))
    v8939333 = jnp.where((i_v9766 == 0.0) | ~jnp.isfinite(i_v9766), 0.0, jnp.divide(-v8471931 - v8725871 - init[41], jnp.where((i_v9766 == 0.0) | ~jnp.isfinite(i_v9766), 1.0, i_v9766)))
    v8948799 = jnp.where((v8939333 == 0.0) | ~jnp.isfinite(v8939333), 0.0, jnp.divide(i_v6430, jnp.where((v8939333 == 0.0) | ~jnp.isfinite(v8939333), 1.0, v8939333)))
    v8948800 = v8948799 < 100.0
    v8948835 = i_v6424 * i_v6022
    v8948837 = v8948835 * v8939333
    v8948840 = jnp.exp(jnp.clip(-v8948799, -709.0, 709.0))
    v8948845 = i_v6424 * i_v6022 * 3.720075976e-44
    v8948868 = jnp.where(v8948800, v8948837 * v8948840, v8948845 * v8939333)
    v8948855 = -v8471939
    v8948854 = v8471939 * v8471939
    v8948856 = v8948855 * v8948854
    v8948866 = init[40] + v8948856
    v8948867 = jnp.where((v8948866 == 0.0) | ~jnp.isfinite(v8948866), 0.0, jnp.divide(v8948856, jnp.where((v8948866 == 0.0) | ~jnp.isfinite(v8948866), 1.0, v8948866)))
    v8948869 = v8948868 * v8948867
    v9375878 = -1.0 * jnp.where(v9123179, v8948869 * nf, v8948869)
    v8607542 = 10.0 - v8471939
    v8607544 = v8607542 < 0.01
    v8607585 = jnp.where((i_v9590 == 0.0) | ~jnp.isfinite(i_v9590), 0.0, jnp.divide(-v8471939, jnp.where((i_v9590 == 0.0) | ~jnp.isfinite(i_v9590), 1.0, i_v9590))) * 10.0
    v8607581 = jnp.where((v8607542 == 0.0) | ~jnp.isfinite(v8607542), 0.0, jnp.divide(1.0, jnp.where((v8607542 == 0.0) | ~jnp.isfinite(v8607542), 1.0, v8607542)))
    v8608602 = 10.0 - v8471939
    v8608604 = v8608602 < 0.01
    v8608675 = jnp.where((i_v9588 == 0.0) | ~jnp.isfinite(i_v9588), 0.0, jnp.divide(-v8471939, jnp.where((i_v9588 == 0.0) | ~jnp.isfinite(i_v9588), 1.0, i_v9588))) * 10.0
    v8608671 = jnp.where((v8608602 == 0.0) | ~jnp.isfinite(v8608602), 0.0, jnp.divide(1.0, jnp.where((v8608602 == 0.0) | ~jnp.isfinite(v8608602), 1.0, v8608602)))
    v8609775 = 10.0 - v8471939
    v8609777 = v8609775 < 0.01
    v8609878 = jnp.where((i_v9589 == 0.0) | ~jnp.isfinite(i_v9589), 0.0, jnp.divide(-v8471939, jnp.where((i_v9589 == 0.0) | ~jnp.isfinite(i_v9589), 1.0, i_v9589))) * 10.0
    v8609874 = jnp.where((v8609775 == 0.0) | ~jnp.isfinite(v8609775), 0.0, jnp.divide(1.0, jnp.where((v8609775 == 0.0) | ~jnp.isfinite(v8609775), 1.0, v8609775)))
    v9553466 = -1.0 * (_ckt_gmin * v8471939 - (i_v8980 * (jnp.where(v8607544, 583461742500000.0 * (1.0 + jnp.where((i_v9590 == 0.0) | ~jnp.isfinite(i_v9590), 0.0, jnp.divide(-v8471939, jnp.where((i_v9590 == 0.0) | ~jnp.isfinite(i_v9590), 1.0, i_v9590))) * 1000.0 - 34.0), 583461742500000.0 * (1.0 + v8607585 * v8607581 - 34.0)) - 1.0) + i_v8984 * (jnp.where(v8608604, 583461742500000.0 * (1.0 + jnp.where((i_v9588 == 0.0) | ~jnp.isfinite(i_v9588), 0.0, jnp.divide(-v8471939, jnp.where((i_v9588 == 0.0) | ~jnp.isfinite(i_v9588), 1.0, i_v9588))) * 1000.0 - 34.0), 583461742500000.0 * (1.0 + v8608675 * v8608671 - 34.0)) - 1.0) + i_v8989 * (jnp.where(v8609777, 583461742500000.0 * (1.0 + jnp.where((i_v9589 == 0.0) | ~jnp.isfinite(i_v9589), 0.0, jnp.divide(-v8471939, jnp.where((i_v9589 == 0.0) | ~jnp.isfinite(i_v9589), 1.0, i_v9589))) * 1000.0 - 34.0), 583461742500000.0 * (1.0 + v8609878 * v8609874 - 34.0)) - 1.0)))
    v9947395 = -1.0 * v9151041
    v9947398 = -1.0 * (-v9151041 - i_v7113 * (v8471947 - v8471939))
    v9285118 = jnp.where((init[5] == 0.0) | ~jnp.isfinite(init[5]), 0.0, jnp.divide(i_v9902 * init[131], jnp.where((init[5] == 0.0) | ~jnp.isfinite(init[5]), 1.0, init[5]))) + jnp.where((init[7] == 0.0) | ~jnp.isfinite(init[7]), 0.0, jnp.divide(i_v9906 * init[130], jnp.where((init[7] == 0.0) | ~jnp.isfinite(init[7]), 1.0, init[7]))) + jnp.where((init[9] == 0.0) | ~jnp.isfinite(init[9]), 0.0, jnp.divide(i_v9908 * init[129], jnp.where((init[9] == 0.0) | ~jnp.isfinite(init[9]), 1.0, init[9])))
    v9285121 = i_v9902 + i_v9906 + i_v9908 + 0.5 * (v8471939 * v9285118)
    v9947400 = -1.0 * (v8471939 * v9285121)
    v9085301 = v8725872 - v8654118
    v9112940 = v8985072 - init[63]
    v9122923 = i_v9839 * (v9085301 * v9112940)
    v9122901 = init[176] * v8985072
    v9122910 = jnp.exp(jnp.clip(i_v9841 * (init[62] + i_v9844 * v8985072 - v9122901 * v8985072), -709.0, 709.0))
    v9085271 = -v8725872 + v8654118 + init[160]
    v9108680 = i_v9830 * (v9085301 * v9085271)
    v8984572 = v8984569 < 0.0
    v9108589 = jnp.where(v8984572, 0.0, v8984569)
    v9108658 = init[172] * v9108589
    v9108667 = jnp.exp(jnp.clip(i_v9831 * (init[61] + i_v9834 * v9108589 - v9108658 * v9108589), -709.0, 709.0))
    v9122950 = v9122923 * v9122910 + v9108680 * v9108667
    v9386314 = -1.0 * jnp.where(v9123179, v9122950 * nf, v9122950)
    v8891105 = nf != 1.0
    v8890903 = i_v6868 * (i_v9749 * v8840293 + v8868910)
    v9447351 = jnp.where(v8891105, v8890903 * nf, v8890903) + i_v7837
    v9447352 = 0.0
    v10042191 = 0.0 - -1.0
    v10045386 = 0.0 - jnp.where((v8904488 * v8904488 == 0.0) | ~jnp.isfinite(v8904488 * v8904488), 0.0, jnp.divide((1.0 + jnp.divide(v8904479 + v8904479, 2.0 * v8904482)) * 0.5 * i_v9524, jnp.where((v8904488 * v8904488 == 0.0) | ~jnp.isfinite(v8904488 * v8904488), 1.0, v8904488 * v8904488)))
    v10045397 = (1.0 + jnp.divide(v8904830 + v8904830, 2.0 * v8904833)) * i_v9762
    v10048514 = 0.0
    v10048515 = (v10042191 * v10045386 + v10042191 * v8897760) * v10045397 * v10048514
    v10048516 = -1.0 * v10045386 * v10045397 * v10048514
    v10045344 = -1.0 * v8897760
    v10048517 = v10045344 * v10045397 * v10048514
    v10045348 = 0.0 - jnp.where((v8897549 * v8897549 == 0.0) | ~jnp.isfinite(v8897549 * v8897549), 0.0, jnp.divide((-1.0 + jnp.divide(-1.0 * v8897332 + -1.0 * v8897332, 2.0 * v8897335)) * 0.5 * i_v9524, jnp.where((v8897549 * v8897549 == 0.0) | ~jnp.isfinite(v8897549 * v8897549), 1.0, v8897549 * v8897549)))
    v10045359 = 2.0 * v8898098
    v10048500 = v9407273 * v9407273
    v10048507 = 0.0
    v10048508 = 0.0
    v10048679 = 0.0 - v9418326
    v10043294 = jnp.where((v8725738 == 0.0) | ~jnp.isfinite(v8725738), 0.0, jnp.divide(2.0, jnp.where((v8725738 == 0.0) | ~jnp.isfinite(v8725738), 1.0, v8725738))) - jnp.where((v8725738 * v8725738 == 0.0) | ~jnp.isfinite(v8725738 * v8725738), 0.0, jnp.divide(jnp.divide(jnp.where((v8725719 == 0.0) | ~jnp.isfinite(v8725719), 0.0, jnp.divide(2.0, jnp.where((v8725719 == 0.0) | ~jnp.isfinite(v8725719), 1.0, v8725719))), 2.0 * v8725736) * v8725733, jnp.where((v8725738 * v8725738 == 0.0) | ~jnp.isfinite(v8725738 * v8725738), 1.0, v8725738 * v8725738)))
    v10043301 = 0.0 - jnp.where((v8725719 == 0.0) | ~jnp.isfinite(v8725719), 0.0, jnp.divide(v10043294 * 0.5 * v8725739 + v10043294 * v8725740, jnp.where((v8725719 == 0.0) | ~jnp.isfinite(v8725719), 1.0, v8725719)))
    v10043314 = 1.0 - (0.0 - (v10043301 + jnp.divide(v10043301 * v8725744 + v10043301 * v8725744, 2.0 * v8725747)) * 0.5)
    v10043317 = jnp.where(v8725690, v10042191 * v10043314, v10042191)
    v10043319 = jnp.where(v8725855, 0.0, v10043317)
    v10042929 = jnp.where(v8650221, 0.0, v10042191)
    v10042940 = jnp.divide(v8653420 + v8653420, 2.0 * v8653425)
    v10042951 = (1.0 + v10042940) * 0.5
    v10042947 = (0.0 - jnp.where((v8653438 * v8653438 == 0.0) | ~jnp.isfinite(v8653438 * v8653438), 0.0, jnp.divide((v10042940 - 1.0) * -0.002, jnp.where((v8653438 * v8653438 == 0.0) | ~jnp.isfinite(v8653438 * v8653438), 1.0, v8653438 * v8653438)))) * i_v7686
    v10042955 = jnp.where(v8653426, v10042929 * v10042951, v10042929 * v10042947)
    v10042966 = (-1.0 + jnp.divide(-1.0 * v8654111 + -1.0 * v8654111, 2.0 * v8654115)) * 0.5
    v10042972 = jnp.divide(v10042966, 2.0 * v8654121)
    v10043125 = v10042972 * i_v7322
    v10042967 = 0.0 - v10042966
    v10043130 = v10043125 * i_v9619 - v10042967 * i_v7688
    v10042977 = jnp.divide(v10042972 * i_v7134, i_v7127)
    v10043003 = jnp.divide(v10042977, 2.0 * v8658363) * i_v5785
    v10043004 = v10042955 * v10043003
    v10042982 = v10042967 * i_v6262
    v10042995 = v10042982 * 3.0 * v8663768 + (0.0 - jnp.where((v8663767 * v8663767 == 0.0) | ~jnp.isfinite(v8663767 * v8663767), 0.0, jnp.divide(v10042982 * 8.0, jnp.where((v8663767 * v8663767 == 0.0) | ~jnp.isfinite(v8663767 * v8663767), 1.0, v8663767 * v8663767)))) * v8663770
    v10043039 = v8664498 * v8664498
    v10043058 = v8667931 - jnp.where((v8667930 * v8667930 == 0.0) | ~jnp.isfinite(v8667930 * v8667930), 0.0, jnp.divide((v8667925 * v8667926 + v8667925 * v8667926 + v8667929) * v8667925, jnp.where((v8667930 * v8667930 == 0.0) | ~jnp.isfinite(v8667930 * v8667930), 1.0, v8667930 * v8667930)))
    v10043061 = jnp.where(v8667923, (0.0 - jnp.where((v10043039 == 0.0) | ~jnp.isfinite(v10043039), 0.0, jnp.divide((v10043004 * v8664497 + jnp.where(v8663761, v10042955 * v10042982, v10042955 * v10042995) * v8664496) * v7319294, jnp.where((v10043039 == 0.0) | ~jnp.isfinite(v10043039), 1.0, v10043039)))) * v10043058, 0.0)
    v10043012 = v10042967 * i_v6280
    v10043025 = v10043012 * 3.0 * v8667278 + (0.0 - jnp.where((v8667277 * v8667277 == 0.0) | ~jnp.isfinite(v8667277 * v8667277), 0.0, jnp.divide(v10043012 * 8.0, jnp.where((v8667277 * v8667277 == 0.0) | ~jnp.isfinite(v8667277 * v8667277), 1.0, v8667277 * v8667277)))) * v8667280
    v10043067 = v8667293 * v8667293
    v10043086 = v8669500 - jnp.where((v8669499 * v8669499 == 0.0) | ~jnp.isfinite(v8669499 * v8669499), 0.0, jnp.divide((v8669494 * v8669495 + v8669494 * v8669495 + v8669498) * v8669494, jnp.where((v8669499 * v8669499 == 0.0) | ~jnp.isfinite(v8669499 * v8669499), 1.0, v8669499 * v8669499)))
    v10043139 = v10042967 * i_v6214 * i_v7387
    v10043097 = v10042967 * i_v6202 * i_v6965
    v10043101 = v10042967 * i_v6646
    v10043114 = (0.0 - v10043101) * v8683389 + (0.0 - jnp.where((v8683388 * v8683388 == 0.0) | ~jnp.isfinite(v8683388 * v8683388), 0.0, jnp.divide(0.0 - v10043101 * 20000.0, jnp.where((v8683388 * v8683388 == 0.0) | ~jnp.isfinite(v8683388 * v8683388), 1.0, v8683388 * v8683388)))) * v8683391
    v10042928 = jnp.where(v8650221, -1.0, --1.0)
    v10043146 = v10042955 * v10043130 - v10043061 * i_v6250 * i_v7354 - jnp.where(v8669492, (0.0 - jnp.where((v10043067 == 0.0) | ~jnp.isfinite(v10043067), 0.0, jnp.divide((v10043004 * v8667292 + jnp.where(v8667271, v10042955 * v10043012, v10042955 * v10043025) * v8664496) * v7318983, jnp.where((v10043067 == 0.0) | ~jnp.isfinite(v10043067), 1.0, v10043067)))) * v10043086, 0.0) * i_v6268 * i_v7354 + v10042955 * v10043139 + v10042955 * v10043097 - (jnp.where(v8683384, v10042955 * v10043114, v10042955 * v10043101) * i_v7337 * v8653449 + v10042928 * v8686040)
    v10043154 = (0.0 - jnp.divide(v10042977 * 1.03594e-10, v8657438 * v8657438)) * i_v7090
    v10043157 = v10042967 * i_v6046
    v10043175 = jnp.divide(v10042955 * v10043154 + ((v10042955 * v10043157 + v10042928 * i_v6052) * v8668387 + v10043061 * v8701956), i_v5750)
    v10043181 = v8701980 * v8701980
    v10043200 = jnp.where(v8701975, v10043175, v10043175 * 3.0 * v8701981 + (0.0 - jnp.where((v10043181 == 0.0) | ~jnp.isfinite(v10043181), 0.0, jnp.divide(v10043175 * 8.0, jnp.where((v10043181 == 0.0) | ~jnp.isfinite(v10043181), 1.0, v10043181)))) * v8701983)
    v10043321 = v10043319 - v10043146
    v10043329 = v8726063 * v8726063
    v10043408 = v8734227 * v8734227
    v10043412 = jnp.where((v8734227 == 0.0) | ~jnp.isfinite(v8734227), 0.0, jnp.divide(v10043321 * i_v7210, jnp.where((v8734227 == 0.0) | ~jnp.isfinite(v8734227), 1.0, v8734227))) - jnp.where((v10043408 == 0.0) | ~jnp.isfinite(v10043408), 0.0, jnp.divide(jnp.where(v8731479, v10043200 * v8734181, jnp.where(v8734190, v10043200 * v8734196, v10043200 * v8734208 + (jnp.where((v8726063 == 0.0) | ~jnp.isfinite(v8726063), 0.0, jnp.divide(0.0 - v10043321 * v8731462, jnp.where((v8726063 == 0.0) | ~jnp.isfinite(v8726063), 1.0, v8726063))) - jnp.where((v10043329 == 0.0) | ~jnp.isfinite(v10043329), 0.0, jnp.divide(v10043200 * i_v5824 * v8731470, jnp.where((v10043329 == 0.0) | ~jnp.isfinite(v10043329), 1.0, v10043329)))) * v8734204 * v8734207 * v8701987)) * v8728744, jnp.where((v10043408 == 0.0) | ~jnp.isfinite(v10043408), 1.0, v10043408)))
    v10044035 = jnp.where((4.23e-09 == 0.0) | ~jnp.isfinite(4.23e-09), 0.0, jnp.divide(v10043412 + v10043146 + v10043146, jnp.where((4.23e-09 == 0.0) | ~jnp.isfinite(4.23e-09), 1.0, 4.23e-09)))
    v10044030 = v10042967 * i_v7039
    v10044054 = v8775829 * v8775829
    v10044064 = (0.0 - jnp.where((v10044054 == 0.0) | ~jnp.isfinite(v10044054), 0.0, jnp.divide(v10043412 + jnp.divide(v10043146 * v8696278 + v10043146 * v8696278, v10044046) * 2.0, jnp.where((v10044054 == 0.0) | ~jnp.isfinite(v10044054), 1.0, v10044054)))) * 4.23e-09
    v10044108 = v10044035 * v8781625 + (v10042955 * v10044030 + v10044035 * i_v7037) * v8775822 + (((v10044064 * i_v7038 * v8775831 + v10044064 * v8778727) * v8696278 + v10043146 * v8778728) * v8696278 + v10043146 * v8778729)
    v10044120 = v8799115 * v8799115
    v10044148 = v8809059 * v8809059
    v10044151 = 0.0 - jnp.where((v10044148 == 0.0) | ~jnp.isfinite(v10044148), 0.0, jnp.divide(jnp.where(v8799108, v10044108, v10044108 * v8799116 + (0.0 - jnp.where((v10044120 == 0.0) | ~jnp.isfinite(v10044120), 0.0, jnp.divide(v10044108 * 10.0, jnp.where((v10044120 == 0.0) | ~jnp.isfinite(v10044120), 1.0, v10044120)))) * v8799117) * i_v7075, jnp.where((v10044148 == 0.0) | ~jnp.isfinite(v10044148), 1.0, v10044148)))
    v10044680 = 0.0 - jnp.where((v8840232 * v8840232 == 0.0) | ~jnp.isfinite(v8840232 * v8840232), 0.0, jnp.divide((0.0 - jnp.where((v8829539 * v8829539 == 0.0) | ~jnp.isfinite(v8829539 * v8829539), 0.0, jnp.divide(jnp.where((v8828641 == 0.0) | ~jnp.isfinite(v8828641), 0.0, jnp.divide(1.1820330969267139, jnp.where((v8828641 == 0.0) | ~jnp.isfinite(v8828641), 1.0, v8828641))) * 0.7 * v8829538 * 1.9e-09, jnp.where((v8829539 * v8829539 == 0.0) | ~jnp.isfinite(v8829539 * v8829539), 1.0, v8829539 * v8829539)))) * i_v5755 * 8.456821984368795e-13, jnp.where((v8840232 * v8840232 == 0.0) | ~jnp.isfinite(v8840232 * v8840232), 1.0, v8840232 * v8840232)))
    v10043425 = v10042972 * i_v6706
    v10043428 = v10043412 * i_v6700 + v10042955 * v10043425
    v10043445 = 4e-08 * v8740235 + (0.0 - jnp.where((v8740234 * v8740234 == 0.0) | ~jnp.isfinite(v8740234 * v8740234), 0.0, jnp.divide(4.0, jnp.where((v8740234 * v8740234 == 0.0) | ~jnp.isfinite(v8740234 * v8740234), 1.0, v8740234 * v8740234)))) * v8740238
    v10043449 = jnp.where(v8740230, v10043428 * v10043445, v10043428 * -2.0)
    v10044699 = v10044151 * v8840286 + jnp.where((i_v6006 == 0.0) | ~jnp.isfinite(i_v6006), 0.0, jnp.divide(v10043412 * v10044680 * v8740245 + v10043449 * v8840233, jnp.where((i_v6006 == 0.0) | ~jnp.isfinite(i_v6006), 1.0, i_v6006))) * v8809061
    v10044173 = v8809061 * v8809061
    v10044176 = 0.0 - jnp.where((v10044173 == 0.0) | ~jnp.isfinite(v10044173), 0.0, jnp.divide(v10044151 * v8809993, jnp.where((v10044173 == 0.0) | ~jnp.isfinite(v10044173), 1.0, v10044173)))
    v10044183 = v10044176 * i_v6006
    v10043484 = jnp.where((v8654119 == 0.0) | ~jnp.isfinite(v8654119), 0.0, jnp.divide(v10042972 * v8745944, jnp.where((v8654119 == 0.0) | ~jnp.isfinite(v8654119), 1.0, v8654119)))
    v10043498 = 0.0 - jnp.where((v8747172 * v8747172 == 0.0) | ~jnp.isfinite(v8747172 * v8747172), 0.0, jnp.divide(jnp.divide(v10042977 * i_v6070, v10043489) * 2.0 * i_v6006, jnp.where((v8747172 * v8747172 == 0.0) | ~jnp.isfinite(v8747172 * v8747172), 1.0, v8747172 * v8747172)))
    v10043509 = (0.0 - v10043484) * v8753495 + v10043498 * i_v6088 * v8746258
    v10043519 = v10043484 * v8756383 + ((v10043498 * v8747173 + v10043498 * v8747173) * v8747173 + v10043498 * v8753496) * v8756382 * v8756384
    v10043529 = v10042955 * v10043509 + (v10042955 * v10043519 * v8734229 + v10043412 * v8756385)
    v10043554 = v8756412 * v8756412
    v10043579 = v10042967 * i_v6112
    v10043598 = 0.0 - jnp.where((v8759418 * v8759418 == 0.0) | ~jnp.isfinite(v8759418 * v8759418), 0.0, jnp.divide(v10043579, jnp.where((v8759418 * v8759418 == 0.0) | ~jnp.isfinite(v8759418 * v8759418), 1.0, v8759418 * v8759418)))
    v10043591 = v10043579 * 20.0 * v8759422 + (0.0 - jnp.where((v8759421 * v8759421 == 0.0) | ~jnp.isfinite(v8759421 * v8759421), 0.0, jnp.divide(v10043579, jnp.where((v8759421 * v8759421 == 0.0) | ~jnp.isfinite(v8759421 * v8759421), 1.0, v8759421 * v8759421)))) * v8759425
    v10043605 = jnp.where(v8756409, (0.0 - v10043529) * v8756413 + (0.0 - jnp.where((v10043554 == 0.0) | ~jnp.isfinite(v10043554), 0.0, jnp.divide(0.0 - v10043529 * 20.0, jnp.where((v10043554 == 0.0) | ~jnp.isfinite(v10043554), 1.0, v10043554)))) * v8756414, v10043529) * v8759430 + jnp.where(v8759416, v10042955 * v10043598, v10042955 * v10043591) * v8756422
    v10044389 = v8813799 * v8813799
    v10044199 = 0.0 - i_v9521
    v10044209 = 0.0 - (v10044199 + jnp.divide(v10044199 * v8813590 + v10044199 * v8813590, 2.0 * jnp.sqrt(jnp.maximum(v8813590 * v8813590 + 0.0004 * i_v9698, 1e-300)))) * 0.5
    v10044210 = v10043412 * v10044209
    v10044256 = i_v9520 * i_v9520
    v10043459 = 0.0 - jnp.where((v8741568 * v8741568 == 0.0) | ~jnp.isfinite(v8741568 * v8741568), 0.0, jnp.divide(i_v9524, jnp.where((v8741568 * v8741568 == 0.0) | ~jnp.isfinite(v8741568 * v8741568), 1.0, v8741568 * v8741568)))
    v10043453 = v10042972 * i_v6628
    v10043474 = (1.0 + jnp.divide(v8744451 + v8744451, 2.0 * v8744454)) * i_v9640
    v10043476 = (v10043412 * v10043459 + v10042955 * v10043453) * v10043474
    v10044166 = v10043449 * i_v7043 * i_v5750 * v8745483 + v10043476 * v8809942
    v10044225 = v10043605 * v8809991 + v10044166 * v8759431
    v10044315 = v10043412 * v8813837 + (0.0 - jnp.where((v10044256 == 0.0) | ~jnp.isfinite(v10044256), 0.0, jnp.divide(v10044210 * 2.0, jnp.where((v10044256 == 0.0) | ~jnp.isfinite(v10044256), 1.0, v10044256)))) * v8813708 + (v10043605 * v8810043 + v10044183 * v8759431) + (v10043412 * v8813823 + v10044225 * v8813708) * 3.0
    v10044271 = v10043605 * 2.0 * v8813834 + (v10044225 + (0.0 - jnp.where((v10044256 == 0.0) | ~jnp.isfinite(v10044256), 0.0, jnp.divide(v10044210, jnp.where((v10044256 == 0.0) | ~jnp.isfinite(v10044256), 1.0, v10044256))))) * v8813828
    v10044357 = 2.0 * v8813851
    v10044364 = v8813835 * v8813835
    v10044417 = jnp.where(v8813727, (v10044183 * v8813708 + v10043412 * v8810043) * v8813800 + (0.0 - jnp.where((v10044389 == 0.0) | ~jnp.isfinite(v10044389), 0.0, jnp.divide(v10043605 * v8810043 + v10044183 * v8759431 + v10043412, jnp.where((v10044389 == 0.0) | ~jnp.isfinite(v10044389), 1.0, v10044389)))) * v8813803, jnp.where((v8813835 == 0.0) | ~jnp.isfinite(v8813835), 0.0, jnp.divide(v10044315 - jnp.divide(v10044315 * v8813843 + v10044315 * v8813843 - (v10044271 * 2.0 * v8813846 + (v10043412 * v8813845 + (v10044183 + (v10043412 * v8809991 + v10044166 * v8813708) * 2.0) * v8813708) * v8813848), v10044357), jnp.where((v8813835 == 0.0) | ~jnp.isfinite(v8813835), 1.0, v8813835))) - jnp.where((v10044364 == 0.0) | ~jnp.isfinite(v10044364), 0.0, jnp.divide(v10044271 * v8813852, jnp.where((v10044364 == 0.0) | ~jnp.isfinite(v10044364), 1.0, v10044364))))
    v10044420 = v10044417 - v10042928
    v10044441 = 2.0 * v8814784
    v10044442 = jnp.divide(v10044420 * v8814779 + v10044420 * v8814779 + v10044417 * v8814781, v10044441)
    v10044448 = v8814799 * v8814799
    v10044485 = jnp.where(v8814817, v10042928, jnp.where(v8814810, 0.0, jnp.where(v8814788, v10044417 - (v10044420 + v10044442) * 0.5, v10044417 * v8814801 + jnp.where((v10044448 == 0.0) | ~jnp.isfinite(v10044448), 0.0, jnp.divide((v10044442 - v10044420) * i_v9705, jnp.where((v10044448 == 0.0) | ~jnp.isfinite(v10044448), 1.0, v10044448))) * v8813856)))
    v10044719 = v10044485 * 0.5
    v10044582 = v8813708 * v8813708
    v10044494 = v8822061 * v8822061
    v10044488 = v10042928 - v10044485
    v10044507 = v8825021 * v8825021
    v10044531 = v8825035 * v8825035
    v10044533 = jnp.where((v10044531 == 0.0) | ~jnp.isfinite(v10044531), 0.0, jnp.divide((v8825033 + v8825033) * 2.0, jnp.where((v10044531 == 0.0) | ~jnp.isfinite(v10044531), 1.0, v10044531)))
    v10044553 = v10044183 * v8825039 + ((0.0 - jnp.where((v10044494 == 0.0) | ~jnp.isfinite(v10044494), 0.0, jnp.divide(v10044151 * i_v6006 * i_v6880, jnp.where((v10044494 == 0.0) | ~jnp.isfinite(v10044494), 1.0, v10044494)))) * v8825037 + (v10044488 * v8825022 + (0.0 - jnp.where((v10044507 == 0.0) | ~jnp.isfinite(v10044507), 0.0, jnp.divide(v10044176 * i_v7144, jnp.where((v10044507 == 0.0) | ~jnp.isfinite(v10044507), 1.0, v10044507)))) * v8814823) * v10044533 * v8822065) * v8810043
    v10044567 = jnp.where(v8821979, v10044553, v10044183)
    v10044743 = v8825196 * v8825196
    v10044768 = v8840324 * v8840324
    v10044772 = jnp.where((v8840324 == 0.0) | ~jnp.isfinite(v8840324), 0.0, jnp.divide(v10044699 * v8840318 + (v10043412 * v8840317 + (0.0 - (v10044719 * v8840302 + (jnp.where((v8813708 == 0.0) | ~jnp.isfinite(v8813708), 0.0, jnp.divide(v10043605, jnp.where((v8813708 == 0.0) | ~jnp.isfinite(v8813708), 1.0, v8813708))) - jnp.where((v10044582 == 0.0) | ~jnp.isfinite(v10044582), 0.0, jnp.divide(v10043412 * v8759431, jnp.where((v10044582 == 0.0) | ~jnp.isfinite(v10044582), 1.0, v10044582)))) * v8840315)) * v8734229) * v8840293, jnp.where((v8840324 == 0.0) | ~jnp.isfinite(v8840324), 1.0, v8840324))) - jnp.where((v10044768 == 0.0) | ~jnp.isfinite(v10044768), 0.0, jnp.divide((jnp.where((v8825196 == 0.0) | ~jnp.isfinite(v8825196), 0.0, jnp.divide(v10044485, jnp.where((v8825196 == 0.0) | ~jnp.isfinite(v8825196), 1.0, v8825196))) - jnp.where((v10044743 == 0.0) | ~jnp.isfinite(v10044743), 0.0, jnp.divide(v10044567 * v8814821, jnp.where((v10044743 == 0.0) | ~jnp.isfinite(v10044743), 1.0, v10044743)))) * v8840326, jnp.where((v10044768 == 0.0) | ~jnp.isfinite(v10044768), 1.0, v10044768)))
    v10044793 = v8840363 * v8840363
    v10044797 = jnp.where((v8840363 == 0.0) | ~jnp.isfinite(v8840363), 0.0, jnp.divide(v10044772, jnp.where((v8840363 == 0.0) | ~jnp.isfinite(v8840363), 1.0, v8840363))) - jnp.where((v10044793 == 0.0) | ~jnp.isfinite(v10044793), 0.0, jnp.divide((v10044772 * v8745483 + v10043476 * v8840327) * v8840327, jnp.where((v10044793 == 0.0) | ~jnp.isfinite(v10044793), 1.0, v10044793)))
    v10044935 = v10043605 * v8813856 + v10044417 * v8759431
    v10044954 = v8847873 * v8847873
    v10044971 = jnp.where((i_v7353 == 0.0) | ~jnp.isfinite(i_v7353), 0.0, jnp.divide(v10043412 - (jnp.where((v8847873 == 0.0) | ~jnp.isfinite(v8847873), 0.0, jnp.divide(v10043412 * v8847853 + v10044935 * v8813708, jnp.where((v8847873 == 0.0) | ~jnp.isfinite(v8847873), 1.0, v8847873))) - jnp.where((v10044954 == 0.0) | ~jnp.isfinite(v10044954), 0.0, jnp.divide((v10043412 + v10044935) * v8847872, jnp.where((v10044954 == 0.0) | ~jnp.isfinite(v10044954), 1.0, v10044954)))), jnp.where((i_v7353 == 0.0) | ~jnp.isfinite(i_v7353), 1.0, i_v7353)))
    v10044974 = v10042967 * i_v6670
    v10045000 = 0.0 - jnp.where((v8850959 * v8850959 == 0.0) | ~jnp.isfinite(v8850959 * v8850959), 0.0, jnp.divide(v10044974, jnp.where((v8850959 * v8850959 == 0.0) | ~jnp.isfinite(v8850959 * v8850959), 1.0, v8850959 * v8850959)))
    v10044986 = v10044974 * 20.0 * v8850965 + (0.0 - jnp.where((v8850964 * v8850964 == 0.0) | ~jnp.isfinite(v8850964 * v8850964), 0.0, jnp.divide(v10044974, jnp.where((v8850964 * v8850964 == 0.0) | ~jnp.isfinite(v8850964 * v8850964), 1.0, v8850964 * v8850964)))) * v8850967
    v10044833 = (0.0 - jnp.where((v10044743 == 0.0) | ~jnp.isfinite(v10044743), 0.0, jnp.divide(v10044567 * i_v6688, jnp.where((v10044743 == 0.0) | ~jnp.isfinite(v10044743), 1.0, v10044743)))) * v8734229 + v10043412 * v8843984
    v10044848 = v8844003 + (0.0 - jnp.where((v8844002 * v8844002 == 0.0) | ~jnp.isfinite(v8844002 * v8844002), 0.0, jnp.divide(20.0, jnp.where((v8844002 * v8844002 == 0.0) | ~jnp.isfinite(v8844002 * v8844002), 1.0, v8844002 * v8844002)))) * v8844004
    v10044856 = jnp.where(v8843996, v10044833, v10044833 * v10044848)
    v10045061 = v8850981 * v8850981
    v10044814 = 0.0 - jnp.where((v8840989 * v8840989 == 0.0) | ~jnp.isfinite(v8840989 * v8840989), 0.0, jnp.divide(0.0 - jnp.where((v10044582 == 0.0) | ~jnp.isfinite(v10044582), 0.0, jnp.divide(v8840983, jnp.where((v10044582 == 0.0) | ~jnp.isfinite(v10044582), 1.0, v10044582))), jnp.where((v8840989 * v8840989 == 0.0) | ~jnp.isfinite(v8840989 * v8840989), 1.0, v8840989 * v8840989)))
    v10044818 = jnp.where(v8840975, 0.0, v10043412 * v10044814)
    v10045086 = v8855415 * v8855415
    v10044634 = i_v9520 * i_v9520
    v10044650 = v8825231 * v8825231
    v10044654 = jnp.where((v8825231 == 0.0) | ~jnp.isfinite(v8825231), 0.0, jnp.divide(v10044567 + v10044417 + ((v10044166 * v8734229 + v10043412 * v8809991) * 2.0 * v8825137 + (0.0 - (jnp.where((v8813708 == 0.0) | ~jnp.isfinite(v8813708), 0.0, jnp.divide(v10043605 * 0.5 * v8813856 + v10044417 * v8825090, jnp.where((v8813708 == 0.0) | ~jnp.isfinite(v8813708), 1.0, v8813708))) - jnp.where((v10044582 == 0.0) | ~jnp.isfinite(v10044582), 0.0, jnp.divide(v10043412 * v8825110, jnp.where((v10044582 == 0.0) | ~jnp.isfinite(v10044582), 1.0, v10044582))))) * v8825199), jnp.where((v8825231 == 0.0) | ~jnp.isfinite(v8825231), 1.0, v8825231))) - jnp.where((v10044650 == 0.0) | ~jnp.isfinite(v10044650), 0.0, jnp.divide((0.0 - jnp.where((v10044634 == 0.0) | ~jnp.isfinite(v10044634), 0.0, jnp.divide(v10044210 * 2.0, jnp.where((v10044634 == 0.0) | ~jnp.isfinite(v10044634), 1.0, v10044634))) + (v10044166 * v8759431 + v10043605 * v8809991)) * v8825201, jnp.where((v10044650 == 0.0) | ~jnp.isfinite(v10044650), 1.0, v10044650)))
    v10044871 = v8844897 * v8844897
    v10044915 = jnp.where((v8844961 == 0.0) | ~jnp.isfinite(v8844961), 0.0, jnp.divide(((v10044818 * v8844931 + v10044856 * v8840993) * v8844864 + (v10043476 * v8840364 + v10044797 * v8745483) * v8844933) * v8844918 + (jnp.where((v8844897 == 0.0) | ~jnp.isfinite(v8844897), 0.0, jnp.divide(v10044417, jnp.where((v8844897 == 0.0) | ~jnp.isfinite(v8844897), 1.0, v8844897))) - jnp.where((v10044871 == 0.0) | ~jnp.isfinite(v10044871), 0.0, jnp.divide(jnp.where(v8821979, jnp.where((i_v6006 == 0.0) | ~jnp.isfinite(i_v6006), 0.0, jnp.divide(v10044553, jnp.where((i_v6006 == 0.0) | ~jnp.isfinite(i_v6006), 1.0, i_v6006))), v10044176) * v8813856, jnp.where((v10044871 == 0.0) | ~jnp.isfinite(v10044871), 1.0, v10044871)))) * v8844934, jnp.where((v8844961 == 0.0) | ~jnp.isfinite(v8844961), 1.0, v8844961)))
    v10045111 = v8825232 * v8825232
    v10045127 = v8859574 * v8859574
    v10045145 = ((v10044797 * v8859396 + (jnp.where((v8850981 == 0.0) | ~jnp.isfinite(v8850981), 0.0, jnp.divide(v10044488, jnp.where((v8850981 == 0.0) | ~jnp.isfinite(v8850981), 1.0, v8850981))) - jnp.where((v10045061 == 0.0) | ~jnp.isfinite(v10045061), 0.0, jnp.divide((jnp.where(v8850957, v10044971 * v8850960 + v10042955 * v10045000 * v8847878, v10044971 * v8850968 + v10042955 * v10044986 * v8847878) * v8844931 + v10044856 * v8850971) * v8814823, jnp.where((v10045061 == 0.0) | ~jnp.isfinite(v10045061), 1.0, v10045061)))) * v8840364) * v8859473 + (jnp.where((v8855415 == 0.0) | ~jnp.isfinite(v8855415), 0.0, jnp.divide(v10044488, jnp.where((v8855415 == 0.0) | ~jnp.isfinite(v8855415), 1.0, v8855415))) - jnp.where((v10045086 == 0.0) | ~jnp.isfinite(v10045086), 0.0, jnp.divide(jnp.where(v8854726, jnp.where((i_v6586 == 0.0) | ~jnp.isfinite(i_v6586), 0.0, jnp.divide(jnp.where(v8854087, 0.0, v10042928 * i_v6592 * v8854089) * i_v9724, jnp.where((i_v6586 == 0.0) | ~jnp.isfinite(i_v6586), 1.0, i_v6586))) * v8840993 + v10044818 * v8855389, 0.0) * v8814823, jnp.where((v10045086 == 0.0) | ~jnp.isfinite(v10045086), 1.0, v10045086)))) * v8859447) * v8859595 + (jnp.where((v8859574 == 0.0) | ~jnp.isfinite(v8859574), 0.0, jnp.divide(jnp.where((v8859546 == 0.0) | ~jnp.isfinite(v8859546), 0.0, jnp.divide(jnp.where((v8825232 == 0.0) | ~jnp.isfinite(v8825232), 0.0, jnp.divide(v10044654 + jnp.where(v8844831, v10044915 * v8814823 + v10044488 * v8844962, 0.0), jnp.where((v8825232 == 0.0) | ~jnp.isfinite(v8825232), 1.0, v8825232))) - jnp.where((v10045111 == 0.0) | ~jnp.isfinite(v10045111), 0.0, jnp.divide(v10044654 * v8851029, jnp.where((v10045111 == 0.0) | ~jnp.isfinite(v10045111), 1.0, v10045111))), jnp.where((v8859546 == 0.0) | ~jnp.isfinite(v8859546), 1.0, v8859546))), jnp.where((v8859574 == 0.0) | ~jnp.isfinite(v8859574), 1.0, v8859574))) - jnp.where((v10045127 == 0.0) | ~jnp.isfinite(v10045127), 0.0, jnp.divide(jnp.where(v8844831, v10044915, 0.0) * v8859547, jnp.where((v10045127 == 0.0) | ~jnp.isfinite(v10045127), 1.0, v10045127)))) * v8859477
    v10045051 = jnp.where((i_v6682 == 0.0) | ~jnp.isfinite(i_v6682), 0.0, jnp.divide((0.0 - jnp.where((v8814823 * v8814823 == 0.0) | ~jnp.isfinite(v8814823 * v8814823), 0.0, jnp.divide(v8859326, jnp.where((v8814823 * v8814823 == 0.0) | ~jnp.isfinite(v8814823 * v8814823), 1.0, v8814823 * v8814823)))) * v8859344 * i_v6006, jnp.where((i_v6682 == 0.0) | ~jnp.isfinite(i_v6682), 1.0, i_v6682)))
    v10045190 = v8859351 * v8859351
    v10045208 = v10045145 * v8868904 + (jnp.where((v8859351 == 0.0) | ~jnp.isfinite(v8859351), 0.0, jnp.divide(v10044488, jnp.where((v8859351 == 0.0) | ~jnp.isfinite(v8859351), 1.0, v8859351))) - jnp.where((v10045190 == 0.0) | ~jnp.isfinite(v10045190), 0.0, jnp.divide(v10044488 * v10045051 * v8814823, jnp.where((v10045190 == 0.0) | ~jnp.isfinite(v10045190), 1.0, v10045190)))) * v8859598
    v10045217 = v10045208 * v8814821 + v10044485 * v8868910
    v10046245 = jnp.where(v9123179, v10045217 * nf, v10045217)
    v10048460 = jnp.where(v9336400, -v10046245, v10046245) * -1.0
    v10046022 = jnp.divide(v9061923 + v9061923, 2.0 * v9061926)
    v10046045 = (v9061926 + v10046022 * v8471984) * i_v7199
    v10046038 = (v10046022 * i_v9823 - (v10046022 * i_v9824 * v9061926 + v10046022 * v9074893)) * i_v7202 * v9074902
    v10046050 = v10042191 * v10046045 * v9074902 + v10042191 * v10046038 * v9074915
    v10048464 = jnp.where(v9123179, v10046050 * nf, v10046050) * -1.0
    v10042968 = v10042955 * v10042967
    v10045687 = 0.0 - v10043319 + v10042968
    v10045708 = jnp.divide(v8984545 + v8984545, 2.0 * v8984554)
    v10045699 = jnp.divide(v8984545 + v8984545, 2.0 * v8984561)
    v10045718 = (v10045687 + jnp.where(v8984547, v10045687 * v10045708, v10045687 * v10045699)) * 0.5
    v10045758 = -(v10043319 - (0.0 - v10045718) - v10042968 - v10043412) + v10043412
    v10045822 = (i_v9815 - (v9031721 + v9031721)) * i_v7207 * v9031730
    v10045848 = v9038568 * v9038568
    v10045856 = jnp.where((v9038568 == 0.0) | ~jnp.isfinite(v9038568), 0.0, jnp.divide(0.0 - jnp.where((v10045848 == 0.0) | ~jnp.isfinite(v10045848), 0.0, jnp.divide(i_v9817, jnp.where((v10045848 == 0.0) | ~jnp.isfinite(v10045848), 1.0, v10045848))), jnp.where((v9038568 == 0.0) | ~jnp.isfinite(v9038568), 1.0, v9038568))) - jnp.where((v10045848 == 0.0) | ~jnp.isfinite(v10045848), 0.0, jnp.divide(v9038569, jnp.where((v10045848 == 0.0) | ~jnp.isfinite(v10045848), 1.0, v10045848)))
    v10045893 = -(v10043412 * v10045856 * v9038767 + (0.0 - (jnp.where((v9038568 == 0.0) | ~jnp.isfinite(v9038568), 0.0, jnp.divide(v10044719, jnp.where((v9038568 == 0.0) | ~jnp.isfinite(v9038568), 1.0, v9038568))) - jnp.where((v10045848 == 0.0) | ~jnp.isfinite(v10045848), 0.0, jnp.divide(v10043412 * v8840315, jnp.where((v10045848 == 0.0) | ~jnp.isfinite(v10045848), 1.0, v10045848))))) * v9038570) * v8814821 + v10044485 * v9038770
    v10045931 = v9038775 * v9038775
    v10045162 = i_v9736 * v8868773 + (0.0 - jnp.where((v8814823 * v8814823 == 0.0) | ~jnp.isfinite(v8814823 * v8814823), 0.0, jnp.divide(v8868768, jnp.where((v8814823 * v8814823 == 0.0) | ~jnp.isfinite(v8814823 * v8814823), 1.0, v8814823 * v8814823)))) * v8868773 * v8868772
    v10045180 = jnp.where(v8868766, v10044488 * v10045162, v10044488 * v8868779) * v8868864 + (v10045145 * v8814821 + v10044485 * v8859598) * v8868865
    v10045555 = jnp.where((i_v9766 == 0.0) | ~jnp.isfinite(i_v9766), 0.0, jnp.divide(-1.0, jnp.where((i_v9766 == 0.0) | ~jnp.isfinite(i_v9766), 1.0, i_v9766)))
    v10045559 = v8925988 * v8925988
    v10045593 = -1.0 * v8935877 + (v10042191 * v8471980 + v10042191 * v8471980) * v8935878
    v10045599 = v8935889 * v8935889
    v10045610 = jnp.where(v8935353, v10045555 * v8935854 * v8935859 + jnp.where((v10045559 == 0.0) | ~jnp.isfinite(v10045559), 0.0, jnp.divide(v10045555 * i_v6748, jnp.where((v10045559 == 0.0) | ~jnp.isfinite(v10045559), 1.0, v10045559))) * v8935859 * v8935856, v10045555 * v8935868) * v8935890 + (jnp.where((v8935889 == 0.0) | ~jnp.isfinite(v8935889), 0.0, jnp.divide(v10045593, jnp.where((v8935889 == 0.0) | ~jnp.isfinite(v8935889), 1.0, v8935889))) - jnp.where((v10045599 == 0.0) | ~jnp.isfinite(v10045599), 0.0, jnp.divide(v10045593 * v8935879, jnp.where((v10045599 == 0.0) | ~jnp.isfinite(v10045599), 1.0, v10045599)))) * v8935891
    v10048469 = (jnp.where(v9123179, v10045180 * nf, v10045180) + jnp.where(v9123179, v10045610 * nf, v10045610)) * -1.0
    v10042628 = jnp.where((i_v9593 == 0.0) | ~jnp.isfinite(i_v9593), 0.0, jnp.divide(-1.0, jnp.where((i_v9593 == 0.0) | ~jnp.isfinite(i_v9593), 1.0, i_v9593))) * 1000.0 * 583461742500000.0
    v10042604 = (jnp.where((i_v9593 == 0.0) | ~jnp.isfinite(i_v9593), 0.0, jnp.divide(-1.0, jnp.where((i_v9593 == 0.0) | ~jnp.isfinite(i_v9593), 1.0, i_v9593))) * 10.0 * v8608110 + (0.0 - jnp.where((v8608056 * v8608056 == 0.0) | ~jnp.isfinite(v8608056 * v8608056), 0.0, jnp.divide(-1.0, jnp.where((v8608056 * v8608056 == 0.0) | ~jnp.isfinite(v8608056 * v8608056), 1.0, v8608056 * v8608056)))) * v8608114) * 583461742500000.0
    v10042750 = jnp.where((i_v9591 == 0.0) | ~jnp.isfinite(i_v9591), 0.0, jnp.divide(-1.0, jnp.where((i_v9591 == 0.0) | ~jnp.isfinite(i_v9591), 1.0, i_v9591))) * 1000.0 * 583461742500000.0
    v10042726 = (jnp.where((i_v9591 == 0.0) | ~jnp.isfinite(i_v9591), 0.0, jnp.divide(-1.0, jnp.where((i_v9591 == 0.0) | ~jnp.isfinite(i_v9591), 1.0, i_v9591))) * 10.0 * v8609259 + (0.0 - jnp.where((v8609175 * v8609175 == 0.0) | ~jnp.isfinite(v8609175 * v8609175), 0.0, jnp.divide(-1.0, jnp.where((v8609175 * v8609175 == 0.0) | ~jnp.isfinite(v8609175 * v8609175), 1.0, v8609175 * v8609175)))) * v8609263) * 583461742500000.0
    v10042872 = jnp.where((i_v9592 == 0.0) | ~jnp.isfinite(i_v9592), 0.0, jnp.divide(-1.0, jnp.where((i_v9592 == 0.0) | ~jnp.isfinite(i_v9592), 1.0, i_v9592))) * 1000.0 * 583461742500000.0
    v10042848 = (jnp.where((i_v9592 == 0.0) | ~jnp.isfinite(i_v9592), 0.0, jnp.divide(-1.0, jnp.where((i_v9592 == 0.0) | ~jnp.isfinite(i_v9592), 1.0, i_v9592))) * 10.0 * v8610516 + (0.0 - jnp.where((v8610402 * v8610402 == 0.0) | ~jnp.isfinite(v8610402 * v8610402), 0.0, jnp.divide(-1.0, jnp.where((v8610402 * v8610402 == 0.0) | ~jnp.isfinite(v8610402 * v8610402), 1.0, v8610402 * v8610402)))) * v8610520) * 583461742500000.0
    v10048546 = (v10042191 * _ckt_gmin - (jnp.where(v8608058, v10042191 * v10042628, v10042191 * v10042604) * i_v8982 + jnp.where(v8609177, v10042191 * v10042750, v10042191 * v10042726) * i_v8986 + jnp.where(v8610404, v10042191 * v10042872, v10042191 * v10042848) * i_v8992)) * -1.0
    v10048760 = v10048460 - v10048464 + v10048469 - v10048515 - v10048546
    v10048190 = (1.0 - jnp.divide(v9301638 + v9301638, 2.0 * v9301642)) * 0.5
    v10048201 = v9306440 - (v10048190 + jnp.divide(0.0 - jnp.where((i_v9274 == 0.0) | ~jnp.isfinite(i_v9274), 0.0, jnp.divide(v10048190 * 4.0, jnp.where((i_v9274 == 0.0) | ~jnp.isfinite(i_v9274), 1.0, i_v9274))), 2.0 * v9306438) * v9306442) * i_v9918
    v10048234 = jnp.where(v9299631, v10042191 * i_v9513, v10042191 * v10048201)
    v10047847 = (v10043319 - jnp.where(v9145086, v10042929, v10042968)) * i_v9853
    v10048556 = ((v10043319 - v10042955 * v10043125) * v9151211 * 0.4 - jnp.where(v9123179, v10048234 * nf, v10048234)) * -1.0
    v10048020 = v9296789 + v9296786 * 0.5 * v8471980
    v10048578 = v10042191 * v10048020 * -1.0
    v10048765 = v10048556 - v10048578
    v10043251 = -1.0 * 2.0
    v10043262 = jnp.where((v8725555 == 0.0) | ~jnp.isfinite(v8725555), 0.0, jnp.divide(v10043251, jnp.where((v8725555 == 0.0) | ~jnp.isfinite(v8725555), 1.0, v8725555))) - jnp.where((v8725555 * v8725555 == 0.0) | ~jnp.isfinite(v8725555 * v8725555), 0.0, jnp.divide(jnp.divide(jnp.where((v8725536 == 0.0) | ~jnp.isfinite(v8725536), 0.0, jnp.divide(v10043251, jnp.where((v8725536 == 0.0) | ~jnp.isfinite(v8725536), 1.0, v8725536))), 2.0 * v8725553) * v8725550, jnp.where((v8725555 * v8725555 == 0.0) | ~jnp.isfinite(v8725555 * v8725555), 1.0, v8725555 * v8725555)))
    v10043269 = 0.0 - jnp.where((v8725536 == 0.0) | ~jnp.isfinite(v8725536), 0.0, jnp.divide(v10043262 * 0.5 * v8725556 + v10043262 * v8725557, jnp.where((v8725536 == 0.0) | ~jnp.isfinite(v8725536), 1.0, v8725536)))
    v10043281 = jnp.where(v8725473, -1.0 - (0.0 - (v10043269 + jnp.divide(v10043269 * v8725561 + v10043269 * v8725561, 2.0 * v8725564)) * 0.5), -1.0)
    v10043318 = jnp.where(v8725690, -1.0 * v10043314, -1.0)
    v10043320 = jnp.where(v8725855, v10043281, v10043318)
    v10043416 = jnp.where((v8734227 == 0.0) | ~jnp.isfinite(v8734227), 0.0, jnp.divide(v10043320 * i_v7210, jnp.where((v8734227 == 0.0) | ~jnp.isfinite(v8734227), 1.0, v8734227))) - jnp.where((v10043408 == 0.0) | ~jnp.isfinite(v10043408), 0.0, jnp.divide(jnp.where(v8731479, 0.0, jnp.where(v8734190, 0.0, jnp.where((v8726063 == 0.0) | ~jnp.isfinite(v8726063), 0.0, jnp.divide(0.0 - v10043320 * v8731462, jnp.where((v8726063 == 0.0) | ~jnp.isfinite(v8726063), 1.0, v8726063))) * v8734204 * v8734207 * v8701987)) * v8728744, jnp.where((v10043408 == 0.0) | ~jnp.isfinite(v10043408), 1.0, v10043408)))
    v10044036 = jnp.where((4.23e-09 == 0.0) | ~jnp.isfinite(4.23e-09), 0.0, jnp.divide(v10043416, jnp.where((4.23e-09 == 0.0) | ~jnp.isfinite(4.23e-09), 1.0, 4.23e-09)))
    v10044065 = (0.0 - jnp.where((v10044054 == 0.0) | ~jnp.isfinite(v10044054), 0.0, jnp.divide(v10043416, jnp.where((v10044054 == 0.0) | ~jnp.isfinite(v10044054), 1.0, v10044054)))) * 4.23e-09
    v10044109 = v10044036 * v8781625 + v10044036 * i_v7037 * v8775822 + (v10044065 * i_v7038 * v8775831 + v10044065 * v8778727) * v8696278 * v8696278
    v10044154 = 0.0 - jnp.where((v10044148 == 0.0) | ~jnp.isfinite(v10044148), 0.0, jnp.divide(jnp.where(v8799108, v10044109, v10044109 * v8799116 + (0.0 - jnp.where((v10044120 == 0.0) | ~jnp.isfinite(v10044120), 0.0, jnp.divide(v10044109 * 10.0, jnp.where((v10044120 == 0.0) | ~jnp.isfinite(v10044120), 1.0, v10044120)))) * v8799117) * i_v7075, jnp.where((v10044148 == 0.0) | ~jnp.isfinite(v10044148), 1.0, v10044148)))
    v10043423 = v10043416 * i_v6700
    v10043450 = jnp.where(v8740230, v10043423 * v10043445, v10043423 * -2.0)
    v10044702 = v10044154 * v8840286 + jnp.where((i_v6006 == 0.0) | ~jnp.isfinite(i_v6006), 0.0, jnp.divide(v10043416 * v10044680 * v8740245 + v10043450 * v8840233, jnp.where((i_v6006 == 0.0) | ~jnp.isfinite(i_v6006), 1.0, i_v6006))) * v8809061
    v10044179 = 0.0 - jnp.where((v10044173 == 0.0) | ~jnp.isfinite(v10044173), 0.0, jnp.divide(v10044154 * v8809993, jnp.where((v10044173 == 0.0) | ~jnp.isfinite(v10044173), 1.0, v10044173)))
    v10044184 = v10044179 * i_v6006
    v10043525 = v10043416 * v8756385
    v10043606 = jnp.where(v8756409, (0.0 - v10043525) * v8756413 + (0.0 - jnp.where((v10043554 == 0.0) | ~jnp.isfinite(v10043554), 0.0, jnp.divide(0.0 - v10043525 * 20.0, jnp.where((v10043554 == 0.0) | ~jnp.isfinite(v10043554), 1.0, v10043554)))) * v8756414, v10043525) * v8759430
    v10044211 = v10043416 * v10044209
    v10043477 = v10043416 * v10043459 * v10043474
    v10044169 = v10043450 * i_v7043 * i_v5750 * v8745483 + v10043477 * v8809942
    v10044228 = v10043606 * v8809991 + v10044169 * v8759431
    v10044316 = v10043416 * v8813837 + (0.0 - jnp.where((v10044256 == 0.0) | ~jnp.isfinite(v10044256), 0.0, jnp.divide(v10044211 * 2.0, jnp.where((v10044256 == 0.0) | ~jnp.isfinite(v10044256), 1.0, v10044256)))) * v8813708 + (v10043606 * v8810043 + v10044184 * v8759431) + (v10043416 * v8813823 + v10044228 * v8813708) * 3.0
    v10044274 = v10043606 * 2.0 * v8813834 + (v10044228 + (0.0 - jnp.where((v10044256 == 0.0) | ~jnp.isfinite(v10044256), 0.0, jnp.divide(v10044211, jnp.where((v10044256 == 0.0) | ~jnp.isfinite(v10044256), 1.0, v10044256))))) * v8813828
    v10044418 = jnp.where(v8813727, (v10044184 * v8813708 + v10043416 * v8810043) * v8813800 + (0.0 - jnp.where((v10044389 == 0.0) | ~jnp.isfinite(v10044389), 0.0, jnp.divide(v10043606 * v8810043 + v10044184 * v8759431 + v10043416, jnp.where((v10044389 == 0.0) | ~jnp.isfinite(v10044389), 1.0, v10044389)))) * v8813803, jnp.where((v8813835 == 0.0) | ~jnp.isfinite(v8813835), 0.0, jnp.divide(v10044316 - jnp.divide(v10044316 * v8813843 + v10044316 * v8813843 - (v10044274 * 2.0 * v8813846 + (v10043416 * v8813845 + (v10044184 + (v10043416 * v8809991 + v10044169 * v8813708) * 2.0) * v8813708) * v8813848), v10044357), jnp.where((v8813835 == 0.0) | ~jnp.isfinite(v8813835), 1.0, v8813835))) - jnp.where((v10044364 == 0.0) | ~jnp.isfinite(v10044364), 0.0, jnp.divide(v10044274 * v8813852, jnp.where((v10044364 == 0.0) | ~jnp.isfinite(v10044364), 1.0, v10044364))))
    v10044443 = jnp.divide(v10044418 * v8814779 + v10044418 * v8814779 + v10044418 * v8814781, v10044441)
    v10044486 = jnp.where(v8814817, 0.0, jnp.where(v8814810, 0.0, jnp.where(v8814788, v10044418 - (v10044418 + v10044443) * 0.5, v10044418 * v8814801 + jnp.where((v10044448 == 0.0) | ~jnp.isfinite(v10044448), 0.0, jnp.divide((v10044443 - v10044418) * i_v9705, jnp.where((v10044448 == 0.0) | ~jnp.isfinite(v10044448), 1.0, v10044448))) * v8813856)))
    v10044489 = 0.0 - v10044486
    v10044556 = v10044184 * v8825039 + ((0.0 - jnp.where((v10044494 == 0.0) | ~jnp.isfinite(v10044494), 0.0, jnp.divide(v10044154 * i_v6006 * i_v6880, jnp.where((v10044494 == 0.0) | ~jnp.isfinite(v10044494), 1.0, v10044494)))) * v8825037 + (v10044489 * v8825022 + (0.0 - jnp.where((v10044507 == 0.0) | ~jnp.isfinite(v10044507), 0.0, jnp.divide(v10044179 * i_v7144, jnp.where((v10044507 == 0.0) | ~jnp.isfinite(v10044507), 1.0, v10044507)))) * v8814823) * v10044533 * v8822065) * v8810043
    v10044568 = jnp.where(v8821979, v10044556, v10044184)
    v10044776 = jnp.where((v8840324 == 0.0) | ~jnp.isfinite(v8840324), 0.0, jnp.divide(v10044702 * v8840318 + (v10043416 * v8840317 + (0.0 - (v10044486 * 0.5 * v8840302 + (jnp.where((v8813708 == 0.0) | ~jnp.isfinite(v8813708), 0.0, jnp.divide(v10043606, jnp.where((v8813708 == 0.0) | ~jnp.isfinite(v8813708), 1.0, v8813708))) - jnp.where((v10044582 == 0.0) | ~jnp.isfinite(v10044582), 0.0, jnp.divide(v10043416 * v8759431, jnp.where((v10044582 == 0.0) | ~jnp.isfinite(v10044582), 1.0, v10044582)))) * v8840315)) * v8734229) * v8840293, jnp.where((v8840324 == 0.0) | ~jnp.isfinite(v8840324), 1.0, v8840324))) - jnp.where((v10044768 == 0.0) | ~jnp.isfinite(v10044768), 0.0, jnp.divide((jnp.where((v8825196 == 0.0) | ~jnp.isfinite(v8825196), 0.0, jnp.divide(v10044486, jnp.where((v8825196 == 0.0) | ~jnp.isfinite(v8825196), 1.0, v8825196))) - jnp.where((v10044743 == 0.0) | ~jnp.isfinite(v10044743), 0.0, jnp.divide(v10044568 * v8814821, jnp.where((v10044743 == 0.0) | ~jnp.isfinite(v10044743), 1.0, v10044743)))) * v8840326, jnp.where((v10044768 == 0.0) | ~jnp.isfinite(v10044768), 1.0, v10044768)))
    v10044801 = jnp.where((v8840363 == 0.0) | ~jnp.isfinite(v8840363), 0.0, jnp.divide(v10044776, jnp.where((v8840363 == 0.0) | ~jnp.isfinite(v8840363), 1.0, v8840363))) - jnp.where((v10044793 == 0.0) | ~jnp.isfinite(v10044793), 0.0, jnp.divide((v10044776 * v8745483 + v10043477 * v8840327) * v8840327, jnp.where((v10044793 == 0.0) | ~jnp.isfinite(v10044793), 1.0, v10044793)))
    v10044938 = v10043606 * v8813856 + v10044418 * v8759431
    v10044972 = jnp.where((i_v7353 == 0.0) | ~jnp.isfinite(i_v7353), 0.0, jnp.divide(v10043416 - (jnp.where((v8847873 == 0.0) | ~jnp.isfinite(v8847873), 0.0, jnp.divide(v10043416 * v8847853 + v10044938 * v8813708, jnp.where((v8847873 == 0.0) | ~jnp.isfinite(v8847873), 1.0, v8847873))) - jnp.where((v10044954 == 0.0) | ~jnp.isfinite(v10044954), 0.0, jnp.divide((v10043416 + v10044938) * v8847872, jnp.where((v10044954 == 0.0) | ~jnp.isfinite(v10044954), 1.0, v10044954)))), jnp.where((i_v7353 == 0.0) | ~jnp.isfinite(i_v7353), 1.0, i_v7353)))
    v10044836 = (0.0 - jnp.where((v10044743 == 0.0) | ~jnp.isfinite(v10044743), 0.0, jnp.divide(v10044568 * i_v6688, jnp.where((v10044743 == 0.0) | ~jnp.isfinite(v10044743), 1.0, v10044743)))) * v8734229 + v10043416 * v8843984
    v10044857 = jnp.where(v8843996, v10044836, v10044836 * v10044848)
    v10044819 = jnp.where(v8840975, 0.0, v10043416 * v10044814)
    v10044658 = jnp.where((v8825231 == 0.0) | ~jnp.isfinite(v8825231), 0.0, jnp.divide(v10044568 + v10044418 + ((v10044169 * v8734229 + v10043416 * v8809991) * 2.0 * v8825137 + (0.0 - (jnp.where((v8813708 == 0.0) | ~jnp.isfinite(v8813708), 0.0, jnp.divide(v10043606 * 0.5 * v8813856 + v10044418 * v8825090, jnp.where((v8813708 == 0.0) | ~jnp.isfinite(v8813708), 1.0, v8813708))) - jnp.where((v10044582 == 0.0) | ~jnp.isfinite(v10044582), 0.0, jnp.divide(v10043416 * v8825110, jnp.where((v10044582 == 0.0) | ~jnp.isfinite(v10044582), 1.0, v10044582))))) * v8825199), jnp.where((v8825231 == 0.0) | ~jnp.isfinite(v8825231), 1.0, v8825231))) - jnp.where((v10044650 == 0.0) | ~jnp.isfinite(v10044650), 0.0, jnp.divide((0.0 - jnp.where((v10044634 == 0.0) | ~jnp.isfinite(v10044634), 0.0, jnp.divide(v10044211 * 2.0, jnp.where((v10044634 == 0.0) | ~jnp.isfinite(v10044634), 1.0, v10044634))) + (v10044169 * v8759431 + v10043606 * v8809991)) * v8825201, jnp.where((v10044650 == 0.0) | ~jnp.isfinite(v10044650), 1.0, v10044650)))
    v10044916 = jnp.where((v8844961 == 0.0) | ~jnp.isfinite(v8844961), 0.0, jnp.divide(((v10044819 * v8844931 + v10044857 * v8840993) * v8844864 + (v10043477 * v8840364 + v10044801 * v8745483) * v8844933) * v8844918 + (jnp.where((v8844897 == 0.0) | ~jnp.isfinite(v8844897), 0.0, jnp.divide(v10044418, jnp.where((v8844897 == 0.0) | ~jnp.isfinite(v8844897), 1.0, v8844897))) - jnp.where((v10044871 == 0.0) | ~jnp.isfinite(v10044871), 0.0, jnp.divide(jnp.where(v8821979, jnp.where((i_v6006 == 0.0) | ~jnp.isfinite(i_v6006), 0.0, jnp.divide(v10044556, jnp.where((i_v6006 == 0.0) | ~jnp.isfinite(i_v6006), 1.0, i_v6006))), v10044179) * v8813856, jnp.where((v10044871 == 0.0) | ~jnp.isfinite(v10044871), 1.0, v10044871)))) * v8844934, jnp.where((v8844961 == 0.0) | ~jnp.isfinite(v8844961), 1.0, v8844961)))
    v10045148 = ((v10044801 * v8859396 + (jnp.where((v8850981 == 0.0) | ~jnp.isfinite(v8850981), 0.0, jnp.divide(v10044489, jnp.where((v8850981 == 0.0) | ~jnp.isfinite(v8850981), 1.0, v8850981))) - jnp.where((v10045061 == 0.0) | ~jnp.isfinite(v10045061), 0.0, jnp.divide((jnp.where(v8850957, v10044972 * v8850960, v10044972 * v8850968) * v8844931 + v10044857 * v8850971) * v8814823, jnp.where((v10045061 == 0.0) | ~jnp.isfinite(v10045061), 1.0, v10045061)))) * v8840364) * v8859473 + (jnp.where((v8855415 == 0.0) | ~jnp.isfinite(v8855415), 0.0, jnp.divide(v10044489, jnp.where((v8855415 == 0.0) | ~jnp.isfinite(v8855415), 1.0, v8855415))) - jnp.where((v10045086 == 0.0) | ~jnp.isfinite(v10045086), 0.0, jnp.divide(jnp.where(v8854726, v10044819 * v8855389, 0.0) * v8814823, jnp.where((v10045086 == 0.0) | ~jnp.isfinite(v10045086), 1.0, v10045086)))) * v8859447) * v8859595 + (jnp.where((v8859574 == 0.0) | ~jnp.isfinite(v8859574), 0.0, jnp.divide(jnp.where((v8859546 == 0.0) | ~jnp.isfinite(v8859546), 0.0, jnp.divide(jnp.where((v8825232 == 0.0) | ~jnp.isfinite(v8825232), 0.0, jnp.divide(v10044658 + jnp.where(v8844831, v10044916 * v8814823 + v10044489 * v8844962, 0.0), jnp.where((v8825232 == 0.0) | ~jnp.isfinite(v8825232), 1.0, v8825232))) - jnp.where((v10045111 == 0.0) | ~jnp.isfinite(v10045111), 0.0, jnp.divide(v10044658 * v8851029, jnp.where((v10045111 == 0.0) | ~jnp.isfinite(v10045111), 1.0, v10045111))), jnp.where((v8859546 == 0.0) | ~jnp.isfinite(v8859546), 1.0, v8859546))), jnp.where((v8859574 == 0.0) | ~jnp.isfinite(v8859574), 1.0, v8859574))) - jnp.where((v10045127 == 0.0) | ~jnp.isfinite(v10045127), 0.0, jnp.divide(jnp.where(v8844831, v10044916, 0.0) * v8859547, jnp.where((v10045127 == 0.0) | ~jnp.isfinite(v10045127), 1.0, v10045127)))) * v8859477
    v10045211 = v10045148 * v8868904 + (jnp.where((v8859351 == 0.0) | ~jnp.isfinite(v8859351), 0.0, jnp.divide(v10044489, jnp.where((v8859351 == 0.0) | ~jnp.isfinite(v8859351), 1.0, v8859351))) - jnp.where((v10045190 == 0.0) | ~jnp.isfinite(v10045190), 0.0, jnp.divide(v10044489 * v10045051 * v8814823, jnp.where((v10045190 == 0.0) | ~jnp.isfinite(v10045190), 1.0, v10045190)))) * v8859598
    v10045220 = v10045211 * v8814821 + v10044486 * v8868910
    v10046246 = jnp.where(v9123179, v10045220 * nf, v10045220)
    v10048461 = jnp.where(v9336400, -v10046246, v10046246) * -1.0
    v10046053 = -1.0 * v10046045 * v9074902 + -1.0 * v10046038 * v9074915
    v10048465 = jnp.where(v9123179, v10046053 * nf, v10046053) * -1.0
    v10045686 = 0.0 - v10043320
    v10045719 = (v10045686 + jnp.where(v8984547, v10045686 * v10045708, v10045686 * v10045699)) * 0.5
    v10045759 = -(v10043320 - (0.0 - v10045719) - v10043416) + v10043416
    v10045183 = jnp.where(v8868766, v10044489 * v10045162, v10044489 * v8868779) * v8868864 + (v10045148 * v8814821 + v10044486 * v8859598) * v8868865
    v10045556 = jnp.where((i_v9766 == 0.0) | ~jnp.isfinite(i_v9766), 0.0, jnp.divide(0.0 - v10043281, jnp.where((i_v9766 == 0.0) | ~jnp.isfinite(i_v9766), 1.0, i_v9766)))
    v10045611 = jnp.where(v8935353, v10045556 * v8935854 * v8935859 + jnp.where((v10045559 == 0.0) | ~jnp.isfinite(v10045559), 0.0, jnp.divide(v10045556 * i_v6748, jnp.where((v10045559 == 0.0) | ~jnp.isfinite(v10045559), 1.0, v10045559))) * v8935859 * v8935856, v10045556 * v8935868) * v8935890
    v10048470 = (jnp.where(v9123179, v10045183 * nf, v10045183) + jnp.where(v9123179, v10045611 * nf, v10045611)) * -1.0
    v10048677 = v10048461 - v10048465 + v10048470 - v10048516
    v10042956 = jnp.where(v8653426, -1.0 * v10042951, -1.0 * v10042947)
    v10043005 = v10042956 * v10043003
    v10043062 = jnp.where(v8667923, (0.0 - jnp.where((v10043039 == 0.0) | ~jnp.isfinite(v10043039), 0.0, jnp.divide((v10043005 * v8664497 + jnp.where(v8663761, v10042956 * v10042982, v10042956 * v10042995) * v8664496) * v7319294, jnp.where((v10043039 == 0.0) | ~jnp.isfinite(v10043039), 1.0, v10043039)))) * v10043058, 0.0)
    v10043147 = v10042956 * v10043130 - v10043062 * i_v6250 * i_v7354 - jnp.where(v8669492, (0.0 - jnp.where((v10043067 == 0.0) | ~jnp.isfinite(v10043067), 0.0, jnp.divide((v10043005 * v8667292 + jnp.where(v8667271, v10042956 * v10043012, v10042956 * v10043025) * v8664496) * v7318983, jnp.where((v10043067 == 0.0) | ~jnp.isfinite(v10043067), 1.0, v10043067)))) * v10043086, 0.0) * i_v6268 * i_v7354 + v10042956 * v10043139 + v10042956 * v10043097 - jnp.where(v8683384, v10042956 * v10043114, v10042956 * v10043101) * i_v7337 * v8653449
    v10043176 = jnp.divide(v10042956 * v10043154 + (v10042956 * v10043157 * v8668387 + v10043062 * v8701956), i_v5750)
    v10043201 = jnp.where(v8701975, v10043176, v10043176 * 3.0 * v8701981 + (0.0 - jnp.where((v10043181 == 0.0) | ~jnp.isfinite(v10043181), 0.0, jnp.divide(v10043176 * 8.0, jnp.where((v10043181 == 0.0) | ~jnp.isfinite(v10043181), 1.0, v10043181)))) * v8701983)
    v10043323 = 0.0 - v10043147
    v10043420 = jnp.where((v8734227 == 0.0) | ~jnp.isfinite(v8734227), 0.0, jnp.divide(v10043323 * i_v7210, jnp.where((v8734227 == 0.0) | ~jnp.isfinite(v8734227), 1.0, v8734227))) - jnp.where((v10043408 == 0.0) | ~jnp.isfinite(v10043408), 0.0, jnp.divide(jnp.where(v8731479, v10043201 * v8734181, jnp.where(v8734190, v10043201 * v8734196, v10043201 * v8734208 + (jnp.where((v8726063 == 0.0) | ~jnp.isfinite(v8726063), 0.0, jnp.divide(0.0 - v10043323 * v8731462, jnp.where((v8726063 == 0.0) | ~jnp.isfinite(v8726063), 1.0, v8726063))) - jnp.where((v10043329 == 0.0) | ~jnp.isfinite(v10043329), 0.0, jnp.divide(v10043201 * i_v5824 * v8731470, jnp.where((v10043329 == 0.0) | ~jnp.isfinite(v10043329), 1.0, v10043329)))) * v8734204 * v8734207 * v8701987)) * v8728744, jnp.where((v10043408 == 0.0) | ~jnp.isfinite(v10043408), 1.0, v10043408)))
    v10044037 = jnp.where((4.23e-09 == 0.0) | ~jnp.isfinite(4.23e-09), 0.0, jnp.divide(v10043420 + v10043147 + v10043147, jnp.where((4.23e-09 == 0.0) | ~jnp.isfinite(4.23e-09), 1.0, 4.23e-09)))
    v10044066 = (0.0 - jnp.where((v10044054 == 0.0) | ~jnp.isfinite(v10044054), 0.0, jnp.divide(v10043420 + jnp.divide(v10043147 * v8696278 + v10043147 * v8696278, v10044046) * 2.0, jnp.where((v10044054 == 0.0) | ~jnp.isfinite(v10044054), 1.0, v10044054)))) * 4.23e-09
    v10044110 = v10044037 * v8781625 + (v10042956 * v10044030 + v10044037 * i_v7037) * v8775822 + (((v10044066 * i_v7038 * v8775831 + v10044066 * v8778727) * v8696278 + v10043147 * v8778728) * v8696278 + v10043147 * v8778729)
    v10044157 = 0.0 - jnp.where((v10044148 == 0.0) | ~jnp.isfinite(v10044148), 0.0, jnp.divide(jnp.where(v8799108, v10044110, v10044110 * v8799116 + (0.0 - jnp.where((v10044120 == 0.0) | ~jnp.isfinite(v10044120), 0.0, jnp.divide(v10044110 * 10.0, jnp.where((v10044120 == 0.0) | ~jnp.isfinite(v10044120), 1.0, v10044120)))) * v8799117) * i_v7075, jnp.where((v10044148 == 0.0) | ~jnp.isfinite(v10044148), 1.0, v10044148)))
    v10043430 = v10043420 * i_v6700 + v10042956 * v10043425
    v10043451 = jnp.where(v8740230, v10043430 * v10043445, v10043430 * -2.0)
    v10044705 = v10044157 * v8840286 + jnp.where((i_v6006 == 0.0) | ~jnp.isfinite(i_v6006), 0.0, jnp.divide(v10043420 * v10044680 * v8740245 + v10043451 * v8840233, jnp.where((i_v6006 == 0.0) | ~jnp.isfinite(i_v6006), 1.0, i_v6006))) * v8809061
    v10044182 = 0.0 - jnp.where((v10044173 == 0.0) | ~jnp.isfinite(v10044173), 0.0, jnp.divide(v10044157 * v8809993, jnp.where((v10044173 == 0.0) | ~jnp.isfinite(v10044173), 1.0, v10044173)))
    v10044185 = v10044182 * i_v6006
    v10043531 = v10042956 * v10043509 + (v10042956 * v10043519 * v8734229 + v10043420 * v8756385)
    v10043609 = jnp.where(v8756409, (0.0 - v10043531) * v8756413 + (0.0 - jnp.where((v10043554 == 0.0) | ~jnp.isfinite(v10043554), 0.0, jnp.divide(0.0 - v10043531 * 20.0, jnp.where((v10043554 == 0.0) | ~jnp.isfinite(v10043554), 1.0, v10043554)))) * v8756414, v10043531) * v8759430 + jnp.where(v8759416, v10042956 * v10043598, v10042956 * v10043591) * v8756422
    v10044212 = v10043420 * v10044209
    v10043478 = (v10043420 * v10043459 + v10042956 * v10043453) * v10043474
    v10044172 = v10043451 * i_v7043 * i_v5750 * v8745483 + v10043478 * v8809942
    v10044231 = v10043609 * v8809991 + v10044172 * v8759431
    v10044317 = v10043420 * v8813837 + (0.0 - jnp.where((v10044256 == 0.0) | ~jnp.isfinite(v10044256), 0.0, jnp.divide(v10044212 * 2.0, jnp.where((v10044256 == 0.0) | ~jnp.isfinite(v10044256), 1.0, v10044256)))) * v8813708 + (v10043609 * v8810043 + v10044185 * v8759431) + (v10043420 * v8813823 + v10044231 * v8813708) * 3.0
    v10044277 = v10043609 * 2.0 * v8813834 + (v10044231 + (0.0 - jnp.where((v10044256 == 0.0) | ~jnp.isfinite(v10044256), 0.0, jnp.divide(v10044212, jnp.where((v10044256 == 0.0) | ~jnp.isfinite(v10044256), 1.0, v10044256))))) * v8813828
    v10044419 = jnp.where(v8813727, (v10044185 * v8813708 + v10043420 * v8810043) * v8813800 + (0.0 - jnp.where((v10044389 == 0.0) | ~jnp.isfinite(v10044389), 0.0, jnp.divide(v10043609 * v8810043 + v10044185 * v8759431 + v10043420, jnp.where((v10044389 == 0.0) | ~jnp.isfinite(v10044389), 1.0, v10044389)))) * v8813803, jnp.where((v8813835 == 0.0) | ~jnp.isfinite(v8813835), 0.0, jnp.divide(v10044317 - jnp.divide(v10044317 * v8813843 + v10044317 * v8813843 - (v10044277 * 2.0 * v8813846 + (v10043420 * v8813845 + (v10044185 + (v10043420 * v8809991 + v10044172 * v8813708) * 2.0) * v8813708) * v8813848), v10044357), jnp.where((v8813835 == 0.0) | ~jnp.isfinite(v8813835), 1.0, v8813835))) - jnp.where((v10044364 == 0.0) | ~jnp.isfinite(v10044364), 0.0, jnp.divide(v10044277 * v8813852, jnp.where((v10044364 == 0.0) | ~jnp.isfinite(v10044364), 1.0, v10044364))))
    v10044444 = jnp.divide(v10044419 * v8814779 + v10044419 * v8814779 + v10044419 * v8814781, v10044441)
    v10044487 = jnp.where(v8814817, 0.0, jnp.where(v8814810, 0.0, jnp.where(v8814788, v10044419 - (v10044419 + v10044444) * 0.5, v10044419 * v8814801 + jnp.where((v10044448 == 0.0) | ~jnp.isfinite(v10044448), 0.0, jnp.divide((v10044444 - v10044419) * i_v9705, jnp.where((v10044448 == 0.0) | ~jnp.isfinite(v10044448), 1.0, v10044448))) * v8813856)))
    v10044721 = v10044487 * 0.5
    v10044490 = 0.0 - v10044487
    v10044559 = v10044185 * v8825039 + ((0.0 - jnp.where((v10044494 == 0.0) | ~jnp.isfinite(v10044494), 0.0, jnp.divide(v10044157 * i_v6006 * i_v6880, jnp.where((v10044494 == 0.0) | ~jnp.isfinite(v10044494), 1.0, v10044494)))) * v8825037 + (v10044490 * v8825022 + (0.0 - jnp.where((v10044507 == 0.0) | ~jnp.isfinite(v10044507), 0.0, jnp.divide(v10044182 * i_v7144, jnp.where((v10044507 == 0.0) | ~jnp.isfinite(v10044507), 1.0, v10044507)))) * v8814823) * v10044533 * v8822065) * v8810043
    v10044569 = jnp.where(v8821979, v10044559, v10044185)
    v10044780 = jnp.where((v8840324 == 0.0) | ~jnp.isfinite(v8840324), 0.0, jnp.divide(v10044705 * v8840318 + (v10043420 * v8840317 + (0.0 - (v10044721 * v8840302 + (jnp.where((v8813708 == 0.0) | ~jnp.isfinite(v8813708), 0.0, jnp.divide(v10043609, jnp.where((v8813708 == 0.0) | ~jnp.isfinite(v8813708), 1.0, v8813708))) - jnp.where((v10044582 == 0.0) | ~jnp.isfinite(v10044582), 0.0, jnp.divide(v10043420 * v8759431, jnp.where((v10044582 == 0.0) | ~jnp.isfinite(v10044582), 1.0, v10044582)))) * v8840315)) * v8734229) * v8840293, jnp.where((v8840324 == 0.0) | ~jnp.isfinite(v8840324), 1.0, v8840324))) - jnp.where((v10044768 == 0.0) | ~jnp.isfinite(v10044768), 0.0, jnp.divide((jnp.where((v8825196 == 0.0) | ~jnp.isfinite(v8825196), 0.0, jnp.divide(v10044487, jnp.where((v8825196 == 0.0) | ~jnp.isfinite(v8825196), 1.0, v8825196))) - jnp.where((v10044743 == 0.0) | ~jnp.isfinite(v10044743), 0.0, jnp.divide(v10044569 * v8814821, jnp.where((v10044743 == 0.0) | ~jnp.isfinite(v10044743), 1.0, v10044743)))) * v8840326, jnp.where((v10044768 == 0.0) | ~jnp.isfinite(v10044768), 1.0, v10044768)))
    v10044805 = jnp.where((v8840363 == 0.0) | ~jnp.isfinite(v8840363), 0.0, jnp.divide(v10044780, jnp.where((v8840363 == 0.0) | ~jnp.isfinite(v8840363), 1.0, v8840363))) - jnp.where((v10044793 == 0.0) | ~jnp.isfinite(v10044793), 0.0, jnp.divide((v10044780 * v8745483 + v10043478 * v8840327) * v8840327, jnp.where((v10044793 == 0.0) | ~jnp.isfinite(v10044793), 1.0, v10044793)))
    v10044941 = v10043609 * v8813856 + v10044419 * v8759431
    v10044973 = jnp.where((i_v7353 == 0.0) | ~jnp.isfinite(i_v7353), 0.0, jnp.divide(v10043420 - (jnp.where((v8847873 == 0.0) | ~jnp.isfinite(v8847873), 0.0, jnp.divide(v10043420 * v8847853 + v10044941 * v8813708, jnp.where((v8847873 == 0.0) | ~jnp.isfinite(v8847873), 1.0, v8847873))) - jnp.where((v10044954 == 0.0) | ~jnp.isfinite(v10044954), 0.0, jnp.divide((v10043420 + v10044941) * v8847872, jnp.where((v10044954 == 0.0) | ~jnp.isfinite(v10044954), 1.0, v10044954)))), jnp.where((i_v7353 == 0.0) | ~jnp.isfinite(i_v7353), 1.0, i_v7353)))
    v10044839 = (0.0 - jnp.where((v10044743 == 0.0) | ~jnp.isfinite(v10044743), 0.0, jnp.divide(v10044569 * i_v6688, jnp.where((v10044743 == 0.0) | ~jnp.isfinite(v10044743), 1.0, v10044743)))) * v8734229 + v10043420 * v8843984
    v10044858 = jnp.where(v8843996, v10044839, v10044839 * v10044848)
    v10044820 = jnp.where(v8840975, 0.0, v10043420 * v10044814)
    v10044662 = jnp.where((v8825231 == 0.0) | ~jnp.isfinite(v8825231), 0.0, jnp.divide(v10044569 + v10044419 + ((v10044172 * v8734229 + v10043420 * v8809991) * 2.0 * v8825137 + (0.0 - (jnp.where((v8813708 == 0.0) | ~jnp.isfinite(v8813708), 0.0, jnp.divide(v10043609 * 0.5 * v8813856 + v10044419 * v8825090, jnp.where((v8813708 == 0.0) | ~jnp.isfinite(v8813708), 1.0, v8813708))) - jnp.where((v10044582 == 0.0) | ~jnp.isfinite(v10044582), 0.0, jnp.divide(v10043420 * v8825110, jnp.where((v10044582 == 0.0) | ~jnp.isfinite(v10044582), 1.0, v10044582))))) * v8825199), jnp.where((v8825231 == 0.0) | ~jnp.isfinite(v8825231), 1.0, v8825231))) - jnp.where((v10044650 == 0.0) | ~jnp.isfinite(v10044650), 0.0, jnp.divide((0.0 - jnp.where((v10044634 == 0.0) | ~jnp.isfinite(v10044634), 0.0, jnp.divide(v10044212 * 2.0, jnp.where((v10044634 == 0.0) | ~jnp.isfinite(v10044634), 1.0, v10044634))) + (v10044172 * v8759431 + v10043609 * v8809991)) * v8825201, jnp.where((v10044650 == 0.0) | ~jnp.isfinite(v10044650), 1.0, v10044650)))
    v10044917 = jnp.where((v8844961 == 0.0) | ~jnp.isfinite(v8844961), 0.0, jnp.divide(((v10044820 * v8844931 + v10044858 * v8840993) * v8844864 + (v10043478 * v8840364 + v10044805 * v8745483) * v8844933) * v8844918 + (jnp.where((v8844897 == 0.0) | ~jnp.isfinite(v8844897), 0.0, jnp.divide(v10044419, jnp.where((v8844897 == 0.0) | ~jnp.isfinite(v8844897), 1.0, v8844897))) - jnp.where((v10044871 == 0.0) | ~jnp.isfinite(v10044871), 0.0, jnp.divide(jnp.where(v8821979, jnp.where((i_v6006 == 0.0) | ~jnp.isfinite(i_v6006), 0.0, jnp.divide(v10044559, jnp.where((i_v6006 == 0.0) | ~jnp.isfinite(i_v6006), 1.0, i_v6006))), v10044182) * v8813856, jnp.where((v10044871 == 0.0) | ~jnp.isfinite(v10044871), 1.0, v10044871)))) * v8844934, jnp.where((v8844961 == 0.0) | ~jnp.isfinite(v8844961), 1.0, v8844961)))
    v10045151 = ((v10044805 * v8859396 + (jnp.where((v8850981 == 0.0) | ~jnp.isfinite(v8850981), 0.0, jnp.divide(v10044490, jnp.where((v8850981 == 0.0) | ~jnp.isfinite(v8850981), 1.0, v8850981))) - jnp.where((v10045061 == 0.0) | ~jnp.isfinite(v10045061), 0.0, jnp.divide((jnp.where(v8850957, v10044973 * v8850960 + v10042956 * v10045000 * v8847878, v10044973 * v8850968 + v10042956 * v10044986 * v8847878) * v8844931 + v10044858 * v8850971) * v8814823, jnp.where((v10045061 == 0.0) | ~jnp.isfinite(v10045061), 1.0, v10045061)))) * v8840364) * v8859473 + (jnp.where((v8855415 == 0.0) | ~jnp.isfinite(v8855415), 0.0, jnp.divide(v10044490, jnp.where((v8855415 == 0.0) | ~jnp.isfinite(v8855415), 1.0, v8855415))) - jnp.where((v10045086 == 0.0) | ~jnp.isfinite(v10045086), 0.0, jnp.divide(jnp.where(v8854726, v10044820 * v8855389, 0.0) * v8814823, jnp.where((v10045086 == 0.0) | ~jnp.isfinite(v10045086), 1.0, v10045086)))) * v8859447) * v8859595 + (jnp.where((v8859574 == 0.0) | ~jnp.isfinite(v8859574), 0.0, jnp.divide(jnp.where((v8859546 == 0.0) | ~jnp.isfinite(v8859546), 0.0, jnp.divide(jnp.where((v8825232 == 0.0) | ~jnp.isfinite(v8825232), 0.0, jnp.divide(v10044662 + jnp.where(v8844831, v10044917 * v8814823 + v10044490 * v8844962, 0.0), jnp.where((v8825232 == 0.0) | ~jnp.isfinite(v8825232), 1.0, v8825232))) - jnp.where((v10045111 == 0.0) | ~jnp.isfinite(v10045111), 0.0, jnp.divide(v10044662 * v8851029, jnp.where((v10045111 == 0.0) | ~jnp.isfinite(v10045111), 1.0, v10045111))), jnp.where((v8859546 == 0.0) | ~jnp.isfinite(v8859546), 1.0, v8859546))), jnp.where((v8859574 == 0.0) | ~jnp.isfinite(v8859574), 1.0, v8859574))) - jnp.where((v10045127 == 0.0) | ~jnp.isfinite(v10045127), 0.0, jnp.divide(jnp.where(v8844831, v10044917, 0.0) * v8859547, jnp.where((v10045127 == 0.0) | ~jnp.isfinite(v10045127), 1.0, v10045127)))) * v8859477
    v10045214 = v10045151 * v8868904 + (jnp.where((v8859351 == 0.0) | ~jnp.isfinite(v8859351), 0.0, jnp.divide(v10044490, jnp.where((v8859351 == 0.0) | ~jnp.isfinite(v8859351), 1.0, v8859351))) - jnp.where((v10045190 == 0.0) | ~jnp.isfinite(v10045190), 0.0, jnp.divide(v10044490 * v10045051 * v8814823, jnp.where((v10045190 == 0.0) | ~jnp.isfinite(v10045190), 1.0, v10045190)))) * v8859598
    v10045223 = v10045214 * v8814821 + v10044487 * v8868910
    v10046247 = jnp.where(v9123179, v10045223 * nf, v10045223)
    v10048462 = jnp.where(v9336400, -v10046247, v10046247) * -1.0
    v10042969 = v10042956 * v10042967
    v10045720 = (v10042969 + jnp.where(v8984547, v10042969 * v10045708, v10042969 * v10045699)) * 0.5
    v10045760 = -(v10045720 - v10042969 - v10043420) + v10043420
    v10045843 = v10043323 * v8725872 * i_v7205 * v9031730 + v10045760 * v10045822 * v9031743
    v10045899 = -(v10043420 * v10045856 * v9038767 + (0.0 - (jnp.where((v9038568 == 0.0) | ~jnp.isfinite(v9038568), 0.0, jnp.divide(v10044721, jnp.where((v9038568 == 0.0) | ~jnp.isfinite(v9038568), 1.0, v9038568))) - jnp.where((v10045848 == 0.0) | ~jnp.isfinite(v10045848), 0.0, jnp.divide(v10043420 * v8840315, jnp.where((v10045848 == 0.0) | ~jnp.isfinite(v10045848), 1.0, v10045848))))) * v9038570) * v8814821 + v10044487 * v9038770
    v10045915 = v10045899 * v9038781
    v10045906 = v10045899 * v9038773
    v10045908 = v10045906 + v10045906
    v10045989 = v10045843 * jnp.where((v9038775 == 0.0) | ~jnp.isfinite(v9038775), 0.0, jnp.divide(v9038813, jnp.where((v9038775 == 0.0) | ~jnp.isfinite(v9038775), 1.0, v9038775))) + (jnp.where((v9038775 == 0.0) | ~jnp.isfinite(v9038775), 0.0, jnp.divide(v10045899 * v9038781 + v10045915 * v9038773 - v10045915, jnp.where((v9038775 == 0.0) | ~jnp.isfinite(v9038775), 1.0, v9038775))) - jnp.where((v10045931 == 0.0) | ~jnp.isfinite(v10045931), 0.0, jnp.divide(v10045908 * v9038813, jnp.where((v10045931 == 0.0) | ~jnp.isfinite(v10045931), 1.0, v10045931)))) * v9031746
    v10048483 = jnp.where(v9123179, v10045989 * nf, v10045989) * -1.0
    v10045186 = jnp.where(v8868766, v10044490 * v10045162, v10044490 * v8868779) * v8868864 + (v10045151 * v8814821 + v10044487 * v8859598) * v8868865
    v10045596 = --1.0 * v8935877 + (-1.0 * v8471980 + -1.0 * v8471980) * v8935878
    v10045612 = (jnp.where((v8935889 == 0.0) | ~jnp.isfinite(v8935889), 0.0, jnp.divide(v10045596, jnp.where((v8935889 == 0.0) | ~jnp.isfinite(v8935889), 1.0, v8935889))) - jnp.where((v10045599 == 0.0) | ~jnp.isfinite(v10045599), 0.0, jnp.divide(v10045596 * v8935879, jnp.where((v10045599 == 0.0) | ~jnp.isfinite(v10045599), 1.0, v10045599)))) * v8935891
    v10048471 = (jnp.where(v9123179, v10045186 * nf, v10045186) + jnp.where(v9123179, v10045612 * nf, v10045612)) * -1.0
    v10048547 = (-1.0 * _ckt_gmin - (jnp.where(v8608058, -1.0 * v10042628, -1.0 * v10042604) * i_v8982 + jnp.where(v8609177, -1.0 * v10042750, -1.0 * v10042726) * i_v8986 + jnp.where(v8610404, -1.0 * v10042872, -1.0 * v10042848) * i_v8992)) * -1.0
    v10048762 = v10048462 - v10048483 + v10048471 - v10048517 - v10048547
    v10048548 = (-1.0 * _ckt_gmin - (jnp.where(v8608058, -1.0 * v10042628, -1.0 * v10042604) * i_v8982 + jnp.where(v8609177, -1.0 * v10042750, -1.0 * v10042726) * i_v8986 + jnp.where(v8610404, -1.0 * v10042872, -1.0 * v10042848) * i_v8992)) * -1.0
    v10048763 = 0.0 - v10048548
    v10048177 = jnp.where(v9297198, 0.0, -1.0)
    v10048235 = jnp.where(v9299631, v10048177 * i_v9513, v10048177 * v10048201)
    v10047848 = v10043320 * i_v9853
    v10048213 = (1.0 - jnp.divide(v9306449 + v9306449, 2.0 * v9306453)) * 0.5
    v10048557 = (v10043320 * v9151211 * 0.4 - jnp.where(v9123179, v10048235 * nf, v10048235)) * -1.0
    v10047849 = (0.0 - jnp.where(v9145086, -1.0, v10042969)) * i_v9853
    v10048558 = (0.0 - v10042956 * v10043125) * v9151211 * 0.4 * -1.0
    v10048579 = -1.0 * v10048020 * -1.0
    v10048767 = v10048558 - v10048579
    v10048178 = jnp.where(v9297198, -1.0, 0.0)
    v10048236 = jnp.where(v9299631, v10048178 * i_v9513, v10048178 * v10048201)
    v10048246 = jnp.where(v9123179, v10048236 * nf, v10048236)
    v10048233 = jnp.where(v9299631, v10048178 * i_v9512, v10048178 * (i_v9512 + i_v9922 - (v10048213 + jnp.divide(0.0 - jnp.where((i_v9271 == 0.0) | ~jnp.isfinite(i_v9271), 0.0, jnp.divide(v10048213 * 4.0, jnp.where((i_v9271 == 0.0) | ~jnp.isfinite(i_v9271), 1.0, i_v9271))), 2.0 * jnp.sqrt(jnp.maximum(1.0 - jnp.where((i_v9271 == 0.0) | ~jnp.isfinite(i_v9271), 0.0, jnp.divide(4.0 * (0.5 * (v9306449 - v9306453)), jnp.where((i_v9271 == 0.0) | ~jnp.isfinite(i_v9271), 1.0, i_v9271))), 1e-300))) * (0.5 * i_v9271)) * i_v9922))
    v10048559 = (0.0 - v10048246) * -1.0
    v10048580 = -1.0 * v10048020 * -1.0
    v10048769 = 0.0 - v10048580
    v10048675 = 0.0 - v9407274
    v10045946 = ((v10043319 * v9006450 + v10043319 * v8725872) * i_v7205 * v9031730 + v10045758 * v10045822 * v9031743) * v9038798 + (jnp.where((v9038775 == 0.0) | ~jnp.isfinite(v9038775), 0.0, jnp.divide(v10045893 * v9038781 - v10045893, jnp.where((v9038775 == 0.0) | ~jnp.isfinite(v9038775), 1.0, v9038775))) - jnp.where((v10045931 == 0.0) | ~jnp.isfinite(v10045931), 0.0, jnp.divide((v10045893 * v9038773 + v10045893 * v9038773) * v9038797, jnp.where((v10045931 == 0.0) | ~jnp.isfinite(v10045931), 1.0, v10045931)))) * v9031746
    v10048475 = jnp.where(v9123179, v10045946 * nf, v10045946) * -1.0
    v10045632 = jnp.where((i_v9766 == 0.0) | ~jnp.isfinite(i_v9766), 0.0, jnp.divide(--1.0 - v10043317, jnp.where((i_v9766 == 0.0) | ~jnp.isfinite(i_v9766), 1.0, i_v9766)))
    v10045636 = v8939333 * v8939333
    v10045673 = jnp.where(v8948800, v10045632 * v8948835 * v8948840 + jnp.where((v10045636 == 0.0) | ~jnp.isfinite(v10045636), 0.0, jnp.divide(v10045632 * i_v6430, jnp.where((v10045636 == 0.0) | ~jnp.isfinite(v10045636), 1.0, v10045636))) * v8948840 * v8948837, v10045632 * v8948845) * v8948867
    v10048472 = jnp.where(v9123179, v10045673 * nf, v10045673) * -1.0
    v10048652 = -v10048460 - v10048475 + v10048472
    v10048549 = v10047847 * -1.0
    v10048563 = -v10047847 * -1.0
    v10048570 = -v10048563
    v10048655 = -v10048556 - v10048549 + v10048570
    v10045996 = jnp.divide(-1.0 * v9042303 + -1.0 * v9042303, 2.0 * v9042306)
    v10046015 = (-1.0 * v9042306 + v10045996 * v8471935) * i_v7197 * v9058422 + (v10045996 * i_v9820 - (v10045996 * i_v9821 * v9042306 + v10045996 * v9058413)) * i_v7202 * v9058422 * v9058435
    v10048463 = jnp.where(v9123179, v10046015 * nf, v10046015) * -1.0
    v10045633 = jnp.where((i_v9766 == 0.0) | ~jnp.isfinite(i_v9766), 0.0, jnp.divide(0.0 - v10043318, jnp.where((i_v9766 == 0.0) | ~jnp.isfinite(i_v9766), 1.0, i_v9766)))
    v10045674 = jnp.where(v8948800, v10045633 * v8948835 * v8948840 + jnp.where((v10045636 == 0.0) | ~jnp.isfinite(v10045636), 0.0, jnp.divide(v10045633 * i_v6430, jnp.where((v10045636 == 0.0) | ~jnp.isfinite(v10045636), 1.0, v10045636))) * v8948840 * v8948837, v10045633 * v8948845) * v8948867
    v10048473 = jnp.where(v9123179, v10045674 * nf, v10045674) * -1.0
    v10048673 = -v10048461 - v10048463 + v10048473 - v10048507
    v10045952 = v10045843 * v9038798 + (jnp.where((v9038775 == 0.0) | ~jnp.isfinite(v9038775), 0.0, jnp.divide(v10045915 - v10045899, jnp.where((v9038775 == 0.0) | ~jnp.isfinite(v9038775), 1.0, v9038775))) - jnp.where((v10045931 == 0.0) | ~jnp.isfinite(v10045931), 0.0, jnp.divide(v10045908 * v9038797, jnp.where((v10045931 == 0.0) | ~jnp.isfinite(v10045931), 1.0, v10045931)))) * v9031746
    v10048477 = jnp.where(v9123179, v10045952 * nf, v10045952) * -1.0
    v10045666 = --1.0 * v8948854 + (-1.0 * v8471939 + -1.0 * v8471939) * v8948855
    v10045675 = (jnp.where((v8948866 == 0.0) | ~jnp.isfinite(v8948866), 0.0, jnp.divide(v10045666, jnp.where((v8948866 == 0.0) | ~jnp.isfinite(v8948866), 1.0, v8948866))) - jnp.where((v8948866 * v8948866 == 0.0) | ~jnp.isfinite(v8948866 * v8948866), 0.0, jnp.divide(v10045666 * v8948856, jnp.where((v8948866 * v8948866 == 0.0) | ~jnp.isfinite(v8948866 * v8948866), 1.0, v8948866 * v8948866)))) * v8948868
    v10048474 = jnp.where(v9123179, v10045675 * nf, v10045675) * -1.0
    v10042516 = 0.0 - -1.0
    v10042518 = v8607542 * v8607542
    v10042640 = v8608602 * v8608602
    v10042762 = v8609775 * v8609775
    v10048544 = (-1.0 * _ckt_gmin - (jnp.where(v8607544, jnp.where((i_v9590 == 0.0) | ~jnp.isfinite(i_v9590), 0.0, jnp.divide(--1.0, jnp.where((i_v9590 == 0.0) | ~jnp.isfinite(i_v9590), 1.0, i_v9590))) * 1000.0 * 583461742500000.0, (jnp.where((i_v9590 == 0.0) | ~jnp.isfinite(i_v9590), 0.0, jnp.divide(--1.0, jnp.where((i_v9590 == 0.0) | ~jnp.isfinite(i_v9590), 1.0, i_v9590))) * 10.0 * v8607581 + (0.0 - jnp.where((v10042518 == 0.0) | ~jnp.isfinite(v10042518), 0.0, jnp.divide(v10042516, jnp.where((v10042518 == 0.0) | ~jnp.isfinite(v10042518), 1.0, v10042518)))) * v8607585) * 583461742500000.0) * i_v8980 + jnp.where(v8608604, jnp.where((i_v9588 == 0.0) | ~jnp.isfinite(i_v9588), 0.0, jnp.divide(--1.0, jnp.where((i_v9588 == 0.0) | ~jnp.isfinite(i_v9588), 1.0, i_v9588))) * 1000.0 * 583461742500000.0, (jnp.where((i_v9588 == 0.0) | ~jnp.isfinite(i_v9588), 0.0, jnp.divide(--1.0, jnp.where((i_v9588 == 0.0) | ~jnp.isfinite(i_v9588), 1.0, i_v9588))) * 10.0 * v8608671 + (0.0 - jnp.where((v10042640 == 0.0) | ~jnp.isfinite(v10042640), 0.0, jnp.divide(v10042516, jnp.where((v10042640 == 0.0) | ~jnp.isfinite(v10042640), 1.0, v10042640)))) * v8608675) * 583461742500000.0) * i_v8984 + jnp.where(v8609777, jnp.where((i_v9589 == 0.0) | ~jnp.isfinite(i_v9589), 0.0, jnp.divide(--1.0, jnp.where((i_v9589 == 0.0) | ~jnp.isfinite(i_v9589), 1.0, i_v9589))) * 1000.0 * 583461742500000.0, (jnp.where((i_v9589 == 0.0) | ~jnp.isfinite(i_v9589), 0.0, jnp.divide(--1.0, jnp.where((i_v9589 == 0.0) | ~jnp.isfinite(i_v9589), 1.0, i_v9589))) * 10.0 * v8609874 + (0.0 - jnp.where((v10042762 == 0.0) | ~jnp.isfinite(v10042762), 0.0, jnp.divide(v10042516, jnp.where((v10042762 == 0.0) | ~jnp.isfinite(v10042762), 1.0, v10042762)))) * v8609878) * 583461742500000.0) * i_v8989)) * -1.0
    v10048747 = -v10048462 - v10048477 + v10048474 - v10048508 - v10048544
    v10042517 = 0.0 - -1.0
    v10048545 = (-1.0 * _ckt_gmin - (jnp.where(v8607544, jnp.where((i_v9590 == 0.0) | ~jnp.isfinite(i_v9590), 0.0, jnp.divide(--1.0, jnp.where((i_v9590 == 0.0) | ~jnp.isfinite(i_v9590), 1.0, i_v9590))) * 1000.0 * 583461742500000.0, (jnp.where((i_v9590 == 0.0) | ~jnp.isfinite(i_v9590), 0.0, jnp.divide(--1.0, jnp.where((i_v9590 == 0.0) | ~jnp.isfinite(i_v9590), 1.0, i_v9590))) * 10.0 * v8607581 + (0.0 - jnp.where((v10042518 == 0.0) | ~jnp.isfinite(v10042518), 0.0, jnp.divide(v10042517, jnp.where((v10042518 == 0.0) | ~jnp.isfinite(v10042518), 1.0, v10042518)))) * v8607585) * 583461742500000.0) * i_v8980 + jnp.where(v8608604, jnp.where((i_v9588 == 0.0) | ~jnp.isfinite(i_v9588), 0.0, jnp.divide(--1.0, jnp.where((i_v9588 == 0.0) | ~jnp.isfinite(i_v9588), 1.0, i_v9588))) * 1000.0 * 583461742500000.0, (jnp.where((i_v9588 == 0.0) | ~jnp.isfinite(i_v9588), 0.0, jnp.divide(--1.0, jnp.where((i_v9588 == 0.0) | ~jnp.isfinite(i_v9588), 1.0, i_v9588))) * 10.0 * v8608671 + (0.0 - jnp.where((v10042640 == 0.0) | ~jnp.isfinite(v10042640), 0.0, jnp.divide(v10042517, jnp.where((v10042640 == 0.0) | ~jnp.isfinite(v10042640), 1.0, v10042640)))) * v8608675) * 583461742500000.0) * i_v8984 + jnp.where(v8609777, jnp.where((i_v9589 == 0.0) | ~jnp.isfinite(i_v9589), 0.0, jnp.divide(--1.0, jnp.where((i_v9589 == 0.0) | ~jnp.isfinite(i_v9589), 1.0, i_v9589))) * 1000.0 * 583461742500000.0, (jnp.where((i_v9589 == 0.0) | ~jnp.isfinite(i_v9589), 0.0, jnp.divide(--1.0, jnp.where((i_v9589 == 0.0) | ~jnp.isfinite(i_v9589), 1.0, i_v9589))) * 10.0 * v8609874 + (0.0 - jnp.where((v10042762 == 0.0) | ~jnp.isfinite(v10042762), 0.0, jnp.divide(v10042517, jnp.where((v10042762 == 0.0) | ~jnp.isfinite(v10042762), 1.0, v10042762)))) * v8609878) * 583461742500000.0) * i_v8989)) * -1.0
    v10048748 = 0.0 - v10048545
    v10048550 = v10047848 * -1.0
    v10048564 = -v10047848 * -1.0
    v10048656 = -v10048557 - v10048550 + -v10048564
    v10048551 = v10047849 * -1.0
    v10048565 = (-v10047849 - v10042191 * i_v7113) * -1.0
    v10048574 = (-1.0 * v9285121 + -1.0 * v9285118 * 0.5 * v8471939) * -1.0
    v10048752 = -v10048558 - v10048551 + -v10048565 - v10048574
    v10048552 = (v10048246 + jnp.where(v9123179, v10048233 * nf, v10048233)) * -1.0
    v10048566 = (0.0 - -1.0 * i_v7113) * -1.0
    v10048658 = -v10048559 - v10048552 + -v10048566
    v10048575 = (-1.0 * v9285121 + -1.0 * v9285118 * 0.5 * v8471939) * -1.0
    v10048754 = 0.0 - v10048575
    v10046088 = v10043319 - v10042968
    v10046172 = (i_v9844 - (v9122901 + v9122901)) * i_v9841 * v9122910
    v10046109 = (i_v9834 - (v9108658 + v9108658)) * i_v9831 * v9108667
    v10046194 = (v10046088 * v9112940 + v10045758 * v9085301) * i_v9839 * v9122910 + v10045758 * v10046172 * v9122923 + ((v10046088 * v9085271 + (-v10043319 + v10042968) * v9085301) * i_v9830 * v9108667 + jnp.where(v8984572, 0.0, v10045718) * v10046109 * v9108680)
    v10048493 = jnp.where(v9123179, v10046194 * nf, v10046194) * -1.0
    v10045299 = (v10044699 * i_v9749 + v10045208) * i_v6868
    v10048531 = 0.0
    v10048686 = v10048475 + v10048464 + v10048493 - v10048531
    v10046195 = (v10043320 * v9112940 + v10045759 * v9085301) * i_v9839 * v9122910 + v10045759 * v10046172 * v9122923 + ((v10043320 * v9085271 + -v10043320 * v9085301) * i_v9830 * v9108667 + jnp.where(v8984572, 0.0, v10045719) * v10046109 * v9108680)
    v10048494 = jnp.where(v9123179, v10046195 * nf, v10046195) * -1.0
    v10045300 = (v10044702 * i_v9749 + v10045211) * i_v6868
    v10048532 = 0.0
    v10048687 = v10048463 + v10048465 + v10048494 - v10048532
    v10046090 = 0.0 - v10042969
    v10046196 = (v10046090 * v9112940 + v10045760 * v9085301) * i_v9839 * v9122910 + v10045760 * v10046172 * v9122923 + ((v10046090 * v9085271 + v10042969 * v9085301) * i_v9830 * v9108667 + jnp.where(v8984572, 0.0, v10045720) * v10046109 * v9108680)
    v10048495 = jnp.where(v9123179, v10046196 * nf, v10046196) * -1.0
    v10045301 = (v10044705 * i_v9749 + v10045214) * i_v6868
    v10048533 = 0.0
    v10048688 = v10048477 + v10048483 + v10048495 - v10048533
    v10048689 = 0.0 - v9447351
    v10048669 = -v10048469 - v10048472 - v10048493
    v10048670 = -v10048470 - v10048473 - v10048494
    v10048671 = -v10048471 - v10048474 - v10048495
    j_resist = jnp.array([[_mfactor * v9418326 + _mfactor * (v10048515 - v9418326) + _mfactor * v10048679 + _mfactor * (v10048760 - v10048679), _mfactor * v10048516 + _mfactor * v10048677, _mfactor * (-v10048515 - v10048516 - v10048517) + _mfactor * (-v10048760 - v10048677 - v10048762 - v10048763), 0.0, _mfactor * v10048517 + _mfactor * v10048762, 0.0, _mfactor * v10048763, 0.0, 0.0, 0.0, 0.0, 0.0], [_mfactor * v10048686 + _mfactor * v10048531, init[186] + i_v10081 + _mfactor * (v10048687 - v10048689) + _mfactor * v10048689 + i_v10081 + _mfactor * (v10048532 - v9447351) + _mfactor * (i_v7837 + v9447351), _mfactor * (-v10048686 - v10048687 - v10048688) + _mfactor * (-v10048531 - v10048532 - v10048533), 0.0, _mfactor * v10048688 + _mfactor * v10048533, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [_mfactor * v10048652, _mfactor * v10048507 + _mfactor * v10048673, _mfactor * v9407274 + _mfactor * (-v10048507 - v10048508 - v9407274) + _mfactor * v10048675 + _mfactor * (-v10048652 - v10048673 - v10048747 - v10048748 - v10048675), 0.0, _mfactor * v10048508 + _mfactor * v10048747, _mfactor * v10048748, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i_v10079, _mfactor, _mfactor, 0.0], [_mfactor * v10048669, _mfactor * v10048670, _mfactor * (-v10048669 - v10048670 - v10048671), 0.0, _mfactor * v10048671, 0.0, 0.0, i_v10079, 0.0, i_v10079, 0.0, _mfactor], [0.0, 0.0, _mfactor * (-v10048544 - v10048545), 0.0, _mfactor * v10048544, _mfactor * v10048545, 0.0, _mfactor, _mfactor, 0.0, 0.0, 0.0], [_mfactor * v10048546, 0.0, _mfactor * (-v10048546 - v10048547 - v10048548), 0.0, _mfactor * v10048547, 0.0, _mfactor * v10048548, 0.0, 0.0, 0.0, i_v10079, i_v10079], [0.0, 0.0, 0.0, 0.0, -0.020000000001, 0.020000000001, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, -0.020000000001, 0.0, 0.020000000001, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.020000000001, -0.020000000001, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.020000000001, 0.0, 0.0, -0.020000000001, 0.0, 0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.020000000001, 0.0, -0.020000000001, 0.0, 0.0, 0.0, 0.0, -1.0]])
    j_react = jnp.array([[_mfactor * v10048765, _mfactor * v10048557 + _mfactor * v10048559, _mfactor * (-v10048765 - v10048557 - v10048767 - v10048559 - v10048769), 0.0, _mfactor * v10048767, 0.0, _mfactor * v10048769, 0.0, 0.0, 0.0, 0.0, 0.0], [_mfactor * v10048549, _mfactor * v10048550 + _mfactor * v10048552, _mfactor * (-v10048549 - v10048550 - v10048551 - v10048552), 0.0, _mfactor * v10048551, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [_mfactor * v10048655, _mfactor * v10048656 + _mfactor * v10048658, _mfactor * (-v10048655 - v10048656 - v10048752 - v10048658 - v10048754), 0.0, _mfactor * v10048752, _mfactor * v10048754, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [_mfactor * v10048563, _mfactor * v10048564 + _mfactor * v10048566, _mfactor * (v10048570 - v10048564 - v10048565 - v10048566), 0.0, _mfactor * v10048565, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, _mfactor * (-v10048574 - v10048575), 0.0, _mfactor * v10048574, _mfactor * v10048575, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [_mfactor * v10048578, 0.0, _mfactor * (-v10048578 - v10048579 - v10048580), 0.0, _mfactor * v10048579, 0.0, _mfactor * v10048580, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    return ({'d': _mfactor * v9418327 + _mfactor * (v9336922 - v9356292 + v9366193 - v9418327 - v9563754), 'g': _mfactor * v9437770 + _mfactor * (v9346607 + v9356292 + v9386314 - v9447352) + _mfactor * (-v9437770 + v9447352), 's': _mfactor * v9407275 + _mfactor * (-v9336922 - v9346607 + v9375878 - v9407275 - v9553466), 'b': _mfactor * (-s.i_br19 + s.i_br21 + s.i_br20), 'v_bi': _mfactor * (-v9366193 - v9375878 - v9386314 - s.i_br18 - s.i_br20 + s.i_br22), 'v_sbulk': _mfactor * (s.i_br18 + s.i_br19 + v9553466), 'v_dbulk': _mfactor * (-s.i_br21 - s.i_br22 + v9563754), 'i_br18': (s.v_sbulk - s.v_bi) * init[111] - s.i_br18, 'i_br19': (s.v_sbulk - signals.b) * init[110] - s.i_br19, 'i_br20': (signals.b - s.v_bi) * init[109] - s.i_br20, 'i_br21': (signals.b - s.v_dbulk) * init[108] - s.i_br21, 'i_br22': (s.v_bi - s.v_dbulk) * init[107] - s.i_br22}, {'d': _mfactor * (v9947397 - v9947401), 's': _mfactor * (-v9947397 - v9947395 + -v9947398 - v9947400), 'g': _mfactor * v9947395, 'v_bi': _mfactor * v9947398, 'v_sbulk': _mfactor * v9947400, 'v_dbulk': _mfactor * v9947401}, j_resist, j_react)

def _SKY130_PFET_01V8_TT_jacobian(signals: Signals, s: States, init, l: float=1.5e-07, w: float=2e-06, nf: float=1.0, _min: float=0.0, ad: float=0.0, ps: float=0.0, pd: float=0.0, sa: float=0.0, sb: float=0.0, sd: float=0.0, delvto: float=0.0, _ckt_gmin: float=1e-12, off: float=0.0, _temperature: float=300.15, _mfactor: float=1.0) -> tuple:
    """Jacobian wrapper — slices the combined function's output."""
    _f, _q, j_resist, j_react = _SKY130_PFET_01V8_TT_combined(signals, s, init, l=l, w=w, nf=nf, _min=_min, ad=ad, ps=ps, pd=pd, sa=sa, sb=sb, sd=sd, delvto=delvto, _ckt_gmin=_ckt_gmin, off=off, _temperature=_temperature, _mfactor=_mfactor)
    return (j_resist, j_react)

@va_component(ports=('d', 'g', 's', 'b'), states=('v_bi', 'v_sbulk', 'v_dbulk', 'i_br18', 'i_br19', 'i_br20', 'i_br21', 'i_br22'), jacobian_fn=_SKY130_PFET_01V8_TT_jacobian, combined_fn=_SKY130_PFET_01V8_TT_combined)
def SKY130_PFET_01V8_TT(signals: Signals, s: States, init, l: float=1.5e-07, w: float=2e-06, nf: float=1.0, _min: float=0.0, ad: float=0.0, ps: float=0.0, pd: float=0.0, sa: float=0.0, sb: float=0.0, sd: float=0.0, delvto: float=0.0, _ckt_gmin: float=1e-12, off: float=0.0, _temperature: float=300.15, _mfactor: float=1.0) -> PhysicsReturn:
    """Auto-generated from Verilog-A — thin wrapper over ``_combined``."""
    f, q, _j_f, _j_q = _SKY130_PFET_01V8_TT_combined(signals, s, init, l=l, w=w, nf=nf, _min=_min, ad=ad, ps=ps, pd=pd, sa=sa, sb=sb, sd=sd, delvto=delvto, _ckt_gmin=_ckt_gmin, off=off, _temperature=_temperature, _mfactor=_mfactor)
    return (f, q)

@SKY130_PFET_01V8_TT.setup
def _SKY130_PFET_01V8_TT_register_setup(*_a, **_kw):
    return _SKY130_PFET_01V8_TT_setup(*_a, **_kw)