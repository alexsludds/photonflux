import numpy as np
from ..Base_model import BaseModel

class laser_model(BaseModel):
    def __init__(self):
        super().__init__(port_names=["o1"])

    def __call__(self):
        return {
            ("o1"): 1
        }