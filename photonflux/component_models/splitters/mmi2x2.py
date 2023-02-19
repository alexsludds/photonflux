import numpy as np
from ..Base_model import BaseModel

class mmi1x2_model(BaseModel):
    def __init__(self, info, settings):
        assert False
        #TODO: Implement this model
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