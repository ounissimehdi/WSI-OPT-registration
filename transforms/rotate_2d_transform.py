import math
import numpy as np
from transforms.transform_base import TransformBase

class Rotate2DTransform(TransformBase):
    def __init__(self):
        TransformBase.__init__(self, 2, 1)

    def copy_child(self):
        return Rotate2DTransform()
    
    def transform(self, pnts):
        theta = self.get_param(0)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        M = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])
        
        return pnts.dot(M)

    def grad(self, pnts, gradients, output_gradients):
        res = np.zeros((1,))
        theta = self.get_param(0)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        Mprime = np.array([[-sin_theta, cos_theta], [-cos_theta, -sin_theta]])
        Mprimepnts = pnts.dot(Mprime)
        res[:] = (Mprimepnts * gradients).sum()

        if output_gradients == True:
            M = np.transpose(np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]]))
        
            return res, gradients.dot(M)
        else:
            return res

    def invert(self):
        self_inv = self.copy()
        self_inv.set_params(-self_inv.get_params())

        return self_inv

    def inverse_to_forward_matrix(self):
        return np.array([[-1.0]])
        
    #def grad_inverse_to_forward(self, inv_grad):
    #    res = np.zeros((1,))
    #    res[:] = -inv_grad
    #    return res
