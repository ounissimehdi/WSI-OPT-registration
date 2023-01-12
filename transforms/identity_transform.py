import math
import numpy as np
from transforms.transform_base import TransformBase

class IdentityTransform(TransformBase):
    def __init__(self, dim):
        TransformBase.__init__(self, dim, 0)

    def copy_child(self):
        return IdentityTransform(self.get_dim())
    
    def transform(self, pnts):
        return pnts

    def grad(self, pnts, gradients, output_gradients):
        #res = gradients.sum(axis=0)
        if output_gradients == True:
            return np.array([]), gradients
        else:
            return np.array([])

    def invert(self):
        self_inv = self.copy()

        return self_inv

    def inverse_to_forward_matrix(self):
        return np.array([])

    def itk_transform_string_rec(self, index):
        s = '#Transform %d\n' % index
        s = s + 'Transform: IdentityTransform_double_%d_%d\n' % (self.get_dim(), self.get_dim())
        s = s + 'Parameters:\n'
        s = s + 'FixedParameters:\n'

        return s 

if __name__ == '__main__':
    t = IdentityTransform(2)

    print(t.get_param_count())
    print(t.get_params())

    pnts = np.array([[2.0, 3.0]])

    tpnts = t.transform(pnts)

    print(tpnts)

    tinv = t.invert()

    print(tinv.get_params())

    tinvpnts = tinv.transform(tpnts)

    print(tinvpnts)

    dd_dv = np.array([[2, 4]])

    # forward grad
    print(t.grad(tpnts, dd_dv, False))

    # inverse grad    
    print(tinv.grad_inverse_to_forward(tinv.grad(tpnts, np.array([-1,1])*dd_dv, False)))
