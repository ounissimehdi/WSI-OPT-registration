import math
import numpy as np
from transforms.transform_base import TransformBase

class TranslationTransform(TransformBase):
    def __init__(self, dim):
        TransformBase.__init__(self, dim, dim)

    def copy_child(self):
        return TranslationTransform(self.get_dim())
    
    def transform(self, pnts):
        offset = self.get_params()
        #print(pnts)
        return pnts + offset

    def grad(self, pnts, gradients, output_gradients):
        res = gradients.sum(axis=0)
        if output_gradients == True:
            return res, gradients
        else:
            return res

    def invert(self):
        self_inv = self.copy()
        self_inv.set_params(-self_inv.get_params())

        return self_inv

    def inverse_to_forward_matrix(self):
        return -np.eye(self.get_param_count(), self.get_param_count())

    def itk_transform_string_rec(self, index):
        s = '#Transform %d\n' % index
        s = s + 'Transform: TranslationTransform_double_%d_%d\n' % (self.get_dim(), self.get_dim())
        s = s + 'Parameters:'
        for i in range(self.get_param_count()):
            s = s + (' %f' % self.get_param(i))
        s = s + '\n'
        s = s + 'FixedParameters:\n'

        return s 

if __name__ == '__main__':
    t = TranslationTransform(2)

    print(t.get_param_count())
    print(t.get_params())

    t.set_param(0, 12.0)
    t.set_param(1, -24.0)

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
    print(t.grad(tpnts, dd_dv))

    # inverse grad
    print(tinv.grad_inverse_to_forward(tinv.grad(tpnts, np.array([-1,1])*dd_dv)))
