import sympy
import numpy as np
from sympy.abc import X

def affine_inv_diff(dim):
    A = sympy.Matrix(sympy.symarray('a', (dim+1, dim+1)))
    for i in range(dim):
        A[dim,i] = 0
    A[dim,dim] = 1

    det = sympy.simplify(A.det())
    det_symbol = sympy.Symbol('det')
    
    B = sympy.simplify(A.inv())

    M = [sympy.simplify(B.diff(A[i, j])).subs(det, det_symbol) for i in range(dim) for j in range(dim)]
    T = [sympy.simplify(B.diff(A[i, dim])).subs(det, det_symbol) for i in range(dim)]

    # Generate code

    print("def diff_inv_to_forward_transform_%dd(param):" % dim)
    
    print("    ")
    print("    # Generate local variables for each parameter")
    for i in range(dim):
        for j in range(dim):
            index = i * dim + j
            print("    a_%d_%d = param[%d]" % (i,j,index))
    for i in range(dim):
        index = dim*dim + i
        print("    a_%d_%d = param[%d]" % (i,dim,index))

    print("    ")
    print("    # Compute determinant")
    print("    det = " + str(det))

    s = ""
    for k in range(dim * dim):
        if k == 0:
            s = s + "["
        else:
            s = s + ", ["
        for i in range(dim):
            for j in range(dim):
                x = M[k][i, j]
                if i == 0 and j == 0:
                    s = s + str(x)
                else:
                    s = s + ", " + str(x)
        for i in range(dim):
            x = M[k][i, dim]
            s = s + ", " + str(x)
        s = s + "]"
    for k in range(dim):
        s = s + ", ["
        for i in range(dim):
            for j in range(dim):
                x = T[k][i, j]
                if i == 0 and j == 0:
                    s = s + str(x)
                else:
                    s = s + ", " + str(x)
        for i in range(dim):
            x = T[k][i, dim]
            s = s + ", " + str(x)
        s = s + "]"
    
    print("    ")
    print("    # Compute and return final matrix")
    print("    return np.array([" + s + "])")

# Example use, for 3d affine transforms
if __name__ == '__main__':
    affine_inv_diff(3)
