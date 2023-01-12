import numpy as np
import scipy as sp
from scipy.spatial import distance

def jaccard_index(ref, flo):
    ref_1d = ref.reshape([ref.size])
    flo_1d = flo.reshape([flo.size])
    
    values = np.unique(np.concatenate((ref_1d, flo_1d)))

    # If no background
    if values[0] > 0:
        values = np.insert(values, 0, 0, axis=0)
    
    jac = np.zeros([values.size+1])

    for i in range(values.size):
        ref_1d_bin = (ref_1d == values[i])
        flo_1d_bin = (flo_1d == values[i])
        jac[i] = distance.jaccard(ref_1d_bin, flo_1d_bin)

    jac[values.size] = distance.jaccard(ref_1d > 0, flo_1d > 0)

    return 1-jac

def main():
    ref_im = np.zeros([4, 4], dtype='int32')
    flo_im = np.zeros([4, 4], dtype='int32')

    ref_im[2, 2] = 1
    flo_im[2, 2] = 1

    ref_im[2, 3] = 2
    flo_im[2, 3] = 3

    ref_im[3, 3] = 3
    flo_im[3, 3] = 2
    
    ref_im[0, 0] = 3
    flo_im[0, 0] = 3

    ref_im[1, 1] = 1

    jac = jaccard_index(ref_im, flo_im)

    print(jac)

if __name__ == '__main__':
    main()

