import numpy as np
import math

# class that calculates the cosine similarity (normalized dot product of 2 matrices v1 and v2)
class Cosim():

    # v1 and v2 are np arrays

    def __init__(self):
        return None

    def sanityCheck(self, v1, v2):
        return v1.shape == v2.shape
    
    def v_norm(self, v1):

        sum_sq = 0

        if (v1.ndim == 1):
            for i in range(v1.shape[0]):
                sum_sq += v1[i]*v1[i]

            return math.sqrt(sum_sq)

        for i in range(v1.shape[0]):
            for j in range(v1.shape[1]):
                sum_sq += v1[i][j]*v1[i][j]

        rt_sum_sq = math.sqrt(sum_sq)

        return rt_sum_sq
    
    def dot_prod(self, v1, v2):

        sum_rowdot = 0

        if (v1.ndim == 1):
            for i in range(v1.shape[0]):
                sum_rowdot += v1[i]*v2[i]

            return sum_rowdot
        
        for i in range(v1.shape[0]):
            row_dot = 0
            for j in range(v1.shape[1]):
                row_dot += v1[i][j]*v2[i][j]
            sum_rowdot += row_dot

        return sum_rowdot

    def sim(self, v1, v2):

        # sanity check
        if (v1.shape != v2.shape):
            print("ERROR: Matrix dimensions do not match!")
            return None
        
        # calculate the normalization constants
        nv1 = self.v_norm(v1)
        nv2 = self.v_norm(v2)

        dot_product = self.dot_prod(v1, v2)

        if ((nv1 == 0) or (nv2 == 0)):
            return 0

        return float(dot_product) / (nv1*nv2)

if (__name__ == '__main__'):

    # sample code to see how this works
    cosim = Cosim()

    # make sure the np array has dtype float64
    v1 = np.array([[1,2,3,4], [1,2,3,4]], dtype = 'float64')
    v2 = np.array([[1,2,3,4], [1,2,3,4]], dtype = 'float64')
    print(cosim.sim(v1, v2))

    v1 = np.array([[1,2,3,4], [5,6,7,8], [1,2,3,4], [5,6,7,8], [1,2,3,4], [1,2,3,4]], dtype = 'float64')
    v2 = np.array([[1,2,3,4], [5,6,7,8], [1,2,3,4], [5,6,7,8], [1,2,3,4], [1,2,3,4]], dtype = 'float64')
    print(cosim.sim(v1, v2))

    v3 = np.array([1,2,3,4], dtype = 'float64')
    v4 = np.array([1,2,3,4], dtype = 'float64')
    print(cosim.sim(v3, v4))