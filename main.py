from skimage import io
from matplotlib import pyplot as plt
import numpy as np

# name of the input file
imname = 'PTZImages/image1.jpeg'

# read in the image
im = plt.imread(imname)

# select the points
plt.imshow(im)
x = plt.ginput(4)
print(x)




def computeH(s, d):
    # 4 points
    known_matrix = np.array([
        [s[0][0], s[0][1], 1, 0, 0, 0, (-d[0][0]*s[0][0]), (-d[0][0]*s[0][1]), -d[0][0]],
        [0, 0, 0, s[0][0], s[0][1], 1, (-d[0][1]*s[0][0]), (-d[0][1]*s[0][1]), d[0][1]],

        [s[1][0], s[1][1], 1, 0, 0, 0, (-d[1][0]*s[1][0]), (-d[1][0]*s[1][1]), -d[1][0]],
        [0, 0, 0, s[1][0], s[1][1], 1, (-d[1][1]*s[1][0]), (-d[1][1]*s[1][1]), d[1][1]],

        [s[2][0], s[2][1], 1, 0, 0, 0, (-d[2][0]*s[2][0]), (-d[2][0]*s[2][1]), -d[2][0]],
        [0, 0, 0, s[2][0], s[2][1], 1, (-d[2][1]*s[2][0]), (-d[2][1]*s[2][1]), d[2][1]],

        [s[3][0], s[3][1], 1, 0, 0, 0, (-d[3][0]*s[3][0]), (-d[3][0]*s[3][1]), -d[3][0]],
        [0, 0, 0, s[3][0], s[3][1], 1, (-d[3][1]*s[3][0]), (-d[3][1]*s[3][1]), d[3][1]],
    ])

    zero_matrix = np.array([0,0,0,0,0,0,0,0,0])

    return np.linalg.solve(known_matrix, zero_matrix)