from skimage import io, transform
from matplotlib import pyplot as plt
import numpy as np

def computeH(s, d):
    # 5 points
    known_matrix = np.array([
        [s[0][0], s[0][1], 1, 0, 0, 0, (-d[0][0]*s[0][0]), (-d[0][0]*s[0][1]), -d[0][0]],
        [0, 0, 0, s[0][0], s[0][1], 1, (-d[0][1]*s[0][0]), (-d[0][1]*s[0][1]), d[0][1]],

        [s[1][0], s[1][1], 1, 0, 0, 0, (-d[1][0]*s[1][0]), (-d[1][0]*s[1][1]), -d[1][0]],
        [0, 0, 0, s[1][0], s[1][1], 1, (-d[1][1]*s[1][0]), (-d[1][1]*s[1][1]), d[1][1]],

        [s[2][0], s[2][1], 1, 0, 0, 0, (-d[2][0]*s[2][0]), (-d[2][0]*s[2][1]), -d[2][0]],
        [0, 0, 0, s[2][0], s[2][1], 1, (-d[2][1]*s[2][0]), (-d[2][1]*s[2][1]), d[2][1]],

        [s[3][0], s[3][1], 1, 0, 0, 0, (-d[3][0]*s[3][0]), (-d[3][0]*s[3][1]), -d[3][0]],
        [0, 0, 0, s[3][0], s[3][1], 1, (-d[3][1]*s[3][0]), (-d[3][1]*s[3][1]), d[3][1]],

        [s[4][0], s[4][1], 1, 0, 0, 0, (-d[4][0]*s[4][0]), (-d[4][0]*s[4][1]), -d[4][0]],
        [0, 0, 0, s[4][0], s[4][1], 1, (-d[4][1]*s[4][0]), (-d[4][1]*s[4][1]), d[4][1]]
    ])

    zero_matrix = np.array([0, 0, 0, 0, 0, 0, 0, 0])

    h = np.linalg.lstsq(known_matrix, zero_matrix)[0]
    print("HELLO")
    print(h)
    H = np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1]
    ])

    return H

def warpImage(im, H):
    tform = transform.ProjectiveTransform(matrix=H)
    tform_im = transform.warp(im, tform.inverse)
    return tform_im

# read in the image
im1 = plt.imread('PTZImages/image1.jpeg')
im2 = plt.imread('PTZImages/image2.jpeg')

# select the points
plt.imshow(im1)
im1_pts = plt.ginput(5)
plt.imshow(im2)
im2_pts = plt.ginput(5)

H = computeH(im1_pts, im2_pts)
print(H)

imwarped = warpImage(im2, H)

plt.imshow(imwarped)