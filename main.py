from skimage import io, transform
from matplotlib import pyplot as plt
import numpy as np

def computeH(s, d):
    # 5 points
    known_matrix = np.array([
        [s[0][0], s[0][1], 1, 0, 0, 0, (-d[0][0]*s[0][0]), (-d[0][0]*s[0][1]), -d[0][0]],
        [0, 0, 0, s[0][0], s[0][1], 1, (-d[0][1]*s[0][0]), (-d[0][1]*s[0][1]), -d[0][1]],

        [s[1][0], s[1][1], 1, 0, 0, 0, (-d[1][0]*s[1][0]), (-d[1][0]*s[1][1]), -d[1][0]],
        [0, 0, 0, s[1][0], s[1][1], 1, (-d[1][1]*s[1][0]), (-d[1][1]*s[1][1]), -d[1][1]],

        [s[2][0], s[2][1], 1, 0, 0, 0, (-d[2][0]*s[2][0]), (-d[2][0]*s[2][1]), -d[2][0]],
        [0, 0, 0, s[2][0], s[2][1], 1, (-d[2][1]*s[2][0]), (-d[2][1]*s[2][1]), -d[2][1]],

        [s[3][0], s[3][1], 1, 0, 0, 0, (-d[3][0]*s[3][0]), (-d[3][0]*s[3][1]), -d[3][0]],
        [0, 0, 0, s[3][0], s[3][1], 1, (-d[3][1]*s[3][0]), (-d[3][1]*s[3][1]), -d[3][1]],

        [s[4][0], s[4][1], 1, 0, 0, 0, (-d[4][0]*s[4][0]), (-d[4][0]*s[4][1]), -d[4][0]],
        [0, 0, 0, s[4][0], s[4][1], 1, (-d[4][1]*s[4][0]), (-d[4][1]*s[4][1]), -d[4][1]]
    ])

    v = np.linalg.svd(known_matrix)[2]
   
   # arrange result into 3x3
    H = np.array([
        [v[8][0], v[8][1], v[8][2]],
        [v[8][3], v[8][4], v[8][5]],
        [v[8][6], v[8][7], v[8][8]]
    ])

    # make bottom right value = 1
    H = H / H[2][2]
    print(H)
    return H

def warpImage(im, H):
    tform = transform.ProjectiveTransform(matrix=H)
    tform_im = transform.warp(im, tform.inverse)
    return tform_im

# read in the image
im1 = plt.imread('PTZImages/image1.jpeg')
im2 = plt.imread('PTZImages/image8.jpeg')

# select the points
plt.imshow(im1)
im1_pts = plt.ginput(5)
plt.imshow(im2)
im2_pts = plt.ginput(5)

H = computeH(im1_pts, im2_pts)

imwarped = warpImage(im1, H)

plt.imshow(im1)
plt.show()

plt.imshow(imwarped)
plt.show()

# make mosaic
im1_dim = im1.shape
warped_dim = imwarped.shape

combined_image = np.zeros(((im1_dim[0]+warped_dim[0]), (im1_dim[1]+warped_dim[1]), 3))
print(combined_image.shape)
for x in range(combined_image.shape[0]):
    for y in range(combined_image.shape[1]):
        if x < im1.shape[0] and y < im1.shape[1] and not np.all(im1[x][y]):
            combined_image[x][y] = im1[x][y]
        elif x < imwarped.shape[0] and y < imwarped.shape[1] and not np.all(imwarped[x][y]):
            combined_image[x][y] = imwarped[x][y]

plt.imshow(combined_image)
plt.show()