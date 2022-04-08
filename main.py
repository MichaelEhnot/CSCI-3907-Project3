from pickletools import uint8
from skimage import io, transform
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

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

# padding method taken from https://ai-pool.com/d/padding-images-with-numpy
def pad(img, h, w):
    #  in case when you have odd number
    top_pad = np.floor((h - img.shape[0]) / 2).astype(np.uint16)
    bottom_pad = np.ceil((h - img.shape[0]) / 2).astype(np.uint16)
    right_pad = np.ceil((w - img.shape[1]) / 2).astype(np.uint16)
    left_pad = np.floor((w - img.shape[1]) / 2).astype(np.uint16)
    return np.copy(np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant', constant_values=0))

# read in the image
im1 = plt.imread('PTZImages/image7.jpeg')
im2 = plt.imread('PTZImages/image8.jpeg')


im1 = pad(im1, 500, 1000)
im2 = pad(im2, 500, 1000)


# select the points
plt.imshow(im1)
im1_pts = plt.ginput(5)
plt.imshow(im2)
im2_pts = plt.ginput(5)

H = computeH(im1_pts, im2_pts)

imwarped = warpImage(im1, H)

plt.imshow(imwarped)
plt.show()


io.imsave("tmp_imwarped.jpg", imwarped)
io.imsave("tmp_im2.jpg", im2)


im2_m = Image.open('tmp_im2.jpg')
imwarped_m = Image.open('tmp_imwarped.jpg')


combined_image = Image.blend(im2_m, imwarped_m, 0.5)

plt.imshow(combined_image)
plt.show()

combined_image.save("result.jpg")