import numpy as np


# Quantization tables
Q_y = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99]])

Q_c = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99]])

zigzag_order = np.array([0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,12,19,26,33,
                         40,48,41,34,27,20,13,6,7,14,21,28,35,42,49,56,57,50,
                         43,36,29,22,15,23,30,37,44,51,58,59,52,45,38,31,39,
                         46,53,60,61,54,47,55,62,63])

# General Functions
def reconstruct_from_blocks(blocks, img_width):
    """ Inverse Discrete Cosine Transform 2D

    Args:
        blocks     :
        img_width  :
        img_height :
        H          :
        W          :

    """

    total_lines = []
    N_blocks    = int(img_width / 8)

    for n in range(0, len(blocks) - N_blocks + 1, N_blocks):
        res = np.concatenate(blocks[n : n + N_blocks], axis=1)
        total_lines.append(res)

    return np.concatenate(total_lines)

def transform_to_block(image):
    """ Transform image into N 8x8 blocks

    Args:
        image : n-dimensional array

    """

    img_w, img_h = image.shape
    blocks = []
    for i in range(0, img_w, 8):
        for j in range(0, img_h, 8):
            blocks.append(image[i:i+8,j:j+8])

    return blocks

def zero_padding(matrix):
    """ Add zero-padding

    Args:
        matrix :

    """
    ncol, nrow = matrix.shape[0], matrix.shape[1]

    if (ncol % 8 != 0):
        img_width = ncol // 8 * 8 + 8
    else:
        img_width = ncol

    if (nrow % 8 != 0):
        img_height = nrow // 8 * 8 + 8
    else:
        img_height = nrow

    # Copy data to new matrix
    new_mtx = np.zeros((img_width, img_height), dtype=np.float64)
    for y in range(ncol):
        for x in range(nrow):
            new_mtx[y][x] = matrix[y][x]

    return new_mtx

