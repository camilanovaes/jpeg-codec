import numpy as np
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
import cv2

def dct2d(blocks):
    """ Discrete Cosine Transform 2D

    Args:
        blocks :

    """
    return dct(dct(blocks, axis=0, norm = 'ortho'), axis=1, norm = 'ortho')

def idct2d(blocks):
    """ Inverse Discrete Cosine Transform 2D

    Args:
        blocks :

    """
    return idct(idct(blocks, axis=0, norm = 'ortho'), axis=1, norm = 'ortho')

def reconstruct_from_blocks(blocks, H, W):
    """ Inverse Discrete Cosine Transform 2D

    Args:
        blocks :
        H      :
        W      :

    """
    total_lines = []
    N_blocks = int(W / blocks[0].shape[0]) + 1

    for n in range(0, len(blocks) - N_blocks + 1, N_blocks):
        res = np.concatenate(blocks[n:n + N_blocks], axis=1)
        total_lines.append(res)
    
    return np.concatenate(total_lines)

def transform_to_block(image, H, W):
    """ Transform image into N HxW blocks

    Args:
        image : n-dimensional array
        h     : block height
        w     : block width

    """

    trans_image = image.copy()
    n_lines     = 0
    n_columns   = 0
    
    l, c = trans_image.shape

    # Fill image with 0-edge if needed
    # Lines
    while (l % H):
        trans_image = np.vstack((trans_image, np.zeros(trans_image.shape[1])))
        l           = trans_image.shape[0]
        n_lines    += 1 

    # Column
    while (c % W):
        trans_image = np.column_stack((trans_image, np.zeros(trans_image.shape[0])))
        c           = trans_image.shape[1]
        n_columns  += 1

    # Save edges to remove later
    edges = np.array([n_lines, n_columns])

    # Create the blocks
    l, c   = trans_image.shape
    blocks = []

    for i in range(0, l - H + 1, H):
        for j in range(0, c - W + 1, W):
            blocks.append(trans_image[i:i+W,j:j+H])

    return trans_image, blocks, edges

def main():
    # Read image
    img        = cv2.imread('dog.jpg', 0)
    img_line   = img.shape[0]
    img_column = img.shape[1]

    # Convert to float64
    img  = img.astype(np.float64)

    # Create the 8x8 blocks
    trans_img, blocks, edges = transform_to_block(img, 8, 8)

    # Calculate the DCT transform for each block
    dct_res = []
    for b in blocks:
        dct_res.append(dct2d(b))

    # Reconstruct the image
    # Calculate the inverse DCT transform for each block
    idct_res = []
    for r in dct_res:
        idct_res.append(idct2d(r))

    # Reconstruct the image
    reconst_img = reconstruct_from_blocks(idct_res, img_line, img_column)

    # Remove the 0-edges
    for lin in range(edges[0]):
        reconst_img = np.delete(reconst_img, -1, axis=0)
    for col in range(edges[1]):
        reconst_img = np.delete(reconst_img, -1, axis=1)

    # Calculate the error
    error = reconst_img - img
    print(np.mean(error))

    # 
    img         = img.astype(np.uint8)
    reconst_img = reconst_img.astype(np.uint8)

    cv2.imshow('original', img)
    cv2.imshow('new', reconst_img)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
