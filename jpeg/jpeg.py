import numpy as np
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
import cv2

#TODO: put Q in the right place
Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
              [12, 12, 14, 19, 26, 58, 60, 55],
              [14, 13, 16, 24, 40, 57, 69, 56],
              [14, 17, 22, 29, 51, 87, 80, 62],
              [18, 22, 37, 56, 68, 109, 103, 77],
              [24, 35, 55, 64, 81, 104, 113, 92],
              [49, 64, 78, 87, 103, 121, 120, 101],
              [72, 92, 95, 98, 112, 100, 103, 99]])


class JPEG():
    def __init__(self, filename):
        """ JPEG encoder / decoder

        Args:
            filename :

        """
        self.filename = filename

    def _dct2d(self, blocks):
        """ Discrete Cosine Transform 2D

        Args:
            blocks :

        """
        return dct(dct(blocks, axis=0, norm = 'ortho'), axis=1, norm = 'ortho')

    def _idct2d(self, blocks):
        """ Inverse Discrete Cosine Transform 2D

        Args:
            blocks :

        """
        return idct(idct(blocks, axis=0, norm = 'ortho'), axis=1, norm = 'ortho')

    def _reconstruct_from_blocks(self, blocks, H, W):
        """ Inverse Discrete Cosine Transform 2D

        Args:
            blocks :
            H      :
            W      :

        """
        total_lines = []
        N_blocks    = int(W / blocks[0].shape[0]) + 1

        for n in range(0, len(blocks) - N_blocks + 1, N_blocks):
            res = np.concatenate(blocks[n:n + N_blocks], axis=1)
            total_lines.append(res)

        return np.concatenate(total_lines)

    def _transform_to_block(self, image, H, W):
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

    def _quantization(self, G):
        """

        Args:
            G : DCT coefficients

        Return:

        """
        return(np.divide(G, Q))

    def _dequantization(self, G):
        """

        Args:
            G :

        Return:

        """
        return(np.multiply(G, Q))


    def process(self):
        # Read image
        img        = cv2.imread('dog.jpg', 0)
        img_line   = img.shape[0]
        img_column = img.shape[1]

        # Encoding
        # Convert to float64
        img  = img.astype(np.float64)
        # Create the 8x8 blocks
        trans_img, blocks, edges = self._transform_to_block(img, 8, 8)
        # Calculate the DCT transform
        dct_res     = self._dct2d(blocks)
        # Quantization
        qnt_res     = self._quantization(dct_res)

        # Decoding
        # Dequantization
        dqnt_res    = self._dequantization(qnt_res)
        # Calculate the inverse DCT transform
        idct_res    = self._idct2d(dqnt_res)
        # Reconstruct the image
        reconst_img = self._reconstruct_from_blocks(idct_res, img_line, img_column)
        # Remove the 0-edges
        # TODO: Move to reconstruct_from_blocks function
        for lin in range(edges[0]):
            reconst_img = np.delete(reconst_img, -1, axis=0)
        for col in range(edges[1]):
            reconst_img = np.delete(reconst_img, -1, axis=1)

        # Calculate the error
        error = reconst_img - img
        print(f"Error: {np.mean(error)}")

        # Convert to uint8
        img         = img.astype(np.uint8)
        reconst_img = reconst_img.astype(np.uint8)

        cv2.imshow('original', img)
        cv2.imshow('new', reconst_img)
        cv2.waitKey(0)
