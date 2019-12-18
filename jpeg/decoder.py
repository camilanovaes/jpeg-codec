from PIL import Image
from scipy.fftpack import idct
import numpy as np
import jpeg.utils as utils
import skimage.util
import cv2
from cv2 import normalize


class Decoder():
    def __init__(self, entropy, img_info):
        """JPEG Decoder"""
        self.width   = img_info[0]
        self.height  = img_info[1]
        self.Y       = entropy[0]
        self.Cb      = entropy[1]
        self.Cr      = entropy[2]

    def idct(self, blocks):
        """Inverse Discrete Cosine Transform 2D"""
        return idct(idct(blocks, axis=0, norm = 'ortho'), axis=1, norm = 'ortho')

    def dequantization(self, G, type):
        """Dequantization"""
        if (type == 'l'):
            return(np.multiply(G, utils.Q_y))
        elif (type == 'c'):
            return(np.multiply(G, utils.Q_c))
        else:
            raise ValueError("Type choice %s unknown" %(type))

    def upsampling(self, cb, cr, nrow, ncol):
        """Upsampling function

        Args:
            cb   :
            cr   :
            nrow :
            ncol :

        """
        up_cb = cv2.resize(cb, dsize=(ncol, nrow))
        up_cr = cv2.resize(cr, dsize=(ncol, nrow))

        return (up_cb, up_cr)

    def entropy_decoding(self, matrix):
        """Entropy decoding

        Args:
            matrix:

        """
        # Rearrange the image components from "zigzag" order
        entropy_mtx = np.zeros((matrix.shape[0], 8, 8))
        for i, block in enumerate(matrix):
            new_block      = [b for _,b in sorted(zip(utils.zigzag_order, block))]
            new_block      = np.array(new_block).reshape(8,8)
            entropy_mtx[i] = new_block

            if (i != 0):
                entropy_mtx[i][0][0] = matrix[i][0] + matrix[i-1][0]

        return entropy_mtx

    def process(self):

        # Entropy decoder
        self.Y  = self.entropy_decoding(self.Y)
        self.Cb = self.entropy_decoding(self.Cb)
        self.Cr = self.entropy_decoding(self.Cr)

        # Dequantization
        dqnt_Y  = self.dequantization(self.Y, 'l')
        dqnt_Cb = self.dequantization(self.Cb, 'c')
        dqnt_Cr = self.dequantization(self.Cr, 'c')

        # Calculate the inverse DCT transform
        idct_Y  = self.idct(dqnt_Y)
        idct_Cb = self.idct(dqnt_Cb)
        idct_Cr = self.idct(dqnt_Cr)

        # Reconstruct image from blocks
        Y  = utils.reconstruct_from_blocks(idct_Y, self.width)
        Cb = utils.reconstruct_from_blocks(idct_Cb, self.width)
        Cr = utils.reconstruct_from_blocks(idct_Cr, self.width)

        # Upsampling Cb and Cr
        Cb, Cr = self.upsampling(Cb, Cr, self.height, self.width)

        img = np.dstack((Y, Cb, Cr)) + 128.0

        # Normalize and convert to uint8
        #
        # The function np.uint8 considers only the lowest byte of the number,
        # so we need first normalize the image and after that we can convert to
        # uint8.
        # Ref: https://stackoverflow.com/questions/46866586/conversion-of-image-type-int16-to-uint8
        #
        img = normalize(img, 0, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img = Image.fromarray(img, 'YCbCr').convert('RGB')
        img.show()

        return np.asarray(img).astype(np.float64)

