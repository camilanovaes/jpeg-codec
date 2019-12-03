from PIL import Image
from scipy.fftpack import idct
import numpy as np
import jpeg.utils as utils
import skimage.util
import cv2
from cv2 import normalize


class Decoder():
    def __init__(self, qnt_coef, img_info, image):
        """JPEG Decoder"""
        self.img_width  = img_info[0]
        self.img_height = img_info[1]
        self.Y          = qnt_coef[0]
        self.Cb         = qnt_coef[1]
        self.Cr         = qnt_coef[2]
        self.image      = image

    def idct(self, blocks):
        """Inverse Discrete Cosine Transform 2D"""
        return idct(idct(blocks, axis=0, norm = 'ortho'), axis=1, norm = 'ortho')

    def dequantization(self, G, type):
        """Dequantization"""
        if (type == 'l'):
            return(np.multiply(G, utils.Q_c))
        elif (type == 'c'):
            return(np.multiply(G, utils.Q_y))
        else:
            raise ValueError("Type choice %s unknown" %(type))

    def process(self):

        # Dequantization
        dqnt_Y  = self.dequantization(self.Y, 'l')
        dqnt_Cb = self.dequantization(self.Cb, 'c')
        dqnt_Cr = self.dequantization(self.Cr, 'c')

        # Calculate the inverse DCT transform
        idct_Y  = self.idct(dqnt_Y)
        idct_Cb = self.idct(dqnt_Cb)
        idct_Cr = self.idct(dqnt_Cr)

        # Reconstruct image from blocks
        Y  = utils.reconstruct_from_blocks(idct_Y, self.img_width)
        Cb = utils.reconstruct_from_blocks(idct_Cb, self.img_width)
        Cr = utils.reconstruct_from_blocks(idct_Cr, self.img_width)

        img = np.dstack((Y, Cb, Cr)) + 128

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

