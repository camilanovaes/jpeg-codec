from PIL import Image
from scipy.fftpack import dct, idct
import numpy as np
import jpeg.utils as utils
import skimage.util


class Encoder():
    def __init__(self, image):
        """JPEG Encoder"""
        self.image  = image
        self.width  = None
        self.height = None

    def dct(self, blocks):
        """Discrete Cosine Transform 2D"""
        return dct(dct(blocks, axis=0, norm = 'ortho'), axis=1, norm = 'ortho')

    def quantization(self, G, type):
        """Quantization"""
        if (type == 'l'):
            return(np.divide(G, utils.Q_y).round().astype(np.float64))
        elif (type == 'c'):
            return(np.divide(G, utils.Q_c).round().astype(np.float64))
        else:
            raise ValueError("Type choice %s unknown" %(type))

    def downsampling(self, img_ycbcr, nrow, ncol, k=2, type=0):
        """ Downsamplig function

        Args:
            img_ycbcr : Image matrix with 3 channels: Y, Cb and Cr
            nrow      : Number of rows
            ncol      : Number of columns
            k         : Downsampling reduction factor
            type      : Downsampling types. Type 0, represents no downsampling.
                        Type 1, represents columns reduction, and type 2, rows
                        and columns reduction.

        Returns:
            {Cr,Cb}: tuple

        """
        col_d = np.arange(k,ncol, step=k+1)
        row_d  = np.arange(k, nrow, step=k+1)

        if type == 1:
            ds_img = np.delete(img_ycbcr, col_d, axis=1)
        elif type == 2:
            ds_img = np.delete(img_ycbcr, col_d, axis=1)
            ds_img = np.delete(ds_img, row_d, axis=0)
        else:
            ds_img = img_ycbcr

        return ds_img[:,:,1],ds_img[:,:,2]

    def process(self):

        # Image width and height
        src_img_height, src_img_width = self.image.size
        print(f'Image: H = {src_img_height}, W = {src_img_width}')

        # Convert to numpy matrix
        src_img_mtx = np.asarray(self.image)

        # Convert 'RGB' to 'YCbCr'
        img_ycbcr = Image.fromarray(src_img_mtx).convert('YCbCr')
        img_ycbcr = np.asarray(img_ycbcr).astype(np.float64)

        # Apply downsampling to Cb and Cr
        Cb, Cr = self.downsampling(img_ycbcr, src_img_height, src_img_width)

        # Convert to numpy array
        Y   = img_ycbcr[:,:,0] - 128
        Cb  = Cb - 128
        Cr  = Cr - 128

        # Add zero-padding if needed
        Y  = utils.zero_padding(Y)
        Cb = utils.zero_padding(Cb)
        Cr = utils.zero_padding(Cr)

        # Save new size
        self.height, self.width = Y.shape

        # Transform channels into blocks
        Y_bck  = utils.transform_to_block(Y)
        Cb_bck = utils.transform_to_block(Cb)
        Cr_bck = utils.transform_to_block(Cr)

        # Calculate the DCT transform
        Y_dct  = self.dct(Y_bck)
        Cb_dct = self.dct(Cb_bck)
        Cr_dct = self.dct(Cr_bck)

        # Quantization
        Y_qnt  = self.quantization(Y_dct, 'l')
        Cb_qnt = self.quantization(Cb_dct, 'c')
        Cr_qnt = self.quantization(Cr_dct, 'c')

        return (Y_qnt, Cb_qnt, Cr_qnt)

