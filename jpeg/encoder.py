from PIL import Image
from scipy.fftpack import dct
import numpy as np
import jpeg.utils as utils
import skimage.util


class Encoder():
    def __init__(self, image):
        """JPEG Encoder"""
        self.image = image

    def dct(self, blocks):
        """Discrete Cosine Transform 2D"""
        return dct(dct(blocks, axis=0, norm = 'ortho'), axis=1, norm = 'ortho')

    def quantization(self, G, type):
        """Quantization"""
        if (type == 'l'):
            return(np.divide(G, utils.Q).round().astype(np.int32))
        elif (type == 'c'):
            return(np.divide(G, utils.Q).round().astype(np.int32))
        else:
            raise ValueError("Type choice %s unknown" %(type))

    def process(self):

        # Image width and height
        src_img_width, src_img_height = self.image.size
        print(f'Image: W = {src_img_width}, H = {src_img_height}')

        # Convert to numpy matrix
        src_img_mtx = np.asarray(self.image)

        # Add zero-padding if needed
        if (src_img_width % 8 != 0):
            img_width = src_img_width // 8 * 8 + 8
        else:
            img_width = src_img_width

        if (src_img_height % 8 != 0):
            img_height = src_img_height // 8 * 8 + 8
        else:
            img_height = src_img_height

        # Copy data to new matrix
        img_mtx = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        for y in range(src_img_height):
            for x in range(src_img_width):
                img_mtx[y][x] = src_img_mtx[y][x]

        print(f'New Image size: W = {img_width}, H = {img_height}')

        # Convert 'RGB' to 'YCbCr'
        Y, Cb, Cr = Image.fromarray(img_mtx).convert('YCbCr').split()

        # Convert to numpy array
        Y   = np.asarray(Y).astype(np.int32) - 128
        Cb  = np.asarray(Cb).astype(np.int32) - 128
        Cr  = np.asarray(Cr).astype(np.int32) - 128
        img = np.dstack((Y, Cb, Cr))

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

        return ((Y_qnt, Cb_qnt, Cr_qnt), (img_width, img_height), img)

