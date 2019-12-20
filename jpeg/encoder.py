from PIL import Image
from scipy.fftpack import dct, idct
import numpy as np
import jpeg.utils as utils
import skimage.util
from bitarray import bitarray, bits2bytes
from .huffman import H_Encoder, H_Decoder, DC, AC, LUMINANCE, CHROMINANCE

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

    def downsampling(self, matrix, k=2, type=2):
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
            Downsampling matrix

        """
        if type == 1:
            ds_img = matrix[:,0::k]
        elif type == 2:
            ds_img = matrix[0::k,0::k]
        else:
            ds_img = matrix

        return ds_img

    def process(self):

        # Image width and height
        src_img_height, src_img_width = self.image.size
        print(f'Image: H = {src_img_height}, W = {src_img_width}')

        # Convert to numpy matrix
        src_img_mtx = np.asarray(self.image)

        # Convert 'RGB' to 'YCbCr'
        img_ycbcr = Image.fromarray(src_img_mtx).convert('YCbCr')
        img_ycbcr = np.asarray(img_ycbcr).astype(np.float64)

        # Convert to numpy array
        Y   = img_ycbcr[:,:,0] - 128
        Cb  = img_ycbcr[:,:,1] - 128
        Cr  = img_ycbcr[:,:,2] - 128

        # Apply downsampling to Cb and Cr
        Cb = self.downsampling(Cb, src_img_height, src_img_width)
        Cr = self.downsampling(Cr, src_img_height, src_img_width)

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

        # Entropy Encoder
        encoded = {
            LUMINANCE: H_Encoder(Y_qnt, LUMINANCE).encode(),
            CHROMINANCE: H_Encoder(
                np.vstack((Cb_qnt, Cr_qnt)),
                CHROMINANCE
            ).encode()
        }

        # Combine RGB data as binary in the order:
        #   LUMINANCE.DC, LUMINANCE.AC, CHROMINANCE.DC, CHROMINANCE.AC
        order = (encoded[LUMINANCE][DC], encoded[LUMINANCE][AC],
                 encoded[CHROMINANCE][DC], encoded[CHROMINANCE][AC])

        bits = bitarray(''.join(order))

        with open('encoded_img.bin','wb') as f:
            bits.tofile(f)

        return {
            'data': bits,
            'header': {
                # Remaining bits length is the fake filled bits for 8 bits as a
                # byte.
                'remaining_bits_length': bits2bytes(len(bits)) * 8 - len(bits),
                'data_slice_lengths': tuple(len(d) for d in order)
        }
    }

