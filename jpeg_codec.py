import argparse
import jpeg.encoder, jpeg.decoder
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description="JPEG codec")
parser.add_argument('-f', '--filename',
                    help='Image')
args = parser.parse_args()

# Read image
img     = Image.open(args.filename)
src_img = np.asarray(img).astype(np.float64)

img_width, img_height = img.size

# Encoder
encoder    = jpeg.encoder.Encoder(img)
compressed = encoder.process()
header     = compressed['header']

# Decoder
decoder     = jpeg.decoder.Decoder(src_img, header, compressed,
                                   (encoder.width, encoder.height))
decoded_img = decoder.process()

# Print image and calculate the error
img       = np.asarray(decoded_img).copy()
img_final = img[:img_height, :img_width, :]

print(f'Error: {np.mean(src_img - img_final.astype(np.float64))}')

img_final = Image.fromarray(img_final)
img_final.show()

