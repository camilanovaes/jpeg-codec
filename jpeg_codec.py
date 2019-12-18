import argparse
import jpeg.encoder, jpeg.decoder
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description="JPEG codec")
parser.add_argument('-f', '--filename',
                    help='Image')
args = parser.parse_args()

# Read image
img = Image.open(args.filename)
src_img = np.asarray(img).astype(np.float64)
img_width, img_height = img.size

# Encoder
encoder = jpeg.encoder.Encoder(img)
entropy = encoder.process()

# Decoder
decoder  = jpeg.decoder.Decoder(entropy, (encoder.width, encoder.height))
decoded_img = decoder.process()

decoded_img.resize(img_height, img_width, 3)
print(f'Error: {np.mean(src_img - decoded_img)}')

