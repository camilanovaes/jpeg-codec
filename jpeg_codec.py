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
encoder  = jpeg.encoder.Encoder(img)
compressed = encoder.process()
header = compressed['header']

# Decoder
decoder  = jpeg.decoder.Decoder(header, compressed, (encoder.width, encoder.height))
decoder.process()

