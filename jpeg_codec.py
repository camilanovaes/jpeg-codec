import argparse
import jpeg.encoder, jpeg.decoder
from PIL import Image

parser = argparse.ArgumentParser(description="JPEG codec")
parser.add_argument('-f', '--filename',
                    help='Image')
args = parser.parse_args()

# Read image
img = Image.open(args.filename)
img_width, img_height = img.size

# Encoder
encoder = jpeg.encoder.Encoder(img)
qnt_coef, img_info, image = encoder.process()

# Decoder
decoder  = jpeg.decoder.Decoder(qnt_coef, img_info, image)
decoder.process()

