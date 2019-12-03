# JPEG Codec

## Usage

``` bash
python jpeg_codec.py -f dog.jpg
```

## Encoder
The encoding process consists of several steps:

#### 1. Color space transformation: RGB to YCbCr
The representation of the colors in the image is converted from RGB to a color space called YCbCr. Is has three components Y, Cb and Cr. The Y component represents the **brightness** of a pixel, and the Cb and Cr components represent the **chrominance** (split into blue and red components). The YCbCr color space conversion allows greater compression without a significant effect on perceptual image quality off the image.

#### 2. Downsampling of chrominance components
The resolution of the chroma data is reduced, usually by a factor of 2 or 3 (downsampling). This reflects the fact that the humans can see considerably more fine detail in the brightness of an image (the Y component) than in the hue and color saturation of an image (Cb and Cr components). The ratios at which the downsampling is ordinarily done for JPEG images are:
- 4:4:4 : No downsampling
- 4:2:2 : Reduction by a factor of 2 in the horizontal direction
- 4:2:0 : Reduction by a factor of 2 in both the horizontal and vertical directions

#### 3. DCT
The image is split into blocks of 8x8 pixels, and for each block, each of the Y, Cb and Cr data undergoes the discrete cosine transform (DCT).

#### 4. Quantization
The amplitudes of the frequency components are quantized.

## Decoder
> TODO: Write
