"""
Tool that builds .sln, tests and benchmark it

Usage:
  conv.py (--image=img) (--output=data | -O DATA) [--alpha | -A]
  conv.py (--input=data_inp) (--output=data | -O DATA) [--alpha | -A]
  conv.py (-h | --help)

Options:
    -h --help               Show this screen.
    --image=img -I IMG      Input image to convert to .data
    --output=data -O DATA   Output data (.png or .data)
    --input=data_inp            Input data to convert to image
    --alpha -A              Use alpha channel
"""
import os.path
import sys
import struct
from PIL import Image
import ctypes
from docopt import docopt
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

_handler = logging.FileHandler(f'{__name__}.log', mode='w')
_formatter = logging.Formatter("%(levelname)s %(message)s %(asctime)s")
_handler.setFormatter(_formatter)
_logger.addHandler(_handler)
pal = [
    (0, 0, 0), (128, 128, 128), (192, 192, 192), (255, 255, 255),
    (255, 0, 255), (128, 0, 128), (255, 0, 0), (128, 0, 0),
    (205, 92, 92), (240, 128, 128), (250, 128, 114), (233, 150, 122),
    (205, 92, 92), (240, 128, 128), (250, 128, 114), (233, 150, 122),
    (173, 255, 47), (127, 255, 0), (124, 252, 0), (0, 255, 0),
    (50, 205, 50), (152, 251, 152), (144, 238, 144), (0, 250, 154),
    (0, 255, 127), (60, 179, 113), (46, 139, 87), (34, 139, 34),
    (0, 128, 0), (0, 100, 0), (154, 205, 50), (107, 142, 35),
    (128, 128, 0), (85, 107, 47), (102, 205, 170), (143, 188, 143),
    (32, 178, 170), (0, 139, 139), (0, 128, 128)
]


def to_img_alfa(src, dst=None):
    fin = open(src, 'rb')
    (w, h) = struct.unpack('hi', fin.read(8))
    buff = ctypes.create_string_buffer(4 * w * h)
    fin.readinto(buff)
    fin.close()
    img = Image.new('RGB', (w, h))
    pix = img.load()
    offset = 0
    sp = len(pal)
    for j in range(h):
        for i in range(w):
            (_, _, _, a) = struct.unpack_from('cccc', buff, offset)
            pix[i, j] = pal[ord(a) % sp]
            offset += 4
    if dst:
        img.save(dst)
    else:
        img.show()


def to_img(src, dst=None):
    fin = open(src, 'rb')
    (w, h) = struct.unpack('hi', fin.read(8))
    buff = ctypes.create_string_buffer(4 * w * h)
    fin.readinto(buff)
    fin.close()
    img = Image.new('RGBA', (w, h))
    pix = img.load()
    offset = 0
    for j in range(h):
        for i in range(w):
            (r, g, b, a) = struct.unpack_from('cccc', buff, offset)
            pix[i, j] = (ord(r), ord(g), ord(b), ord(a))
            offset += 4
    if dst:
        img.save(dst)
    else:
        img.show()


def from_img(src, dst):
    img = Image.open(src)
    (w, h) = img.size[0:2]
    pix = img.load()
    buff = ctypes.create_string_buffer(4 * w * h)
    offset = 0
    for j in range(h):
        for i in range(w):
            r = bytes((pix[i, j][0],))
            g = bytes((pix[i, j][1],))
            b = bytes((pix[i, j][2],))
            a = bytes((255,))
            struct.pack_into('cccc', buff, offset, r, g, b, a)
            offset += 4;
    out = open(dst, 'wb')
    out.write(struct.pack('ii', w, h))
    out.write(buff.raw)
    out.close()


def main(opts):
    image = opts['--image']
    data = opts['--output']
    img_to_data = True
    if not image:
        image = opts['--output']
        data = opts['--input']
        img_to_data = False
    image = os.path.abspath(image)
    data = os.path.abspath(data)
    alpha = bool(opts['--alpha'])
    _logger.info('Running tool with this cmd options')
    _logger.info(' * Image: %s', image)
    _logger.info(' * Data: %s', data)
    _logger.info(' * Alpha: %s', alpha)
    _logger.info(' * Mode: %s', 'Img to data' if img_to_data else 'Data to img')

    if img_to_data:
        from_img(image, data)
    elif alpha:
        to_img_alfa(data, image)
    else:
        to_img(data, image)


if __name__ == '__main__':
    main(docopt(__doc__))
