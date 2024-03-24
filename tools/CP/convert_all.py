"""
Usage:
    convert_all.py [--help] [--input=<dir>] [--output=<path>]

Options:
    --help                   Show this screen.
    --input=<dir>            Input directory containing .data files.
    --output=<dir>           Output directory for converted image file.
"""

from docopt import docopt
import os
import logging
import conv

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

_handler = logging.FileHandler(f'{__name__}.log', mode='w')
_formatter = logging.Formatter("%(levelname)s %(message)s %(asctime)s")
_handler.setFormatter(_formatter)
_logger.addHandler(_handler)


def convert(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        file, ext = os.path.splitext(filename)
        if ext == '.data':
            _logger.info('Converting: %s', os.path.join(input_dir, filename))
            conv.to_img_alfa(os.path.join(input_dir, filename), os.path.join(output_dir, file + '.png'))


def main(args):
    input_dir = os.path.abspath(args['--input'])
    output_dir = os.path.abspath(args['--output'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        _logger.debug('Created output directory: %s', output_dir)
    convert(input_dir, output_dir)


if __name__ == '__main__':
    main(docopt(__doc__))
