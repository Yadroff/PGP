"""
Usage:
    gif_conv.py [--help] [--version] [--input-dir=<dir>] [--output=<path>]

Options:
    --help                   Show this screen.
    --version                Show version.
    --input-dir=<dir>        Input directory containing .ppm files.
    --output=<path>          Output path for converted .gif file.
"""
from PIL import Image
from docopt import docopt
import os
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

_handler = logging.FileHandler(f'{__name__}.log', mode='w')
_formatter = logging.Formatter("%(levelname)s %(message)s %(asctime)s")
_handler.setFormatter(_formatter)
_logger.addHandler(_handler)


def converter(input_dir, output_path):
    frames = []
    for filename in os.listdir(input_dir):
        base_name, ext = os.path.splitext(filename)
        if ext == '.ppm':
            frame = Image.open(os.path.abspath(os.path.join(input_dir, filename)))
            frames.append(frame)
    if len(frames) == 0:
        return
    frames[0].save(os.path.abspath(output_path), save_all=True, append_images=frames[1:], optimize=True, duration=100, loop=0)


def main(opts):
    input_dir = opts['--input-dir']
    output_path = opts['--output']
    _logger.info('Input directory: %s', os.path.abspath(input_dir))
    _logger.info('Output GIF path: %s', os.path.abspath(output_path))
    out_dir = os.path.dirname(output_path)
    if not os.path.exists(out_dir):
        _logger.debug('Creating output directory: %s', out_dir)
        os.makedirs(out_dir)
    converter(input_dir, output_path)


if __name__ == "__main__":
    main(docopt(__doc__))
