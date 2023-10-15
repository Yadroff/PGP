import os
import logging
import subprocess
import sys

import conv

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

_handler = logging.FileHandler(f'{__name__}.log', mode='w')
_formatter = logging.Formatter("%(levelname)s %(message)s %(asctime)s")
_handler.setFormatter(_formatter)
_logger.addHandler(_handler)


def get_floats(file_path):
    result = []
    with open(file_path, 'r') as file:
        data = file.read().split()
    for number in data:
        try:
            result.append(float(number))
        except ValueError:
            pass
    return result


def check_output(ans_path, output_path):
    ans_floats = get_floats(ans_path)
    output_floats = get_floats(output_path)
    if len(ans_floats) != len(output_floats):
        _logger.error('Check size of floats %s and output', os.path.basename(ans_path))
        sys.exit(0)
    for i in range(len(ans_floats)):
        ans = ans_floats[i]
        out = output_floats[i]
        if abs(ans - out) > 1e-10:
            _logger.error('WA: Excepted: %f Founded: %f', ans, out)
            sys.exit(0)


def check(exe_path, images_dir, data_dir):
    images_dir = os.path.abspath(images_dir)
    data_dir = os.path.abspath(data_dir)
    exe_path = os.path.abspath(exe_path)
    _logger.info('Check configuration:')
    _logger.info(' * Exe: %s', exe_path)
    _logger.info(' * Images: %s', images_dir)
    _logger.info(' * Data: %s', data_dir)
    os.makedirs(os.path.join(images_dir, 'output'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'input'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'output'), exist_ok=True)

    images_inp = os.path.join(images_dir, 'input')
    for file in os.listdir(images_inp):
        name = os.path.basename(file)
        name = os.path.splitext(name)[0]
        name_data = name + '.data'
        _logger.info('Basename: %s', name)
        input_data = os.path.join(data_dir, 'input', name_data)
        output_data = os.path.join(data_dir, 'output', name_data)
        output_image = os.path.join(images_dir, 'output', name + '.png')
        _logger.info('Converting %s', file)
        conv.from_img(os.path.join(images_inp, file), input_data)
        _logger.info('Creating input file: %s')
        with open('input.txt', 'w') as inp:
            inp.write(f'{input_data}\n{output_data}\n')
        _logger.info('Checking %s', os.path.basename(file))
        with open('input.txt', 'r') as inp:
            if not subprocess.call(exe_path, stdin=inp):
                conv.to_img(output_data, output_image)


if __name__ == '__main__':
    check('.\\build\\Release\\Lab02.exe', '.\\images', '.\\data')
