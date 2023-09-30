import os
import logging
import subprocess
import sys

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


def check(exe_path, tests_dir):
    tests_dir = os.path.abspath(tests_dir)
    exe_path = os.path.abspath(exe_path)
    output_path = os.path.join(tests_dir, 'output.txt')
    for file in os.listdir(tests_dir):
        test = os.path.splitext(file)[0]
        ext = os.path.splitext(file)[1]
        if ext == '.in':
            inp_path = os.path.join(tests_dir, file)
            _logger.info('Checking %s', os.path.basename(file))
            with open(inp_path, 'r') as inp:
                with open(output_path, 'w') as out:
                    subprocess.run(exe_path, stdin=inp, stdout=out)
            check_output(os.path.join(tests_dir, f'{test}.out'), output_path)


if __name__ == '__main__':
    check('.\\build\\Release\\Lab01.exe', '.\\tests')
