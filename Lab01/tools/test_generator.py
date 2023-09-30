import os
import random
import logging
import subprocess

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

_handler = logging.FileHandler(f'{__name__}.log', mode='w')
_formatter = logging.Formatter("%(levelname)s %(message)s %(asctime)s")
_handler.setFormatter(_formatter)
_logger.addHandler(_handler)


def generate_double():
    return round(random.uniform(-1e10, 1e10), 10)


def generate_test(n, file_name):
    with open(file_name, 'w') as file:
        file.write(f'{n}\n')
        for i in range(n):
            file.write(f'{generate_double()} ')
        file.write('\n')
        for i in range(n):
            file.write(f'{generate_double()} ')
        file.write('\n')


def generate_tests(tests_dir):
    tests_dir = os.path.abspath(tests_dir)
    os.makedirs(tests_dir, exist_ok=True)
    inputs = [10, 1000, 10_000, 100_000, 1_000_000]
    for i in range(0, len(inputs)):
        generate_test(inputs[i], os.path.join(tests_dir, f'{i}.in'))


def generate_answers(exe_path, tests_dir):
    tests_dir = os.path.abspath(tests_dir)
    exe_path = os.path.abspath(exe_path)
    for file in os.listdir(tests_dir):
        test = os.path.splitext(file)[0]
        ext = os.path.splitext(file)[1]
        if ext == '.in':
            out_path = os.path.join(tests_dir, f'{test}.out')
            inp_path = os.path.join(tests_dir, file)
            _logger.info('Generate %s', os.path.basename(out_path))
            with open(inp_path, 'r') as inp:
                with open(out_path, 'w') as out:
                    subprocess.run(exe_path, stdin=inp, stdout=out)


if __name__ == '__main__':
    generate_tests('.\\tests')
    generate_answers('..\\cpu_realization.exe', '.\\tests')
