import re
import os
import logging
import subprocess

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

_handler = logging.FileHandler(f'{__name__}.log', mode='w')
_formatter = logging.Formatter("%(levelname)s %(message)s %(asctime)s")
_handler.setFormatter(_formatter)
_logger.addHandler(_handler)

time_re = re.compile(r"Elapsed time for call kernel: [-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? [a-zA-Z]")
kernel_re = re.compile(r"< [0-9]+, [0-9]+ >")

_NUMBER_EXEC = 5
_CONFIGURATIONS = [(128, 128), (256, 256), (512, 512), (1024, 1024)]


def _bench(inp_path, output_path, exe_path, blocks, threads):
    info = {'time': float('inf'),
            'time_measure_unit': 'ms',
            'kernel': None
            }
    for i in range(_NUMBER_EXEC):
        with open(inp_path, 'r') as inp:
            with open(output_path, 'w') as out:
                # args = f'Start-Process -NoNewWindow -FilePath "{exe_path}" -ArgumentList "{blocks} {threads}"'
                args = ['powershell.exe', 'Start-Process', '-NoNewWindow', '-FilePath', f'"{exe_path}"',
                        '-ArgumentList', f'"{blocks} {threads}"']
                _logger.debug("Command args: %s", args)
                subprocess.run(args, stdin=inp, stdout=out, shell=True)
            with open(output_path, 'r') as out:
                for line in out:
                    if time_re.match(line):
                        time = float(line.split(': ')[1].split()[0])
                        _logger.debug('Found time: %f', time)
                        info['time'] = min(info['time'], float(time))
                        info['time_measure_unit'] = line.split()[-1]
                    kernel = kernel_re.findall(line)
                    if kernel:
                        split = line[line.find('<<<') + 3: line.find('>>>')].split(', ')
                        info['kernel'] = (int(split[0]), int(split[1]))
                        _logger.debug('Found kernel <<< %d, %d >>>', info['kernel'][0], info['kernel'][1])
    info['time'] /= _NUMBER_EXEC
    return info


def benchmark(exe_path, tests_dir, is_cpu=False):
    for blocks, threads in _CONFIGURATIONS:
        tests_dir = os.path.abspath(tests_dir)
        exe_path = os.path.abspath(exe_path)
        output_path = os.path.join(tests_dir, 'output.txt')
        results_path = os.path.abspath(
            os.path.join(os.path.curdir, 'benchmark' if is_cpu else f'benchmark_{blocks}_{threads}'))
        os.makedirs(results_path, exist_ok=True)
        for file in os.listdir(tests_dir):
            test = os.path.splitext(file)[0]
            ext = os.path.splitext(file)[1]
            if ext == '.in':
                inp_path = os.path.join(tests_dir, file)
                _logger.info('Benchmarking %s', os.path.basename(file))
                info = _bench(inp_path, output_path, exe_path, blocks, threads)
                with open(os.path.join(results_path, f'{test}.txt'), 'w') as out:
                    out.write(str(info))
        if is_cpu:
            break


if __name__ == '__main__':
    benchmark('.\\build\\Debug\\Lab01.exe', '.\\tests')
