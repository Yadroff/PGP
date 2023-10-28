import os
import re
import logging
import sys
import subprocess

_DEFAULT_VS_PATH = R'C:\Program Files\Microsoft Visual Studio\2022\Community'

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

_handler = logging.FileHandler(f'{__name__}.log', mode='w')
_formatter = logging.Formatter("%(levelname)s %(message)s %(asctime)s")
_handler.setFormatter(_formatter)
_logger.addHandler(_handler)


def _get_vs_path():
    paths = os.getenv('PATH').split(';')
    for path in paths:
        if re.match(r".*Visual Studio.*", path):
            path = os.path.join(path.split('Community')[0], 'Community')
            _logger.info('Found Visual Studio at: %s', path)
            return path
    if os.path.exists(_DEFAULT_VS_PATH):
        _logger.info('Found Visual Studio at: %s', _DEFAULT_VS_PATH)
        return _DEFAULT_VS_PATH
    raise Exception('Can not find VS')


def _get_ms_build_path():
    try:
        vs_path = _get_vs_path()
        path = os.path.join(vs_path, 'MSBuild', 'Current', 'Bin', 'MSBuild.exe')
        _logger.debug('Try find MSBuild at: %s', path)
        if not os.path.exists(path):
            raise Exception('Can not find MSBuild')
        _logger.info('MSBuild path: %s', path)
        return path
    except Exception as error:
        _logger.exception(error.args)
        sys.exit(-1)


def build(solution, configuration, build_path):
    solution = os.path.abspath(solution)
    _logger.debug('Solution path: %s', solution)
    path = os.path.abspath(os.path.join(build_path, configuration))
    _logger.debug('Try to create path: %s', path)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(os.curdir, f'{configuration}.txt'), 'w', encoding='utf-16') as out:
        args = f'"-property:Configuration={configuration};OutDir={path}"'
        _logger.debug('MSBuild args: %s', args)
        exe_path = _get_ms_build_path()
        command = f'"{exe_path}" "{solution}" {args}'
        _logger.debug('Try to execute: %s', command)
        subprocess.call(command, stdout=out, text=True)
        _logger.debug(out.encoding)
    exe_path = os.path.join(build_path, configuration, os.path.splitext(os.path.basename(solution))[0] + '.exe')
    return os.path.abspath(exe_path)


if __name__ == '__main__':
    # build('..\\..\\src\\GPU\\Lab03.sln', 'Release', '.\\build')
    build('..\\..\\src\\GPU\\Lab03\\Lab03.sln', 'Profile', '.\\build')
