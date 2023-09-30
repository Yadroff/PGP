"""Tool that builds .sln, tests and benchmark it

Usage:
  main.py [--build | -B] [--solution=sln | -S SLN] [--tests | -T] [--check | -C] [--benchmark] [--cpu_exe=EXE] [--benchmarked_cpu=EXE]
  main.py (-h | --help)

Options:
  -h --help                     Show this screen.
  --build -B                    Disable build .sln
  --solution=SLN -S SLN         Path to directory with .sln [default: ..\\gpu\\Lab01.sln]
  --tests -T                    Disable generate tests
  --check -C                    Check tests
  --benchmark                   Disable benchmark .sln
  --cpu_exe=EXE                 Path to cpu realization .exe [default: ..\\cpu\\cmake-build-release\\cpu_realization.exe]
  --benchmarked_cpu=EXE         Path to cpu realization with benchmark [default: ..\\cpu\\cmake-build-release\\benchmarked_cpu_realization.exe]

"""
import os.path

from docopt import docopt
import logging

# custom files
import test_generator
import builder
import checker
import benchmark as bench

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

_handler = logging.FileHandler(f'{__name__}.log', mode='w')
_formatter = logging.Formatter("%(levelname)s %(message)s %(asctime)s")
_handler.setFormatter(_formatter)
_logger.addHandler(_handler)


def main(opts):
    release_exe = None
    profile_exe = None

    benchmark = not opts['--benchmark']
    benchmark_cpu_path = os.path.abspath(opts['--benchmarked_cpu'])
    cpu_exe_path = os.path.abspath(opts['--cpu_exe'])
    build = not opts['--build']
    check = opts['--check']
    tests = opts['--tests']
    solution_path = os.path.abspath(opts['--solution'])
    tests_dir = os.path.abspath('.\\tests')
    build_dir = os.path.abspath('.\\build')

    build = build or check or benchmark
    tests = tests or check or benchmark

    _logger.info('Running tool with this cmd options')
    _logger.info(' * Benchmark: %s', benchmark)
    _logger.info(' * Benchmark CPU exe path: %s', benchmark_cpu_path)
    _logger.info(' * CPU exe path: %s', cpu_exe_path)
    _logger.info(' * Build solution: %s', build)
    _logger.info(' * Solution path: %s', solution_path)
    _logger.info(' * Generate tests: %s', tests)
    _logger.info(' * Check tests: %s', check)

    if build:
        _logger.info('Make build start')
        profile_exe = builder.build(solution_path, 'Profile', build_dir)
        _logger.debug('Profile exe path: %s', profile_exe)
        release_exe = builder.build(solution_path, 'Release', build_dir)
        _logger.debug('Release exe path: %s', release_exe)
        _logger.info('Make build finish')

    if tests:
        _logger.info('Generate tests start')
        test_generator.generate_tests(tests_dir)
        test_generator.generate_answers(cpu_exe_path, tests_dir)
        _logger.info('Generate tests finish')

    if check:
        _logger.info('Checker start')
        checker.check(release_exe, tests_dir)
        _logger.info('Cheker finish')

    if benchmark:
        _logger.info('Benchmark start')
        bench.benchmark(profile_exe, tests_dir)
        bench.benchmark(benchmark_cpu_path, tests_dir, True)
        _logger.info('Benchmark finish')

    _logger.info('All done!')


if __name__ == '__main__':
    main(docopt(__doc__))
