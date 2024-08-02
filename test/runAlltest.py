import unittest
import os
import sys
import importlib.util

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_test_file(file_path):
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    suite = unittest.TestSuite()
    loader = unittest.defaultTestLoader
    suite.addTests(loader.loadTestsFromModule(module))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result

if __name__ == '__main__':
    print(f"Current working directory: {os.getcwd()}")
    test_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Test directory: {test_dir}")

    total_tests = 0
    total_successes = 0
    total_failures = 0
    total_errors = 0
    all_failures = []
    all_errors = []

    # CNN Test
    cnn_test_files = [
        os.path.join(test_dir, 'CNN Test', 'conv_test.py'),
        os.path.join(test_dir, 'CNN Test', 'miniflow_cnn_minst_test.py')
    ]

    # Dense Layer Test
    dense_test_files = [
        os.path.join(test_dir, 'Dense Layer Test', 'eminst_test.py'),
        os.path.join(test_dir, 'Dense Layer Test', 'minst_test.py')
    ]

    # GPU test
    gpu_test_file = os.path.join(test_dir, 'GPU test.py')

    all_test_files = cnn_test_files + dense_test_files + [gpu_test_file]

    for file_path in all_test_files:
        if os.path.exists(file_path):
            print(f"\nRunning tests from file: {file_path}")
            result = run_test_file(file_path)

            total_tests += result.testsRun
            total_successes += result.testsRun - len(result.failures) - len(result.errors)
            total_failures += len(result.failures)
            total_errors += len(result.errors)

            all_failures.extend([(file_path, test, traceback) for test, traceback in result.failures])
            all_errors.extend([(file_path, test, traceback) for test, traceback in result.errors])
        else:
            print(f"Warning: Test file not found: {file_path}")

    print("\n==== Test Result Summary ====")
    print(f"Ran {total_tests} tests")
    print(f"Successes: {total_successes}")
    print(f"Failures: {total_failures}")
    print(f"Errors: {total_errors}")

    if all_failures:
        print("\n==== Failures ====")
        for file, test, traceback in all_failures:
            print(f"\nFile: {file}")
            print(f"Test: {test}")
            print(traceback)

    if all_errors:
        print("\n==== Errors ====")
        for file, test, traceback in all_errors:
            print(f"\nFile: {file}")
            print(f"Test: {test}")
            print(traceback)

    sys.exit(total_failures + total_errors)