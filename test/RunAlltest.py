import os
import sys
import subprocess


def run_python_file(file_path):
    print(f"Running {file_path}...")
    result = subprocess.run([sys.executable, file_path], check=False)
    print(f"Finished running {file_path}")
    return result.returncode == 0


def run_all_tests():
    # 获取项目根目录的路径
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 切换到项目根目录
    os.chdir(root_dir)

    test_files = [
        os.path.join("test", "CNN Test", "miniflow_cnn_minst_test.py"),
        os.path.join("test", "Dense Layer Test", "eminst_test.py"),
        os.path.join("test", "Dense Layer Test", "minst_test.py")
    ]

    all_passed = True
    for test_file in test_files:
        if not run_python_file(test_file):
            all_passed = False

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    print("All tests passed!" if success else "Some tests failed.")
    sys.exit(0 if success else 1)