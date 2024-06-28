from setuptools import setup, find_packages

setup(
    name="miniflow",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "matplotlib",
        # 其他依赖
    ],
    description="A mini deep learning framework",
    long_description="A detailed description of your project",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/miniflow",
)