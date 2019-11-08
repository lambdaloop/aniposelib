import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="calligator",
    version="0.3.1",
    author="Pierre Karashchuk",
    author_email="krchtchk@gmail.com",
    description="An easy-to-use library for calibrating cameras in python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lambdaloop/calligator",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Image Recognition"
    ],
    install_requires=[
        'opencv-python~=3.4',
        'opencv-contrib-python~=3.4',
        'numba',
        'numpy', 'scipy', 'toml'
    ],
    extras_require={
        'full':  ["checkerboard"]
    }
)
