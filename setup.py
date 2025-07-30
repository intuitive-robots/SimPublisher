from setuptools import setup

setup(
    name="simpub",
    version="2.1.0",
    install_requires=[
        "zmq",
        "numpy",
        "scipy",
        "colorama",
        "opencv-python",
        "trimesh",
    ],
    include_package_data=True,
    packages=["simpub", "simpub.parser", "simpub.sim"],
)
