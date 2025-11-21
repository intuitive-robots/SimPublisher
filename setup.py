from setuptools import setup

setup(
    name="simpub",
    version="2.2.0",
    install_requires=[
        "zmq",
        "numpy",
        "colorama",
        "trimesh",
        "Pillow",
        "flask",
        "pyyaml",
    ],
    include_package_data=True,
    packages=["simpub", "simpub.parser", "simpub.sim"],
)
