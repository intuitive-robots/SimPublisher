from setuptools import setup

setup(
    name="simpub",
    version="2.2.0",
    description="A cross-environment tool to publish objects from simulation for Augmented Reality and Human Robot Interaction.",
    author="Xinkai Jiang",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    url="https://intuitive-robots.github.io/iris-project-page/",
    license="Apache-2.0",
    install_requires=[
        "pyzmq",
        "numpy",
        "colorama",
        "trimesh",
        "Pillow",
        "flask",
        "pyyaml",
    ],
    include_package_data=False,
    packages=["simpub", "simpub.parser", "simpub.sim"],
)
