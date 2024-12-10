from setuptools import setup

setup(
    name='simpub',
    version='1.3.1',
    install_requires=["zmq", "pillow", "numpy", "scipy", "colorama", "opencv-python"],
    include_package_data=True,
    packages=['simpub', 'simpub.parser', 'simpub.sim']
)
