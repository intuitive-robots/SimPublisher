from setuptools import setup

setup(
    name='simpub',
    version='1.2.0',
    install_requires=["zmq", "pillow", "numpy", "scipy", "colorama"],
    include_package_data=True,
    packages=['simpub', 'simpub.parser', 'simpub.sim']
)
