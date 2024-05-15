from setuptools import setup, find_packages

setup(
    name='simpub',
    version='0.1',
    install_requires=["zmq", "trimesh", "dm_control", "pillow", "numpy"],
    include_package_data=True,
    packages = ['simpub', 'simpub.connection', 'simpub.model_loader']
)
