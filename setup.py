from setuptools import find_packages
from setuptools import setup
       
setup(
    name='trainer',
    version='0.1',
    #install_requires=find_packages(),
    #packages=find_packages(),
    install_requires = ['opencv-python', 'keras', 'pillow', 'numpy', 'scipy', 'tensorflow>=1.13,<2', 'tensorboard>=1.13,<2'],
    packages=find_packages(),
    include_package_data=True,
    description='My trainer application package.'
)