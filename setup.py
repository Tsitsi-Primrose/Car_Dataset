from setuptools import find_packages
from setuptools import setup

#REQUIRED_PACKAGES = ['some_PyPI_package>=1.0',
#                    'pillow', 
 #                   'numpy', 
#                    'scipy'
 #                   'tensorflow>=1.13,<2',
#                    'tensorboard>=1.13,<2',
#                    'cv2']

setup(
    name='trainer',
    version='0.1',
    #install_requires=find_packages(),
    #packages=find_packages(),
    install_requires = ['opencv-python', 'keras', 'pillow', 'numpy', 'scipy', 'tensorflow>=1.13,<2', 'tensorboard>=1.13,<2', 'cv2', 'some_PyPI_package>=1.0'],
    packages=find_packages(),
    include_package_data=True,
    description='My trainer application package.'
)