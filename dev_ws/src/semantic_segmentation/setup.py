from setuptools import setup, find_packages
import os
from glob import glob


package_name = 'semantic_segmentation'
models = 'semantic_segmentation/models'


setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, models],

     # Files we want to install, specifically launch files

    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + "/pretrained", ['semantic_segmentation/pretrained/hardnet70_cityscapes_model.pkl']),
        ('share/' + package_name + "/pretrained", ['semantic_segmentation/PerspectiveTransform.npz'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jdgalviss',
    maintainer_email='jdgalviss@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    scripts=["semantic_segmentation/fchardnet.py", 
                "semantic_segmentation/utils.py"],
    entry_points={
        'console_scripts': [
            'semantic_segmentation = semantic_segmentation.semantic_segmentation:main'
        ],
    },
)
