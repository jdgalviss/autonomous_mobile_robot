from setuptools import setup

package_name = 'vision'
models = 'vision/models'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, models],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + "/pretrained", ['vision/pretrained/hardnet70_cityscapes_model.pkl']),
        ('share/' + package_name + "/pretrained", ['vision/PerspectiveTransform.npz'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jdgalvis',
    maintainer_email='juan.galvis@tii.ae',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    scripts=["vision/segmentation.py", 
                "vision/utils.py"],
    entry_points={
        'console_scripts': [
            'vision_node = vision.vision_node:main'
        ],
    },
)
