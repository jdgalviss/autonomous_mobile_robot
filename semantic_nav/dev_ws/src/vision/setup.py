from setuptools import setup

package_name = 'vision'
models = 'vision/models'
utils = 'vision/utils'


setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, models, utils],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + "/pretrained", ['vision/pretrained/hardnet70_cityscapes_model.pkl']),
        ('share/' + package_name + "/pretrained", ['vision/pretrained/yolov5s.pt']),
        ('share/' + package_name + "/pretrained", ['vision/PerspectiveTransform.npz']),
        ('share/' + package_name + "/pretrained", ['vision/PerspectiveTransformSim.npz']),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jdgalvis',
    maintainer_email='juan.galvis@tii.ae',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    scripts=["vision/fchardnet_segmentation.py", 
            "vision/pspnet_segmentation.py", 
            "vision/object_detection.py", 
            "vision/costmap.py", 
            "vision/dl_perception.py", 
            "vision/planner.py", 
            "vision/navigation.py", 
            "vision/helpers.py"],
    entry_points={
        'console_scripts': [
            'vision_node = vision.vision_node:main'
        ],
    },
)
