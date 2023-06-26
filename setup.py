from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=['sawyer_cloth_flatten'],
    package_dir={'': 'src'},
    requires=['actionlib', 'cv_bridge', 'geometry_msgs', 'moveit_commander', 'std_msgs',
              'sensor_msgs', 'tf', 'tf2_geometry_msgs', 'tf2_ros',
              'robotiq_2f_gripper_control', 'intera_interface']
)

setup(**setup_args)