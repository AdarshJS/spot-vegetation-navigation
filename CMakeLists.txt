cmake_minimum_required(VERSION 3.5.1)
project(spot-vegetation-navigation)

find_package(catkin REQUIRED)

catkin_package(
#  INCLUDE_DIRS include
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(
# include
# ${catkin_INCLUDE_DIRS}
)


catkin_python_setup()


#############
## Install ##
#############

  catkin_install_python(PROGRAMS
    scripts/dwa_costmap_v3.py
    siamese_network/inference_v2.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

  install(DIRECTORY launch scripts siamese_network/pyimageseach
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

# ## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   geometry_msgs
#   sensor_msgs
#   std_msgs
# )