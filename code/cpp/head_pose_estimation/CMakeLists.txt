cmake_minimum_required(VERSION 2.8.3)
project(head_pose_estimation)

## System dependencies are found with CMake's conventions

# OpenCV for image processing:
find_package(OpenCV REQUIRED)

#Tensorflow Trained files
add_definitions(-DTF_TRAINED_FILE="${PROJECT_SOURCE_DIR}/model/hopenet_alpha1_onnx.pb")

## Executables using the pylon API must be linked using the -Wl,-E compiler option
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-E")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# types folder shall be one level up
set(LCM_WORKSPACE_DIR ${PROJECT_SOURCE_DIR}/..)

# search path for LCM header files
set(LCM_IncludeSearchPaths
        /usr/include/
        /usr/local/include/
        /opt/local/include
        )

# search path for LCM static/dynamic libraries
set(LCM_LibrarySearchPaths
        /usr/lib/
        /usr/local/lib/
        /opt/local/lib/
        )

find_path(LCM_INCLUDE_DIR
        NAMES lcm/lcm.h
        HINTS ${LCM_IncludeSearchPaths}
        )

FIND_LIBRARY(LCM_LIBS
        NAMES lcm
        HINTS ${LCM_LibrarySearchPaths}
        PATH_SUFFIXES lib
        )
message(STATUS ${LCM_LIBS})

#Set common include 
set(COMMON_DIR $ENV{DSM_WORKSPACE}/src/common)
## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(${LCM_INCLUDE_DIR} include LINK_PUBLIC common/include)

###########
## Build ##
###########

## create headers files from the .lcm data type files
## path to all data types for the messages to publish/subscribe to
set(DATA_TYPE_DIR ${LCM_WORKSPACE_DIR}/types)
set(IMAGE_DATA_TYPE_FILES ${LCM_WORKSPACE_DIR}/types/image_t.lcm)
set(DATA_TYPE_DESTINATION ${PROJECT_SOURCE_DIR}/include)

## this will execute before the MakeFile is created
if (EXISTS ${DATA_TYPE_DIR})
    execute_process(
            COMMAND lcm-gen -x ${IMAGE_DATA_TYPE_FILES} --cpp-hpath ${DATA_TYPE_DESTINATION}
    )
    message(STATUS "${Green}Created LCM data type header files for ${PROJECT_NAME}.${ColourReset}")
endif ()

add_executable(test_head_pose_estimation_video
        test/test_head_pose_estimation_video.cc
        src/head_pose_estimation_api.cc)
target_link_libraries(test_head_pose_estimation_video dsm_common_static
        ${LCM_LIBS})

# recursively, on all folders, find all the .h and cpp files
# Because of speed issues, some of these files are actually C implementations,
# and not C++
file(
        GLOB_RECURSE
        ${PROJECT_NAME}_src
        ${PROJECT_SOURCE_DIR}/*.cpp
)
file(
        GLOB_RECURSE
        ${PROJECT_NAME}_include
        ${PROJECT_SOURCE_DIR}/include/*.h
        ${PROJECT_SOURCE_DIR}/include/*.hpp
)
add_library(
        dsm_${PROJECT_NAME}
        SHARED
        ${${PROJECT_NAME}_src}
        ${${PROJECT_NAME}_include}
)
target_link_libraries(dsm_${PROJECT_NAME} dsm_common ${OpenCV_LIBS})
target_include_directories(dsm_${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
add_library(
        dsm_${PROJECT_NAME}_static
        STATIC
        ${${PROJECT_NAME}_src}
        ${${PROJECT_NAME}_include}
)
target_link_libraries(dsm_${PROJECT_NAME}_static dsm_common_static ${OpenCV_LIBS})
target_include_directories(dsm_${PROJECT_NAME}_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(
        dsm_${PROJECT_NAME}_static
        PROPERTIES
        OUTPUT_NAME
        dsm_${PROJECT_NAME}
)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
#install(TARGETS gesture_namespace_lib v_recognize_gesture_node v_gesture_timed_service_caller
#  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#)

# Mark cpp header files for installation
#install(DIRECTORY include/${PROJECT_NAME}/
#  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#  FILES_MATCHING PATTERN "*.h"
#  PATTERN ".svn" EXCLUDE
#)

## Mark other files for installation (e.g. launch and bag files, etc.)
#install(FILES
#  launch/visual.launch
#  launch/recognize_gesture_params.yaml
#  # myfile2
#  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
#)

## create symbolic link to NiTe folder to obtain the training data
#set(LINK_NAME $ENV{HOME}/.ros/NiTE2)
#set(NITE_DIR ${PROJECT_SOURCE_DIR}/../NiTE-Linux-x64-2.2/Samples/Bin/NiTE2/)
#install(CODE "
#    if(EXISTS ${LINK_NAME})
#      message(\"Link ${LINK_NAME} to ${NITE_DIR} already exists. Re-linking ...\")
#      execute_process(COMMAND rm ${LINK_NAME})
#      execute_process(COMMAND ECHO Test = ${LINK_NAME})
#    endif()
#    execute_process(COMMAND ln -s ${NITE_DIR} ${LINK_NAME})
#    message(\"Created link ${LINK_NAME} to ${NITE_DIR}.\")
#  "
#)

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_gesture_tracker.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
