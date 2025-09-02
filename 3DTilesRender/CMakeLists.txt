cmake_minimum_required(VERSION 3.22)

project(
    ModelRenderScene
    VERSION 0.1.0
    LANGUAGES CXX C
)
OPTION(BUILD_AS_PYTHON_MODULE "Build the code as library for python module." OFF)
IF(BUILD_AS_PYTHON_MODULE)
    set(PYBIND11_PYTHON_VERSION 3)
    ADD_DEFINITIONS(-DPYTHON_MODULE)
ENDIF(BUILD_AS_PYTHON_MODULE)

# 设置你想要使用的Python版本
set(Python_FIND_VERSION 3)
set(Python_FIND_STRATEGY VERSION)
set(PYTHON_EXECUTABLE "/home/ubuntu/miniconda3/envs/gigapose/bin/python3.9") 
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMakeModules;${CMAKE_MODULE_PATH}")

find_package(OpenSceneGraph REQUIRED COMPONENTS osg osgDB osgGA osgUtil osgText osgViewer) 
find_package(OpenCV REQUIRED)
find_package(OsgEarth REQUIRED)
find_package(CesiumNative REQUIRED)
find_package(EGL REQUIRED)
IF(BUILD_AS_PYTHON_MODULE)
    find_package(pybind11 REQUIRED)
ENDIF(BUILD_AS_PYTHON_MODULE)
message("CESIUM_NATIVE_FOUND= ${CESIUM_NATIVE_FOUND}")
# find_package(osgEarth REQUIRED)

IF(BUILD_AS_PYTHON_MODULE)
    pybind11_add_module(${PROJECT_NAME} pywrap.cpp ModelRenderScene.cpp)
    target_include_directories(${PROJECT_NAME} PRIVATE ${PYBIND11_INCLUDE_DIRS} ${OSGEARTH_INCLUDE_DIRS} ${OSG_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
ELSE()
    add_executable(${PROJECT_NAME} main.cpp EarthWalkManipulator.cpp ModelRenderScene.cpp)
    target_include_directories(${PROJECT_NAME} PRIVATE ${OSGEARTH_INCLUDE_DIRS} ${OSG_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
ENDIF(BUILD_AS_PYTHON_MODULE)

message("OSG Library: ${OPENSCENEGRAPH_LIBRARIES}")
message("EGL Library:" ${EGL_LIBRARY})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OSGEARTHCESIUM_LIBRARY} ${OSGEARTH_LIBRARY} ${OPENSCENEGRAPH_LIBRARIES} ${OpenCV_LIBS} ${EGL_LIBRARY})

install(TARGETS ${PROJECT_NAME} 
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        COMPONENT library)
