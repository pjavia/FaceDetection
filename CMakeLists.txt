cmake_minimum_required(VERSION 3.10)
project(faceDetection)

set(CMAKE_CXX_STANDARD 17)

set(SOURCE_FILES src/main.cpp header/hog.hpp src/hog.cpp src/patch.cpp header/patch.hpp)

find_package(Boost 1.65.1 COMPONENTS)
find_package(OpenCV 3.3.1 REQUIRED)
# find_package(OpenGL 4.5.0 REQUIRED)
# find_package(GLUT REQUIRED)
# find_package(PkgConfig REQUIRED)
# pkg_search_module(GLFW REQUIRED glfw3)

if(Boost_FOUND AND OpenCV_FOUND) # AND OpenGL_FOUND AND GLUT_FOUND)
    # execute_process(COMMAND glxinfo
    #                COMMAND grep "OpenGL version"
    #                OUTPUT_VARIABLE bar)
    # MESSAGE(STATUS "OpenGL version: " ${bar} " FOUND")
    MESSAGE(STATUS "Boost version: " ${Boost_VERSION} " FOUND")
    MESSAGE(STATUS "OpenCV version: " ${OpenCV_VERSION} " FOUND")
    include_directories(${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS}) # ${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS} ${GLFW_INCLUDE_DIRS})
    link_directories(${Boost_LIBRARY_DIRS})
    add_executable(faceDetection ${SOURCE_FILES})
    target_link_libraries(faceDetection ${Boost_LIBRARIES} ${OpenCV_LIBS} -ljpeg -lboost_system)# ${GLFW_STATIC_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})#${GLFW_LIBRARIES})

elseif(NOT OpenCV_FOUND)
    MESSAGE(STATUS "Error cannot find OpenCV: " ${OpenCV_VERSION})

elseif(NOT Boost_FOUND)
    MESSAGE(STATUS "Error cannot find Boost: " ${Boost_VERSION})

# elseif(NOT OpenGL_FOUND)
#    MESSAGE(STATUS "Error cannot find OpenGL")

# elseif(NOT GLUT_FOUND)
#     MESSAGE(STATUS "Error cannot find GLUT")
endif()


