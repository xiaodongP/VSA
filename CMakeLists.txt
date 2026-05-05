cmake_minimum_required(VERSION 3.15)
project(variational)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# libigl - 启用 OpenGL 和 GLFW 模块
set(LIBIGL_OPENGL ON CACHE BOOL "Build target igl::opengl" FORCE)
set(LIBIGL_GLFW ON CACHE BOOL "Build target igl::glfw" FORCE)

# 使用 add_subdirectory 方式添加 libigl
set(LIBIGL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../libigl" CACHE PATH "libigl directory")
add_subdirectory(${LIBIGL_DIR} libigl)

# Add your project files
#set(SRCS main.cpp MeshTransformation.cpp)
#set(HEADERS MeshTransformation.h)
#add_executable(${PROJECT_NAME}_bin ${SRCS} ${HEADERS})

file(GLOB SRCFILES *.cpp)
add_executable(${PROJECT_NAME}_bin ${SRCFILES})

target_link_libraries(${PROJECT_NAME}_bin igl::core igl::opengl igl::glfw)

# Fix MinGW "too many sections" error with template-heavy Eigen code
if(WIN32)
    target_compile_options(${PROJECT_NAME}_bin PRIVATE -Wa,-mbig-obj)
endif()

# Quadric proxy unit test
add_executable(quadric_test tests/test_quadric.cpp quadric_proxy.cpp)
target_include_directories(quadric_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quadric_test igl::core)
if(WIN32)
    target_compile_options(quadric_test PRIVATE -Wa,-mbig-obj)
endif()
