cmake_minimum_required(VERSION 3.15)
project(spline_polyscope_viewer)

include(FetchContent)

set(POLYSCOPE_LOCAL_DIR "" CACHE PATH "Optional local Polyscope source directory")
if(POLYSCOPE_LOCAL_DIR AND EXISTS "${POLYSCOPE_LOCAL_DIR}/CMakeLists.txt")
    add_subdirectory("${POLYSCOPE_LOCAL_DIR}" polyscope)
else()
    FetchContent_Declare(
        polyscope
        GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
        GIT_TAG master
    )
    FetchContent_MakeAvailable(polyscope)
endif()

add_executable(spline_polyscope_viewer spline_polyscope_viewer.cpp)
target_link_libraries(spline_polyscope_viewer PRIVATE polyscope)

if(WIN32)
    target_compile_options(spline_polyscope_viewer PRIVATE -Wa,-mbig-obj)
endif()
