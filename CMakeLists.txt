cmake_minimum_required(VERSION 3.15)
project(variational)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(FetchContent)

# libigl - 启用 OpenGL 和 GLFW 模块
set(LIBIGL_OPENGL ON CACHE BOOL "Build target igl::opengl" FORCE)
set(LIBIGL_GLFW ON CACHE BOOL "Build target igl::glfw" FORCE)
set(LIBIGL_IMGUI ON CACHE BOOL "Build target igl::imgui" FORCE)

# 使用 add_subdirectory 方式添加 libigl
set(LIBIGL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../libigl" CACHE PATH "libigl directory")
add_subdirectory(${LIBIGL_DIR} libigl)

# Header-only CGAL CDT backend for reconstruction.
# CGAL 6.x is header-only for this use case, but it needs Boost headers.
set(BOOST_HEADERS_LOCAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/boost-1.84.0")
set(CGAL_HEADERS_LOCAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/CGAL-6.0.1")
if(EXISTS "${BOOST_HEADERS_LOCAL_DIR}/libs/config/include/boost/version.hpp")
    set(boost_headers_SOURCE_DIR "${BOOST_HEADERS_LOCAL_DIR}")
else()
    FetchContent_Declare(
        boost_headers
        URL https://github.com/boostorg/boost/releases/download/boost-1.84.0/boost-1.84.0.tar.xz
        TLS_VERIFY FALSE
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_GetProperties(boost_headers)
    if(NOT boost_headers_POPULATED)
        FetchContent_Populate(boost_headers)
    endif()
endif()

if(EXISTS "${CGAL_HEADERS_LOCAL_DIR}/include/CGAL/Constrained_Delaunay_triangulation_2.h")
    set(cgal_headers_SOURCE_DIR "${CGAL_HEADERS_LOCAL_DIR}")
else()
    FetchContent_Declare(
        cgal_headers
        URL https://github.com/CGAL/cgal/releases/download/v6.0.1/CGAL-6.0.1-library.tar.xz
        TLS_VERIFY FALSE
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_GetProperties(cgal_headers)
    if(NOT cgal_headers_POPULATED)
        FetchContent_Populate(cgal_headers)
    endif()
endif()

add_library(vsa_cgal_headers INTERFACE)
file(GLOB BOOST_MODULE_INCLUDE_DIRS
    "${boost_headers_SOURCE_DIR}/libs/*/include"
    "${boost_headers_SOURCE_DIR}/libs/*/*/include")
target_include_directories(vsa_cgal_headers INTERFACE
    ${BOOST_MODULE_INCLUDE_DIRS}
    ${cgal_headers_SOURCE_DIR}/include)
target_compile_definitions(vsa_cgal_headers INTERFACE
    VSA_USE_CGAL_CDT
    CGAL_HEADER_ONLY
    CGAL_NO_GMP
    CGAL_DISABLE_GMP
    CGAL_USE_BOOST_MP
    CGAL_CORE_USE_BOOST_BACKEND)

# Add your project files
#set(SRCS main.cpp MeshTransformation.cpp)
#set(HEADERS MeshTransformation.h)
#add_executable(${PROJECT_NAME}_bin ${SRCS} ${HEADERS})

file(GLOB SRCFILES *.cpp)
add_executable(${PROJECT_NAME}_bin ${SRCFILES})

target_link_libraries(${PROJECT_NAME}_bin igl::core igl::opengl igl::glfw igl::imgui vsa_cgal_headers)

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

# Boundary smoothing synthetic test
add_executable(boundary_smoothing_test
    tests/test_boundary_smoothing.cpp
    boundary_smoothing.cpp
    feature_barrier.cpp
    proxies.cpp
    distance.cpp
    quadric_proxy.cpp)
target_include_directories(boundary_smoothing_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(boundary_smoothing_test igl::core igl::opengl igl::glfw)
if(WIN32)
    target_compile_options(boundary_smoothing_test PRIVATE -Wa,-mbig-obj)
endif()

# B-spline basis/curve/surface unit test
add_executable(bspline_test tests/test_bspline.cpp bspline.cpp)
target_include_directories(bspline_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(bspline_test igl::core)
if(WIN32)
    target_compile_options(bspline_test PRIVATE -Wa,-mbig-obj)
endif()

# B-spline curve fitting synthetic test
add_executable(bspline_fit_test tests/test_bspline_fit.cpp bspline.cpp)
target_include_directories(bspline_fit_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(bspline_fit_test igl::core)
if(WIN32)
    target_compile_options(bspline_fit_test PRIVATE -Wa,-mbig-obj)
endif()

# Region boundary loop extraction synthetic test
add_executable(region_boundary_test
    tests/test_region_boundary.cpp
    region_boundary.cpp)
target_include_directories(region_boundary_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(region_boundary_test igl::core)
if(WIN32)
    target_compile_options(region_boundary_test PRIVATE -Wa,-mbig-obj)
endif()

# Trimmed-region paper input adaptation and boundary segmentation test
add_executable(trimmed_region_input_test
    tests/test_trimmed_region_input.cpp
    trimmed_region_input.cpp)
target_include_directories(trimmed_region_input_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(trimmed_region_input_test igl::core)
if(WIN32)
    target_compile_options(trimmed_region_input_test PRIVATE -Wa,-mbig-obj)
endif()

# Trimmed-surface automatic labeling synthetic test
add_executable(trimmed_labeling_test
    tests/test_trimmed_labeling.cpp
    trimmed_labeling.cpp
    trimmed_region_input.cpp)
target_include_directories(trimmed_labeling_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(trimmed_labeling_test igl::core)
if(WIN32)
    target_compile_options(trimmed_labeling_test PRIVATE -Wa,-mbig-obj)
endif()

# Bezier guiding frame synthetic test
add_executable(bezier_guiding_frame_test
    tests/test_bezier_guiding_frame.cpp
    bezier_guiding_frame.cpp
    trimmed_labeling.cpp
    trimmed_region_input.cpp)
target_include_directories(bezier_guiding_frame_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(bezier_guiding_frame_test igl::core)
if(WIN32)
    target_compile_options(bezier_guiding_frame_test PRIVATE -Wa,-mbig-obj)
endif()

# Constrained ARAP MVP parameterization synthetic test
add_executable(constrained_arap_parameterization_test
    tests/test_constrained_arap_parameterization.cpp
    constrained_arap_parameterization.cpp
    bezier_guiding_frame.cpp
    trimmed_labeling.cpp
    trimmed_region_input.cpp)
target_include_directories(constrained_arap_parameterization_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(constrained_arap_parameterization_test igl::core)
if(WIN32)
    target_compile_options(constrained_arap_parameterization_test PRIVATE -Wa,-mbig-obj)
endif()

# Rotation-angle initialization and KKT stitching synthetic test
add_executable(rotation_angle_parameterization_test
    tests/test_rotation_angle_parameterization.cpp
    rotation_angle_parameterization.cpp
    constrained_arap_parameterization.cpp
    bezier_guiding_frame.cpp
    trimmed_labeling.cpp
    trimmed_region_input.cpp)
target_include_directories(rotation_angle_parameterization_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(rotation_angle_parameterization_test igl::core)
if(WIN32)
    target_compile_options(rotation_angle_parameterization_test PRIVATE -Wa,-mbig-obj)
endif()

# Rectangular parameter-domain extension synthetic test
add_executable(rectangular_domain_extension_test
    tests/test_rectangular_domain_extension.cpp
    rectangular_domain_extension.cpp
    rotation_angle_parameterization.cpp
    constrained_arap_parameterization.cpp
    bezier_guiding_frame.cpp
    trimmed_labeling.cpp
    trimmed_region_input.cpp)
target_include_directories(rectangular_domain_extension_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(rectangular_domain_extension_test igl::core vsa_cgal_headers)
if(WIN32)
    target_compile_options(rectangular_domain_extension_test PRIVATE -Wa,-mbig-obj)
endif()

# Polyharmonic 3D rectangular extension synthetic test
add_executable(polyharmonic_3d_extension_test
    tests/test_polyharmonic_3d_extension.cpp
    polyharmonic_3d_extension.cpp
    rectangular_domain_extension.cpp
    rotation_angle_parameterization.cpp
    constrained_arap_parameterization.cpp
    bezier_guiding_frame.cpp
    trimmed_labeling.cpp
    trimmed_region_input.cpp)
target_include_directories(polyharmonic_3d_extension_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(polyharmonic_3d_extension_test igl::core vsa_cgal_headers)
if(WIN32)
    target_compile_options(polyharmonic_3d_extension_test PRIVATE -Wa,-mbig-obj)
endif()

# Tensor-product cubic B-spline surface fitter synthetic test
add_executable(tensor_product_bspline_fitter_test
    tests/test_tensor_product_bspline_fitter.cpp
    tensor_product_bspline_fitter.cpp
    bspline.cpp)
target_include_directories(tensor_product_bspline_fitter_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tensor_product_bspline_fitter_test igl::core)
if(WIN32)
    target_compile_options(tensor_product_bspline_fitter_test PRIVATE -Wa,-mbig-obj)
endif()

# Integrated single-region trimmed B-spline fitting pipeline test
add_executable(trimmed_bspline_pipeline_test
    tests/test_trimmed_bspline_pipeline.cpp
    trimmed_bspline_pipeline.cpp
    reusable_trimmed_bspline_surface.cpp
    trimmed_bspline_surface.cpp
    vsa_cgal_cdt.cpp
    tensor_product_bspline_fitter.cpp
    trimmed_mesh_validation.cpp
    polyharmonic_3d_extension.cpp
    rectangular_domain_extension.cpp
    rotation_angle_parameterization.cpp
    constrained_arap_parameterization.cpp
    bezier_guiding_frame.cpp
    trimmed_labeling.cpp
    trimmed_region_input.cpp
    quad_like_boundary.cpp
    region_boundary.cpp
    bspline.cpp)
target_include_directories(trimmed_bspline_pipeline_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(trimmed_bspline_pipeline_test igl::core vsa_cgal_headers)
if(WIN32)
    target_compile_options(trimmed_bspline_pipeline_test PRIVATE -Wa,-mbig-obj)
endif()

add_executable(trimmed_pipeline_snapshot_regression
    tests/test_trimmed_pipeline_snapshot_regression.cpp
    trimmed_bspline_pipeline.cpp
    reusable_trimmed_bspline_surface.cpp
    trimmed_bspline_surface.cpp
    vsa_cgal_cdt.cpp
    tensor_product_bspline_fitter.cpp
    trimmed_mesh_validation.cpp
    polyharmonic_3d_extension.cpp
    rectangular_domain_extension.cpp
    rotation_angle_parameterization.cpp
    constrained_arap_parameterization.cpp
    bezier_guiding_frame.cpp
    trimmed_labeling.cpp
    trimmed_region_input.cpp
    quad_like_boundary.cpp
    region_boundary.cpp
    bspline.cpp)
target_include_directories(trimmed_pipeline_snapshot_regression PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(trimmed_pipeline_snapshot_regression igl::core vsa_cgal_headers)
if(WIN32)
    target_compile_options(trimmed_pipeline_snapshot_regression PRIVATE -Wa,-mbig-obj)
endif()

# Quad-like boundary corner and macro-side synthetic test
add_executable(quad_like_boundary_test
    tests/test_quad_like_boundary.cpp
    quad_like_boundary.cpp
    region_boundary.cpp)
target_include_directories(quad_like_boundary_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quad_like_boundary_test igl::core)
if(WIN32)
    target_compile_options(quad_like_boundary_test PRIVATE -Wa,-mbig-obj)
endif()

# Initial bicubic B-spline surface from four boundary curves test
add_executable(initial_bspline_surface_test
    tests/test_initial_bspline_surface.cpp
    initial_bspline_surface.cpp
    bspline.cpp
    quad_like_boundary.cpp
    region_boundary.cpp)
target_include_directories(initial_bspline_surface_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(initial_bspline_surface_test igl::core)
if(WIN32)
    target_compile_options(initial_bspline_surface_test PRIVATE -Wa,-mbig-obj)
endif()

# Region square harmonic/Tutte parameterization test
add_executable(region_square_parameterization_test
    tests/test_region_square_parameterization.cpp
    region_square_parameterization.cpp
    quad_like_boundary.cpp
    region_boundary.cpp)
target_include_directories(region_square_parameterization_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(region_square_parameterization_test igl::core)
if(WIN32)
    target_compile_options(region_square_parameterization_test PRIVATE -Wa,-mbig-obj)
endif()

# Fixed-boundary B-spline surface interior control fit test
add_executable(fit_bspline_surface_interior_test
    tests/test_fit_bspline_surface_interior.cpp
    fit_bspline_surface_interior.cpp
    initial_bspline_surface.cpp
    bspline.cpp
    quad_like_boundary.cpp
    region_boundary.cpp
    region_square_parameterization.cpp)
target_include_directories(fit_bspline_surface_interior_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fit_bspline_surface_interior_test igl::core)
if(WIN32)
    target_compile_options(fit_bspline_surface_interior_test PRIVATE -Wa,-mbig-obj)
endif()

# Trimmed B-spline surface sampling test
add_executable(trimmed_bspline_surface_test
    tests/test_trimmed_bspline_surface.cpp
    trimmed_bspline_surface.cpp
    vsa_cgal_cdt.cpp
    bspline.cpp)
target_include_directories(trimmed_bspline_surface_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(trimmed_bspline_surface_test igl::core vsa_cgal_headers)
if(WIN32)
    target_compile_options(trimmed_bspline_surface_test PRIVATE -Wa,-mbig-obj)
endif()

# Two adjacent patches sharing one B-spline boundary test
add_executable(shared_spline_boundary_test
    tests/test_shared_spline_boundary.cpp
    shared_spline_boundary.cpp
    bspline.cpp)
target_include_directories(shared_spline_boundary_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(shared_spline_boundary_test igl::core)
if(WIN32)
    target_compile_options(shared_spline_boundary_test PRIVATE -Wa,-mbig-obj)
endif()

# Global fixed-boundary B-spline control point optimization test
add_executable(global_bspline_optimization_test
    tests/test_global_bspline_optimization.cpp
    global_bspline_optimization.cpp
    shared_spline_boundary.cpp
    fit_bspline_surface_interior.cpp
    initial_bspline_surface.cpp
    region_square_parameterization.cpp
    quad_like_boundary.cpp
    region_boundary.cpp
    bspline.cpp)
target_include_directories(global_bspline_optimization_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(global_bspline_optimization_test igl::core)
if(WIN32)
    target_compile_options(global_bspline_optimization_test PRIVATE -Wa,-mbig-obj)
endif()
