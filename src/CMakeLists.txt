set(lama_SOURCES

    buffer_compressor.cpp
    graph_slam2d.cpp
    hybrid_pf_slam2d.cpp
    image_io.cpp
    kdtree.cpp
    kld_sampling.cpp
    loc2d.cpp
    landmark_pf_slam2d.cpp
    lidar_odometry_2d.cpp
    match_surface_2d.cpp
    pf_slam2d.cpp
    print.cpp
    pose2d.cpp
    pose3d.cpp
    random.cpp
    slam2d.cpp
    thread_pool.cpp
    simple_pgo.cpp

    nlls/gauss_newton.cpp
    nlls/levenberg_marquardt.cpp
    nlls/robust_cost.cpp
    nlls/solver.cpp

    sdm/container.cpp
    sdm/dynamic_distance_map.cpp
    sdm/export.cpp
    sdm/frequency_occupancy_map.cpp
    sdm/map.cpp
    sdm/marching_cubes.cpp
    sdm/probabilistic_occupancy_map.cpp
    sdm/simple_occupancy_map.cpp
    sdm/truncated_signed_distance_map.cpp
)

# Vendor (or third-party) software
add_subdirectory(../vendor/lz4      vendor/lz4)
add_subdirectory(../vendor/minisam  vendor/minisam)
add_subdirectory(../vendor/zstd     vendor/zstd)

# Create iris_lama library
add_library(iris_lama ${lama_SOURCES}
    $<TARGET_OBJECTS:lz4>
    $<TARGET_OBJECTS:minisam>
    $<TARGET_OBJECTS:zstd>
)

# Namespace alias
add_library(iris_lama::iris_lama ALIAS iris_lama)

# Include directories
target_include_directories(iris_lama
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>

    PRIVATE
    ../vendor/lz4
    ../vendor/minisam
    ../vendor/moodycamel
    ../vendor/nanoflann
    ../vendor/stb
    ../vendor/zstd
)

# Compiler options
target_compile_features(iris_lama PUBLIC cxx_std_14)
target_compile_options(iris_lama PRIVATE -Wall -Wextra)

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
    target_compile_options(iris_lama PRIVATE -faligned-new)
endif()

# Link dependencies
target_link_libraries(iris_lama PUBLIC pthread Eigen3::Eigen)

# Install headers
install(DIRECTORY ../include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install the library
install(TARGETS iris_lama
  EXPORT iris_lamaTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Export library targets
install(EXPORT iris_lamaTargets
  FILE iris_lamaTargets.cmake
  NAMESPACE iris_lama::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/iris_lama
)

# Generate package configuration files
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/iris_lamaConfigVersion.cmake
  VERSION ${IRIS_LAMA_VERSION}
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/iris_lamaConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/iris_lamaConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/iris_lama
  PATH_VARS CMAKE_INSTALL_INCLUDEDIR
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/iris_lamaConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/iris_lamaConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/iris_lama
)

