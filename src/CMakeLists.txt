
set(lama_SOURCES
    # Base
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

    # Non-Linear Least Squares Optimization
    nlls/gauss_newton.cpp
    nlls/levenberg_marquardt.cpp
    nlls/robust_cost.cpp
    nlls/solver.cpp

    # Sparse-Dense Mapping
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

# vendor (or third party) software
add_subdirectory(../vendor/lz4      vendor/lz4)
add_subdirectory(../vendor/minisam  vendor/minisam)
add_subdirectory(../vendor/zstd     vendor/zstd)

add_library(lama_core ${lama_SOURCES}
    $<TARGET_OBJECTS:lz4>
    $<TARGET_OBJECTS:minisam>
    $<TARGET_OBJECTS:zstd>
)

target_include_directories(lama_core
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

target_compile_features(lama_core PUBLIC cxx_std_14)
target_compile_options(lama_core PRIVATE -Wall -Wextra)

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
    target_compile_options(lama_core PRIVATE -faligned-new)
endif()

target_link_libraries(lama_core PUBLIC pthread Eigen3::Eigen)
target_include_directories(lama_core PRIVATE extern/minisam)

# libs and binaries, generate export targets
install(TARGETS lama_core
  EXPORT ${CMAKE_PROJECT_NAME}Targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
