cmake_minimum_required(VERSION 3.5)
project(iris_lama LANGUAGES CXX C)

set(IRIS_LAMA_VERSION 1.3.1)

# Include directories
include(GNUInstallDirs)

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Add the source directory
add_subdirectory(src)

# Install headers
install(DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Export library targets
ament_export_include_directories(include)
ament_export_dependencies(Eigen3)

# Install package.xml
install(FILES package.xml
  DESTINATION share/${PROJECT_NAME})

# Generate and install CMake package configuration files
ament_package()
