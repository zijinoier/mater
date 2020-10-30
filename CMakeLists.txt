cmake_minimum_required(VERSION 3.5)

project(mater)

find_package(Boost 1.58 REQUIRED COMPONENTS program_options)
find_package(PkgConfig)
#pkg_check_modules(YamlCpp yaml-cpp)
set(YamlCpp_DIR "/usr/lib/x86_64-linux-gnu/cmake/yaml-cpp/")
find_package(YamlCpp)

# Enable C++14 and warnings
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wextra")
add_compile_options(-fPIC)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -fpic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -fpic")
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

# Creates compile database used by clang-tidy.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(
  include
)

# pybind
set(PROJ_NAME "train_env")
find_package(pybind11 REQUIRED)

pybind11_add_module(${PROJ_NAME} src/train_env.cpp)

target_link_libraries(${PROJ_NAME} PRIVATE pybind11::module  yaml-cpp)
# set(Python_ADDITIONAL_VERSIONS 3.7)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -fpic")
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

install(TARGETS ${PROJ_NAME} RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/bin
							 LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/src/
							 ARCHIVE DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib/static
               )
               
# Examples

# ## a_star
# add_executable(a_star
#   example/a_star.cpp
# )
# target_link_libraries(a_star
#   ${Boost_LIBRARIES}
# )

# ## joint_a_star
# add_executable(jointstate_astar
#   example/jointstate_astar.cpp
# )
# target_link_libraries(jointstate_astar
#   ${Boost_LIBRARIES}
#   yaml-cpp
# )

# ## cbs
# add_executable(cbs
#   example/cbs.cpp
# )
# target_link_libraries(cbs
#   ${Boost_LIBRARIES}
#   yaml-cpp
# )


