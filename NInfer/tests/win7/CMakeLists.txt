cmake_minimum_required(VERSION 3.20)
project(ninfer_win7_host_tests LANGUAGES CXX)

enable_testing()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(NINFER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

add_executable(ninfer_win7_cli_options_test
  test_cli_options.cpp
  ${NINFER_ROOT}/apps/cli/options.cpp)
target_include_directories(ninfer_win7_cli_options_test PRIVATE
  ${NINFER_ROOT}
  ${NINFER_ROOT}/include
  ${NINFER_ROOT}/src)
target_compile_definitions(ninfer_win7_cli_options_test PRIVATE
  NINFER_WIN7=1 NINFER_TEXT_ONLY=1)
add_test(NAME ninfer_win7_cli_options COMMAND ninfer_win7_cli_options_test)
