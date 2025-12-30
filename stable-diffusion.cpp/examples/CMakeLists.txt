include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(cli)
if (SD_BUILD_SERVER)
    add_subdirectory(server)
endif()
