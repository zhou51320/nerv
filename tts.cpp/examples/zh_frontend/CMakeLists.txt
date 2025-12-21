set(TARGET tts-zh-frontend)

add_executable(${TARGET}
    zh_frontend_cli.cpp
)

target_link_libraries(${TARGET} PRIVATE ggml tts)

