target_sources(tts PRIVATE
        loader.cpp
        model.cpp
        model.h
        multilingual.cpp
        multilingual.h
        phonemizer.cpp
        phonemizer.h
        ja_frontend.cpp
        ja_frontend.h
        zh_frontend.cpp
        zh_frontend.h
        zh_dict_builtin.h
        zh_pinyin_data.cpp
        zh_pinyin_data.h
)

# 内置中文词典（默认开启）：把 dict/pinyin_phrase.txt / dict/pinyin.txt 编译进二进制，便于移植。
option(TTS_KOKORO_ZH_DICT_BUILTIN "TTS.cpp: embed Kokoro zh pinyin dictionaries into binary" ON)
if (TTS_KOKORO_ZH_DICT_BUILTIN)
    set(_TTS_ZH_PHRASE_DICT "${CMAKE_SOURCE_DIR}/dict/pinyin_phrase.txt")
    set(_TTS_ZH_SINGLE_DICT "${CMAKE_SOURCE_DIR}/dict/pinyin.txt")
    set(_TTS_ZH_DICT_BUILTIN_CPP "${CMAKE_CURRENT_BINARY_DIR}/zh_dict_builtin.cpp")

    # 说明：将词典嵌入源码（生成一个 .cpp）后再编译进目标，运行时即可脱离外部词典文件。
    # 这里选择在 configure 阶段直接生成文件，以兼容部分生成器对 “不存在的 generated source” 的检查。
    execute_process(
        COMMAND "${CMAKE_COMMAND}"
            "-DINPUT_PHRASE=${_TTS_ZH_PHRASE_DICT}"
            "-DINPUT_SINGLE=${_TTS_ZH_SINGLE_DICT}"
            "-DOUTPUT_CPP=${_TTS_ZH_DICT_BUILTIN_CPP}"
            "-P" "${CMAKE_SOURCE_DIR}/cmake/embed_zh_dict.cmake"
        RESULT_VARIABLE _TTS_ZH_DICT_GEN_RESULT
    )
    if (NOT _TTS_ZH_DICT_GEN_RESULT EQUAL 0)
        message(FATAL_ERROR "failed to generate builtin zh dict source: ${_TTS_ZH_DICT_BUILTIN_CPP}")
    endif()
    unset(_TTS_ZH_DICT_GEN_RESULT)

    target_sources(tts PRIVATE "${_TTS_ZH_DICT_BUILTIN_CPP}")
    target_compile_definitions(tts PRIVATE TTS_ZH_DICT_BUILTIN=1)

    unset(_TTS_ZH_PHRASE_DICT)
    unset(_TTS_ZH_SINGLE_DICT)
    unset(_TTS_ZH_DICT_BUILTIN_CPP)
endif()
