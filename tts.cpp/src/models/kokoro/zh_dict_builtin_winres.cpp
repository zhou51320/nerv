#include "zh_dict_builtin.h"

#if defined(_WIN32) && defined(TTS_ZH_DICT_BUILTIN)

#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>

namespace kokoro_zh {

namespace {

// 说明：
// - Windows 资源（RCDATA）会被链接进可执行文件的 .rsrc 段中。
// - 这里直接通过 WinAPI 获取资源指针与大小，并以 std::string_view 视图返回。
// - 资源指针在模块生命周期内有效，因此可安全缓存为 static。
static std::string_view load_rcdata_utf8(const wchar_t * name) {
    // 说明：这里显式使用主模块（exe）的资源表。
    // - 本项目中 `tts` 是静态库，资源文件会被编译进最终 exe（例如 tts-cli.exe）。
    // - 因此这里使用 GetModuleHandleW(nullptr) 总能定位到资源。
    HMODULE self = GetModuleHandleW(nullptr);
    if (self == nullptr) {
        return {};
    }

    // 说明：RT_RCDATA 的类型会随 UNICODE 宏变化（A/W），这里显式用 W 版本以匹配 FindResourceW。
    // RCDATA 的预定义数值类型为 10。
    HRSRC res = FindResourceW(self, name, MAKEINTRESOURCEW(10));
    if (res == nullptr) {
        return {};
    }

    DWORD size = SizeofResource(self, res);
    if (size == 0) {
        return {};
    }

    HGLOBAL loaded = LoadResource(self, res);
    if (loaded == nullptr) {
        return {};
    }

    const void * ptr = LockResource(loaded);
    if (ptr == nullptr) {
        return {};
    }

    return std::string_view{static_cast<const char *>(ptr), static_cast<size_t>(size)};
}

} // namespace

std::string_view zh_builtin_pinyin_phrase_utf8() {
    static const std::string_view k = load_rcdata_utf8(L"EVA_TTS_ZH_PHRASE_DICT");
    return k;
}

std::string_view zh_builtin_pinyin_single_utf8() {
    static const std::string_view k = load_rcdata_utf8(L"EVA_TTS_ZH_SINGLE_DICT");
    return k;
}

} // namespace kokoro_zh

#endif // defined(_WIN32) && defined(TTS_ZH_DICT_BUILTIN)
