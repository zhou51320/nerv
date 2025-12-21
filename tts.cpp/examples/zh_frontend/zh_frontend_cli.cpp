#include <cstdio>
#include <string>
#include <vector>

#include "common.h"
#include "../../src/models/kokoro/multilingual.h"
#include "../../src/models/kokoro/zh_frontend.h"

#ifdef _WIN32
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <shellapi.h>
#endif

// 说明：
// - 这是一个“仅用于调试中文前端”的小工具，帮助快速确认：
//   1) 中文词典是否生效（多音字/短语是否命中）；
//   2) 数字/标点归一化后的输出 token 串是否符合预期。
// - 输出的是 Kokoro 使用的“单字符 token 串”（不是可直接阅读的拼音），但可以用来对比：
//   例如 “行为” 的 “为” 若正确应输出 tone=2，否则多半会变成 tone=4。

static void set_console_utf8() {
#ifdef _WIN32
    // 说明：仅影响控制台输入/输出的代码页（不保证影响 argv 的编码转换），但可以避免中文输出乱码。
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
}

#ifdef _WIN32
static std::string wide_to_utf8(const wchar_t * w) {
    if (w == nullptr || w[0] == L'\0') {
        return {};
    }
    const int needed = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    if (needed <= 1) {
        return {};
    }
    std::string out;
    out.resize(static_cast<size_t>(needed - 1));
    WideCharToMultiByte(CP_UTF8, 0, w, -1, out.data(), needed, nullptr, nullptr);
    return out;
}
#endif

static void print_usage(const char * argv0) {
    std::fprintf(stderr,
                 "Usage:\n"
                 "  %s --text \"<中文/混合文本>\" [--zh-dict-dir <dir|:builtin|->] [--multilingual]\n"
                 "\n"
                 "Examples:\n"
                 "  %s --text \"行为\" --zh-dict-dir dict\n"
                 "  %s --text \"温度23.5°C\" --zh-dict-dir :builtin --multilingual\n",
                 argv0,
                 argv0,
                 argv0);
}

static int main_utf8(int argc, const char ** argv) {
    set_console_utf8();

    std::string text;
    std::string zh_dict_dir;
    bool        multilingual = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        if (a == "--text" || a == "-t") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: --text needs a value\n");
                return 1;
            }
            text = argv[++i];
            continue;
        }
        if (a == "--zh-dict-dir" || a == "-zd") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: --zh-dict-dir needs a value\n");
                return 1;
            }
            zh_dict_dir = argv[++i];
            continue;
        }
        if (a == "--multilingual") {
            multilingual = true;
            continue;
        }

        // 兼容：把第一个非参数当作 text（方便快速测试）。
        if (!a.empty() && a[0] != '-' && text.empty()) {
            text = a;
            continue;
        }

        std::fprintf(stderr, "error: unknown argument: %s\n", a.c_str());
        print_usage(argv[0]);
        return 1;
    }

    if (text.empty()) {
        std::fprintf(stderr, "error: missing --text\n");
        print_usage(argv[0]);
        return 1;
    }

    std::string out;
    if (multilingual) {
        // 说明：不加载英文 phonemizer（传 nullptr），仅用于验证“中文前端 + 多语言分段 + 单位归一化”是否符合预期。
        out = kokoro_phonemize_multilingual(text, nullptr, tts_language::ZH, zh_dict_dir);
    } else {
        out = kokoro_zh::text_to_zh_phonemes(text, zh_dict_dir);
    }
    std::fprintf(stdout, "%s\n", out.c_str());
    return 0;
}

int main(int argc, char ** argv) {
#ifdef _WIN32
    // 说明：MinGW 下直接用 main(argc, argv) 会遇到“中文参数按本地代码页编码”的问题，
    // 这里改为使用 WinAPI 获取 UTF-16 参数，再手动转 UTF-8，保证 zh_frontend 能正确解码。
    (void) argc;
    (void) argv;

    int wargc = 0;
    wchar_t ** wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    if (wargv == nullptr || wargc <= 0) {
        return main_utf8(0, nullptr);
    }

    std::vector<std::string> argv_store;
    argv_store.reserve(static_cast<size_t>(wargc));
    for (int i = 0; i < wargc; ++i) {
        argv_store.push_back(wide_to_utf8(wargv[i]));
    }
    LocalFree(wargv);

    std::vector<const char *> argv_utf8;
    argv_utf8.reserve(argv_store.size());
    for (const auto & s : argv_store) {
        argv_utf8.push_back(s.c_str());
    }

    return main_utf8(wargc, argv_utf8.data());
#else
    std::vector<const char *> argv_utf8;
    argv_utf8.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; ++i) {
        argv_utf8.push_back(argv[i]);
    }
    return main_utf8(argc, argv_utf8.data());
#endif
}
