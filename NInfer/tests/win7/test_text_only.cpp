#include "media/decode/decode.h"
#include "product/load_progress/load_progress.h"
#include "product/media_acquire/acquire.h"
#include "product/prompt_input/prompt_input.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

template <typename Function>
void expect_unsupported(Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument& error) {
        const std::string_view message(error.what());
        require(message.find("not supported") != std::string_view::npos &&
                    message.find("NINFER_TEXT_ONLY") != std::string_view::npos,
                "media rejection omitted the text-only limitation");
        return;
    }
    throw std::runtime_error("text-only build accepted media");
}

void test_disabled_media() {
    namespace acquire = ninfer::product::media_acquire;
    for (const auto kind : {acquire::SourceKind::Path, acquire::SourceKind::Url,
                            acquire::SourceKind::Data, acquire::SourceKind::Bytes}) {
        acquire::Source source;
        source.kind = kind;
        source.value = kind == acquire::SourceKind::Path ? "missing.png"
                       : kind == acquire::SourceKind::Url ? "https://example.invalid/image.png"
                                                         : "data:image/png;base64,AQID";
        source.bytes = {1, 2, 3};
        expect_unsupported([&] { acquire::acquire_bytes(source); });
    }
    expect_unsupported([&] { acquire::acquire_bytes({}); });
    const std::array<std::uint8_t, 3> bytes = {1, 2, 3};
    for (const auto input : {std::span<const std::uint8_t>{}, std::span<const std::uint8_t>{bytes}}) {
        expect_unsupported([&] { ninfer::media::decode::decode_image(input, {}); });
        expect_unsupported([&] { ninfer::media::decode::decode_video(input, {}, 2.0, 1, 8); });
    }
}

struct MessagesFile {
    std::filesystem::path path =
        std::filesystem::temp_directory_path() / "ninfer_win7_text_only_messages.json";

    ~MessagesFile() {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
    }

    void write(std::string_view json) const {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output.exceptions(std::ios::failbit | std::ios::badbit);
        output.write(json.data(), static_cast<std::streamsize>(json.size()));
    }
};

void test_text_messages() {
    const auto text = ninfer::product::prompt_from_text("hello", false);
    require(text.messages.size() == 1 && text.messages[0].parts[0].text == "hello" &&
                !text.options.enable_thinking,
            "plain text prompt changed in text-only build");

    MessagesFile file;
    file.write(R"([{"role":"system","content":"Be concise."},
                   {"role":"user","content":[{"type":"text","text":"hello"},
                                                {"type":"text","text":" world"}]}])");
    for (const bool vision_enabled : {false, true}) {
        const auto input = ninfer::product::prompt_from_messages(file.path, true, vision_enabled);
        require(input.messages.size() == 2 && input.options.enable_thinking &&
                    input.messages[0].role == ninfer::ChatRole::System &&
                    input.messages[0].parts[0].text == "Be concise." &&
                    input.messages[1].role == ninfer::ChatRole::User &&
                    input.messages[1].parts.size() == 2 &&
                    input.messages[1].parts[0].kind == ninfer::MessagePartKind::Text &&
                    input.messages[1].parts[0].text == "hello" &&
                    input.messages[1].parts[1].text == " world",
                "ordinary messages were rejected or changed by disabled media acquisition");
    }
    for (const auto type : {"image_url", "video_url"}) {
        file.write(std::string(R"([{"role":"user","content":[{"type":")") + type + "\",\"" + type +
                   R"(":{"url":"https://example.invalid/media"}}]}])");
        expect_unsupported([&] {
            (void)ninfer::product::prompt_from_messages(file.path, false, true);
        });
        try {
            (void)ninfer::product::prompt_from_messages(file.path, false, false);
        } catch (const std::invalid_argument& error) {
            require(std::string_view(error.what()).find("Vision is disabled") != std::string_view::npos,
                    "disabled Vision rejection changed");
            continue;
        }
        throw std::runtime_error("Vision-disabled messages accepted media");
    }
}

void test_progress_rendering() {
    using namespace ninfer::product;
    for (const auto mode : {LoadProgressOutputMode::Interactive, LoadProgressOutputMode::Log}) {
        std::ostringstream output;
        {
            LoadProgressRenderer renderer(output, {.mode = mode,
                                                    .min_refresh_interval = std::chrono::milliseconds(0),
                                                    .line_prefix = [] { return "[test] "; }});
            const auto progress = renderer.callback();
            progress.callback("weights", 0, 1024);
            progress.callback("weights", 512, 1024);
            progress.callback("weights", 1024, 1024);
            renderer.finish();
        }
        const auto rendered = output.str();
        require(rendered.find("[test] ") != std::string::npos &&
                    rendered.find("weights") != std::string::npos &&
                    rendered.find("50.00%") != std::string::npos &&
                    rendered.find("100.00%") != std::string::npos &&
                    rendered.find("1.00 KiB") != std::string::npos,
                "progress rendering lost its prefix, phase, percentage or byte units");
        const bool interactive = mode == LoadProgressOutputMode::Interactive;
        require(std::count(rendered.begin(), rendered.end(), '\r') == (interactive ? 3 : 0) &&
                    std::count(rendered.begin(), rendered.end(), '\n') == (interactive ? 1 : 3),
                "progress line termination changed");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        test_disabled_media();
        test_text_messages();
        test_progress_rendering();
        if (argc == 2 && std::string_view(argv[1]) == "--expect-stderr-log") {
            const auto options = ninfer::product::stderr_load_progress_options();
            require(options.mode == ninfer::product::LoadProgressOutputMode::Log &&
                        options.min_refresh_interval == std::chrono::seconds(10),
                    "redirected stderr did not select log progress output");
        }
        std::cout << "text-only media, messages and progress tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
