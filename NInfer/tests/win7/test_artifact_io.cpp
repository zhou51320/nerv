#include "artifact/reader.h"
#include "artifact_fixture.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winioctl.h>
#endif

namespace {

using ninfer::artifact::ArtifactError;
using ninfer::artifact::Reader;
using namespace ninfer::test::artifact_fixture;

void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

template <typename Function>
void expect_artifact_error(Function&& function) {
    try {
        function();
    } catch (const ArtifactError&) { return; }
    throw std::runtime_error("invalid artifact input was accepted");
}

Json directory(std::uint64_t bytes = 4096 + 37) {
    return {
        {"identity", {{"model_id", "io-fixture"}, {"weights_id", "raw"}}},
        {"objects", Json::array({{{"name", "payload"},
                                  {"kind", "resource"},
                                  {"encoding", "raw-bytes-v1"},
                                  {"offset", 0},
                                  {"bytes", bytes}}})},
    };
}

bool filled(std::span<const std::byte> data, std::byte value) {
    return std::all_of(data.begin(), data.end(), [value](std::byte byte) { return byte == value; });
}

void test_unicode_tail_and_ownership() {
    auto fixture = write_fixture(directory(), "win7_io");
    const auto unicode_path = fixture.path.parent_path() /
                              std::filesystem::path(u8"ninfer_\xE6\xA8\xA1\xE5\x9E\x8B_\U0001f4be.ninfer");
    std::filesystem::rename(fixture.path, unicode_path);
    fixture.path = unicode_path;
    {
        Reader original(fixture.path);
        const auto saved = original.payload("payload");
        Reader moved(std::move(original));
        Reader reader(fixture.path);
        reader = std::move(moved);
        require(saved.data.data() == reader.payload("payload").data.data() &&
                    saved.absolute_offset == 4096 && saved.data.size() == 4096 + 37 &&
                    filled(saved.data, std::byte{1}),
                "mapped payload did not survive Reader moves");

        alignas(Reader::direct_io_alignment) std::array<std::byte, 8192> buffer{};
        const auto bytes = reader.read_direct(saved.absolute_offset, buffer);
        require(bytes == saved.data.size() &&
                    std::equal(saved.data.begin(), saved.data.end(), buffer.begin()),
                "aligned read did not return exact payload bytes through EOF");
        const auto tail = reader.read_direct(saved.absolute_offset + 4096, buffer);
        require(tail == 37 && filled(std::span(buffer).first(tail), std::byte{1}),
                "unaligned file tail was not read exactly");
        require(reader.read_direct(saved.absolute_offset + 8192, buffer) == 0,
                "read past EOF must return zero");
        require(reader.read_direct(0, {}) == 0, "empty read must return zero");
        expect_artifact_error([&] { reader.read_direct(1, buffer); });
        expect_artifact_error([&] { reader.read_direct(0, std::span(buffer).first(1)); });
        expect_artifact_error([&] { reader.read_direct(0, std::span(buffer).subspan(1, 4096)); });
        expect_artifact_error([&] { reader.read_direct(std::uint64_t{1} << 63, buffer); });
        expect_artifact_error([&] { reader.payload("missing"); });
    }
    // On Windows an unreleased mapping/file handle would prevent this writable open.
    std::ofstream replacement(fixture.path, std::ios::binary | std::ios::trunc);
    require(bool(replacement), "Reader destruction retained a file or mapping handle");
}

void test_failed_construction_releases_file() {
    for (int scenario = 0; scenario < 7; ++scenario) {
        auto fixture = write_fixture(directory(), "win7_invalid_io");
        if (scenario <= 2) {
            const std::array<std::uint64_t, 3> sizes = {0, 15, 4096 + 4096 + 36};
            std::filesystem::resize_file(fixture.path, sizes[scenario]);
        } else {
            std::fstream output(fixture.path, std::ios::binary | std::ios::in | std::ios::out);
            output.exceptions(std::ios::failbit | std::ios::badbit);
            if (scenario == 3) {
                output.seekp(16);
                output.put('?');
            } else {
                const std::array<std::uint64_t, 3> sizes = {
                    0, std::numeric_limits<std::uint64_t>::max(), 1ULL << 32,
                };
                std::array<std::byte, 8> encoded{};
                write_u64_le(encoded.data(), sizes[scenario - 4]);
                output.seekp(8);
                output.write(reinterpret_cast<const char*>(encoded.data()), encoded.size());
            }
        }
        expect_artifact_error([&] { Reader reader(fixture.path); });
        std::ofstream replacement(fixture.path, std::ios::binary | std::ios::trunc);
        require(bool(replacement), "failed construction retained a file or mapping handle");
    }
}

void mark_sparse(const std::filesystem::path& path) {
#ifdef _WIN32
    const HANDLE file = ::CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING,
                                     FILE_ATTRIBUTE_NORMAL, nullptr);
    require(file != INVALID_HANDLE_VALUE, "failed to open sparse fixture");
    DWORD bytes = 0;
    const BOOL ok = ::DeviceIoControl(file, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &bytes, nullptr);
    ::CloseHandle(file);
    require(ok != FALSE, "large-offset test requires a sparse-capable filesystem (NTFS)");
#else
    (void)path;
#endif
}

void test_large_sparse_offsets() {
    constexpr std::uint64_t far_offset = (16ULL << 30) + 4096;
    constexpr std::size_t tail_bytes = 37;
    auto metadata = directory(32);
    metadata["objects"].push_back({{"name", "far"},
                                   {"kind", "resource"},
                                   {"encoding", "raw-bytes-v1"},
                                   {"offset", far_offset},
                                   {"bytes", tail_bytes}});
    const auto json = metadata.dump();
    const auto payload_start = align_up(16 + json.size(), 4096);
    TemporaryArtifact fixture{std::filesystem::temp_directory_path() / "ninfer_win7_sparse.ninfer"};
    {
        std::ofstream output(fixture.path, std::ios::binary | std::ios::trunc);
        output.exceptions(std::ios::failbit | std::ios::badbit);
        output.write(reinterpret_cast<const char*>(kMagic.data()), kMagic.size());
        std::array<std::byte, 8> encoded{};
        write_u64_le(encoded.data(), json.size());
        output.write(reinterpret_cast<const char*>(encoded.data()), encoded.size());
        output.write(json.data(), static_cast<std::streamsize>(json.size()));
    }
    mark_sparse(fixture.path);
    const auto far_absolute = payload_start + far_offset;
    {
        std::fstream output(fixture.path, std::ios::binary | std::ios::in | std::ios::out);
        output.exceptions(std::ios::failbit | std::ios::badbit);
        std::array<char, 32> first{};
        first.fill(1);
        output.seekp(static_cast<std::streamoff>(payload_start));
        output.write(first.data(), first.size());
        std::array<char, tail_bytes> last{};
        last.fill(0x7b);
        output.seekp(static_cast<std::streamoff>(far_absolute));
        output.write(last.data(), last.size());
    }
    Reader reader(fixture.path);
    const auto far = reader.payload("far");
    require(reader.file_bytes() == far_absolute + tail_bytes &&
                far.absolute_offset == far_absolute && far.data.size() == tail_bytes &&
                filled(far.data, std::byte{0x7b}) && filled(reader.payload("payload").data, std::byte{1}),
            "mapped payload or file size truncated a 64-bit offset");

    // Cross the Windows 64 MiB ReadFile chunk boundary, but never materialize the 16 GiB hole.
    constexpr std::size_t chunk = 64ULL << 20;
    constexpr std::size_t alignment = Reader::direct_io_alignment;
    std::vector<std::byte> storage(chunk + 2 * alignment);
    const auto padding = (alignment - reinterpret_cast<std::uintptr_t>(storage.data()) % alignment) %
                         alignment;
    const auto buffer = std::span(storage).subspan(padding, chunk + alignment);
    const auto bytes = reader.read_direct(far_absolute - chunk, buffer);
    require(bytes == chunk + tail_bytes && filled(buffer.first(chunk), std::byte{0}) &&
                filled(buffer.subspan(chunk, tail_bytes), std::byte{0x7b}),
            "large-offset read lost a chunk or EOF tail");
}

} // namespace

int main() {
    try {
        test_unicode_tail_and_ownership();
        test_failed_construction_releases_file();
        test_large_sparse_offsets();
        std::cout << "artifact I/O tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
