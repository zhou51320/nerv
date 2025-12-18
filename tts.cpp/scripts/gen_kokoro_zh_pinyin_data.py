#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import re


def main() -> int:
    try:
        from pypinyin import lazy_pinyin, Style
    except Exception as exc:
        raise SystemExit(
            "Missing dependency: pypinyin\n"
            "Install: python -m pip install pypinyin"
        ) from exc

    repo_root = Path(__file__).resolve().parents[1]
    out_h = repo_root / "src" / "models" / "kokoro" / "zh_pinyin_data.h"
    out_cpp = repo_root / "src" / "models" / "kokoro" / "zh_pinyin_data.cpp"

    # Derived from pypinyin outputs (strict initials + finals tone3) + Kokoro zh_frontend needs.
    initials = [
        "",
        "b",
        "c",
        "ch",
        "d",
        "f",
        "g",
        "h",
        "j",
        "k",
        "l",
        "m",
        "n",
        "p",
        "q",
        "r",
        "s",
        "sh",
        "t",
        "x",
        "z",
        "zh",
    ]
    finals = [
        "a",
        "ai",
        "an",
        "ang",
        "ao",
        "e",
        "ei",
        "en",
        "eng",
        "er",
        "i",
        "ia",
        "ian",
        "iang",
        "iao",
        "ie",
        "ii",
        "iii",
        "in",
        "ing",
        "iong",
        "iou",
        "n",
        "o",
        "ong",
        "ou",
        "u",
        "ua",
        "uai",
        "uan",
        "uang",
        "uei",
        "uen",
        "ueng",
        "uo",
        "v",
        "van",
        "ve",
        "vn",
    ]

    ini_id = {s: i for i, s in enumerate(initials)}
    fin_id = {s: i for i, s in enumerate(finals)}

    start = 0x4E00
    end = 0x9FFF
    size = end - start + 1
    packed: list[int] = [0] * size

    for cp in range(start, end + 1):
        ch = chr(cp)
        ini = lazy_pinyin(
            ch, style=Style.INITIALS, neutral_tone_with_five=True, errors=lambda _: [""]
        )[0]
        fin = lazy_pinyin(
            ch,
            style=Style.FINALS_TONE3,
            neutral_tone_with_five=True,
            errors=lambda _: [""],
        )[0]

        # Workaround from PaddleSpeech: for "嗯" both initial and final may be empty.
        if cp == 0x55EF and not fin:
            ini = ""
            fin = "n2"

        if not fin or not fin[-1].isdigit():
            continue

        # Discriminate i/ii/iii per Mandarin apical vowels (zi/ci/si vs zhi/chi/shi/ri).
        if re.match(r"i\\d", fin):
            if ini in ("z", "c", "s"):
                fin = "ii" + fin[1:]
            elif ini in ("zh", "ch", "sh", "r"):
                fin = "iii" + fin[1:]

        base = fin[:-1]
        tone = int(fin[-1])
        if ini not in ini_id or base not in fin_id or tone <= 0 or tone > 5:
            continue

        p = (ini_id[ini] << 9) | (fin_id[base] << 3) | tone
        packed[cp - start] = p

    def cpp_string_list(values: list[str]) -> str:
        return ",\n    ".join(f"\"{v}\"" for v in values)

    def cpp_u16_table(values: list[int], per_line: int = 16) -> str:
        lines: list[str] = []
        for i in range(0, len(values), per_line):
            chunk = values[i : i + per_line]
            lines.append(", ".join(str(v) for v in chunk))
        return ",\n    ".join(lines)

    out_h.write_text(
        "\n".join(
            [
                "#pragma once",
                "",
                "#include <cstdint>",
                "#include <string_view>",
                "",
                "namespace kokoro_zh {",
                "",
                "struct zh_syllable_base {",
                "    std::string_view initial;",
                "    std::string_view final;",
                "    uint8_t          tone = 0;",
                "};",
                "",
                "bool lookup_zh_syllable(uint32_t codepoint, zh_syllable_base & out);",
                "",
                "} // namespace kokoro_zh",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )

    out_cpp.write_text(
        "\n".join(
            [
                "#include \"zh_pinyin_data.h\"",
                "",
                "namespace kokoro_zh {",
                "namespace {",
                "",
                f"static constexpr std::string_view kInitials[] = {{\n    {cpp_string_list(initials)}\n}};",
                "",
                f"static constexpr std::string_view kFinals[] = {{\n    {cpp_string_list(finals)}\n}};",
                "",
                "static constexpr uint32_t kZhPinyinStart = 0x4E00;",
                "static constexpr uint32_t kZhPinyinEnd   = 0x9FFF;",
                f"static constexpr uint32_t kZhPinyinSize  = {size};",
                "",
                f"static constexpr uint16_t kZhPinyinPacked[kZhPinyinSize] = {{\n    {cpp_u16_table(packed)}\n}};",
                "",
                "static_assert(sizeof(kZhPinyinPacked) / sizeof(kZhPinyinPacked[0]) == kZhPinyinSize);",
                "",
                "} // namespace",
                "",
                "bool lookup_zh_syllable(uint32_t codepoint, zh_syllable_base & out) {",
                "    if (codepoint < kZhPinyinStart || codepoint > kZhPinyinEnd) {",
                "        return false;",
                "    }",
                "",
                "    const uint16_t packed = kZhPinyinPacked[codepoint - kZhPinyinStart];",
                "    if (packed == 0) {",
                "        return false;",
                "    }",
                "",
                "    const uint8_t tone       =  packed        & 0x7;",
                "    const uint8_t final_id   = (packed >> 3)  & 0x3F;",
                "    const uint8_t initial_id = (packed >> 9)  & 0x1F;",
                "",
                "    out.initial = kInitials[initial_id];",
                "    out.final   = kFinals[final_id];",
                "    out.tone    = tone;",
                "    return tone >= 1 && tone <= 5;",
                "}",
                "",
                "} // namespace kokoro_zh",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )

    print(f"Wrote: {out_h}")
    print(f"Wrote: {out_cpp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
