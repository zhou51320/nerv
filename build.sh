#!/usr/bin/env bash
set -euo pipefail

# EVA backend build script (Linux/Unix). Builds CPU and Vulkan by default; CUDA/OpenCL optional.
# Artifacts are placed into EVA_BACKEND/<arch>/<os>/<device>/<project>/
# NOTE: This script does NOT clone repositories. Prepare sources yourself.

# Defaults
PROJECTS="all"       # all|llama|whisper|sd|tts (stable-diffusion|tts)
DEVICES="auto"       # auto|cpu|vulkan|cuda|opencl|all (comma-separated allowed)
JOBS=""              # empty => cmake default; else e.g. -j 8
CLEAN=0
ROOT_DIR="$(pwd)"
EXTERN_DIR="$ROOT_DIR/external"
BUILD_DIR="$ROOT_DIR/build"
OUT_DIR="$ROOT_DIR/EVA_BACKEND"
ALL_DEVICES=()

# Optional source path overrides (env or CLI)
LLAMA_SRC_CLI="";   : "${LLAMA_SRC:=}"
WHISPER_SRC_CLI=""; : "${WHISPER_SRC:=}"
SD_SRC_CLI="";      : "${SD_SRC:=}"
TTS_SRC_CLI="";     : "${TTS_SRC:=}"

# Optional extra CMake flags (space-separated) applied globally or per-project
declare -a EVA_COMMON_CMAKE_ARGS=()
declare -a LLAMA_CMAKE_ARGS_ARR=()
declare -a WHISPER_CMAKE_ARGS_ARR=()
declare -a SD_CMAKE_ARGS_ARR=()
declare -a TTS_CMAKE_ARGS_ARR=()

if [[ -n "${EVA_CMAKE_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EVA_COMMON_CMAKE_ARGS=(${EVA_CMAKE_ARGS})
fi
if [[ -n "${LLAMA_CMAKE_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  LLAMA_CMAKE_ARGS_ARR=(${LLAMA_CMAKE_ARGS})
fi
if [[ -n "${WHISPER_CMAKE_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  WHISPER_CMAKE_ARGS_ARR=(${WHISPER_CMAKE_ARGS})
fi
if [[ -n "${SD_CMAKE_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  SD_CMAKE_ARGS_ARR=(${SD_CMAKE_ARGS})
fi
if [[ -n "${TTS_CMAKE_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  TTS_CMAKE_ARGS_ARR=(${TTS_CMAKE_ARGS})
fi

# 禁用 AVX512 加速，避免在不支持/稳定性欠佳的平台上触发
EVA_COMMON_CMAKE_ARGS+=(-DGGML_AVX512=OFF)

# Toolchain overrides for tts.cpp (optional; empty => CMake default toolchain)
: "${TTS_CC:=}"
: "${TTS_CXX:=}"
: "${TTS_USE_LIBCXX:=auto}"
: "${TTS_STATIC_STDLIB:=auto}"
: "${TTS_LIBCXX_FLAG:=-stdlib=libc++}"

# Pinned refs (only checked and warned; no automatic checkout)
LLAMA_EXPECT_REF="b6880"
WHISPER_EXPECT_TAG="v1.8.1"
SD_EXPECT_REF="0585e2609d26fc73cde0dd963127ae585ca62d49"
TTS_EXPECT_REF="e4634fb"

usage() {
  echo "Usage: $0 [-p projects] [-d devices] [-j jobs] [--clean] [--llama-src PATH] [--whisper-src PATH] [--sd-src PATH] [--tts-src PATH]"
  echo "  -p, --projects  all|llama|whisper|sd|tts (comma-separated)"
  echo "  -d, --devices   auto|cpu|vulkan|cuda|opencl|all (comma-separated)"
  echo "  -j, --jobs      parallel build jobs (passed to cmake --build --parallel)"
  echo "      --clean     remove prior build trees for selected projects/devices"
  echo "      --llama-src PATH     explicit llama.cpp source dir"
  echo "      --whisper-src PATH   explicit whisper.cpp source dir"
  echo "      --sd-src PATH        explicit stable-diffusion.cpp source dir"
  echo "      --tts-src PATH       explicit tts.cpp source dir"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -p|--projects) PROJECTS="$2"; shift 2;;
      -d|--devices)  DEVICES="$2"; shift 2;;
      -j|--jobs)     JOBS="$2"; shift 2;;
      --clean)       CLEAN=1; shift;;
      --llama-src)   LLAMA_SRC_CLI="$2"; shift 2;;
      --whisper-src) WHISPER_SRC_CLI="$2"; shift 2;;
      --sd-src)      SD_SRC_CLI="$2"; shift 2;;
      --tts-src)     TTS_SRC_CLI="$2"; shift 2;;
      -h|--help)     usage; exit 0;;
      *) echo "Unknown arg: $1"; usage; exit 1;;
    esac
  done
}

os_id() {
  local u; u=$(uname -s | tr '[:upper:]' '[:lower:]')
  case "$u" in
    linux*) echo linux;;
    darwin*) echo linux;; # normalized as linux per layout spec
    msys*|mingw*|cygwin*) echo win;;
    *) echo linux;;
  esac
}

arch_id() {
  local m; m=$(uname -m)
  case "$m" in
    x86_64|amd64) echo x86_64;;
    i386|i686) echo x86_32;;
    aarch64) echo arm64;;
    armv7l|armv7|arm) echo arm32;;
    *) echo x86_64;;
  esac
}

have() { command -v "$1" >/dev/null 2>&1; }

clang_major_version() {
  local bin="$1"
  local ver_line major=""
  ver_line=$("$bin" --version 2>/dev/null | head -n1 || true)
  if [[ "$ver_line" =~ ([0-9]+)\.[0-9]+\.[0-9]+ ]]; then
    major="${BASH_REMATCH[1]}"
  elif [[ "$ver_line" =~ version[[:space:]]([0-9]+) ]]; then
    major="${BASH_REMATCH[1]}"
  fi
  if [[ -n "$major" ]]; then
    echo "$major"
    return 0
  fi
  return 1
}

detect_clang_binary() {
  local base="$1"
  local min_major="${2:-0}"
  local versions=("" "-18" "-17" "-16" "-15" "-14" "-13" "-12" "-11" "-10")
  local candidate suffix
  for suffix in "${versions[@]}"; do
    candidate="${base}${suffix}"
    if have "$candidate"; then
      if [[ "$min_major" -gt 0 ]]; then
        local major
        major=$(clang_major_version "$candidate" 2>/dev/null || echo "")
        if [[ "$major" =~ ^[0-9]+$ && "$min_major" =~ ^[0-9]+$ ]]; then
          if (( major < min_major )); then
            continue
          fi
        elif [[ "$min_major" =~ ^[0-9]+$ && "$min_major" -gt 0 ]]; then
          # Could not determine version; skip when minimum specified.
          continue
        fi
      fi
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

# Device detection
can_vulkan() {
  if [[ -n "${VULKAN_SDK:-}" ]]; then return 0; fi
  if have vulkaninfo; then return 0; fi
  if have glslc; then return 0; fi
  if pkg-config --exists vulkan 2>/dev/null; then return 0; fi
  return 1
}

can_cuda() {
  if have nvcc; then return 0; fi
  if [[ -d /usr/local/cuda ]]; then return 0; fi
  if have nvidia-smi; then return 0; fi
  return 1
}

can_opencl() {
  if pkg-config --exists OpenCL 2>/dev/null; then return 0; fi
  if have clinfo; then return 0; fi
  return 1
}

resolve_devices() {
  local req="$DEVICES"; IFS=',' read -r -a arr <<< "$req"
  local out=()
  if [[ "${#arr[@]}" -eq 1 ]]; then
    case "${arr[0]}" in
      auto)
        out+=(cpu)
        if can_vulkan; then out+=(vulkan); fi
        ;;
      all)
        out=(cpu)
        if can_vulkan; then out+=(vulkan); fi
        if can_cuda; then out+=(cuda); fi
        if can_opencl; then out+=(opencl); fi
        ;;
      *) out=("${arr[@]}");;
    esac
  else
    out=("${arr[@]}")
  fi
  # de-dup
  local seen="" d uniq=()
  for d in "${out[@]}"; do
    if [[ ",$seen," != *",$d,"* ]]; then uniq+=("$d"); seen+="$d,"; fi
  done
  echo "${uniq[*]}"
}

resolve_src_dir() {
  # $1=name (llama|whisper|sd|tts), $2=cliOverride, $3=envVarName
  local name="$1" cli="$2" envname="$3"
  local val=""
  if [[ -n "$cli" ]]; then echo "$cli"; return 0; fi
  eval "val=\${$envname:-}"
  if [[ -n "$val" ]]; then echo "$val"; return 0; fi
  local cand
  case "$name" in
    llama)
      for cand in "$ROOT_DIR/llama.cpp" "$EXTERN_DIR/llama.cpp"; do
        if [[ -f "$cand/CMakeLists.txt" ]]; then echo "$cand"; return 0; fi
      done;;
    whisper)
      for cand in "$ROOT_DIR/whisper.cpp" "$EXTERN_DIR/whisper.cpp"; do
        if [[ -f "$cand/CMakeLists.txt" ]]; then echo "$cand"; return 0; fi
      done;;
    sd)
      for cand in "$ROOT_DIR/stable-diffusion.cpp" "$EXTERN_DIR/stable-diffusion.cpp"; do
        if [[ -f "$cand/CMakeLists.txt" ]]; then echo "$cand"; return 0; fi
      done;;
    tts)
      for cand in "$ROOT_DIR/tts.cpp" "$EXTERN_DIR/tts.cpp"; do
        if [[ -f "$cand/CMakeLists.txt" ]]; then echo "$cand"; return 0; fi
      done;;
  esac
  echo ""
}

show_version_note() {
  # $1=label, $2=path, $3=expectRef (short) or empty, $4=expectTag or empty
  local label="$1" path="$2" expectRef="$3" expectTag="$4"
  if [[ -d "$path/.git" ]] && have git; then
    local head tag
    head=$(git -C "$path" rev-parse --short HEAD 2>/dev/null || true)
    tag=$(git -C "$path" describe --tags --exact-match 2>/dev/null || true)
    local want="" have_ref="${tag:-$head}"
    if [[ -n "$expectTag" ]]; then
      if [[ "$tag" == "$expectTag" ]]; then want="(matches $expectTag)"; else want="(want $expectTag, have $have_ref)"; fi
    elif [[ -n "$expectRef" ]]; then
      if [[ "$head" == "$expectRef"* ]]; then want="(matches $expectRef)"; else want="(want $expectRef, have $have_ref)"; fi
    fi
    echo "[$label] $path ref=$have_ref $want"
  else
    echo "[$label] $path (not a git repo or git not available)"
  fi
}

# Build helpers
cmake_gen() {
  if have ninja; then echo "-G Ninja"; else echo ""; fi
}

cmake_jobs_flag() {
  if [[ -n "$JOBS" ]]; then echo "--parallel $JOBS"; else echo ""; fi
}

copy_bin() {
  local src_dir="$1" tgt_name="$2" out_dir="$3" exe_suf="$4"
  local cand
  mkdir -p "$out_dir"
  # Common candidate locations
  for cand in \
    "$src_dir/$tgt_name$exe_suf" \
    "$src_dir/bin/$tgt_name$exe_suf" \
    "$src_dir/Release/$tgt_name$exe_suf" \
    "$src_dir/bin/Release/$tgt_name$exe_suf" \
    "$src_dir/$tgt_name" \
    "$src_dir/bin/$tgt_name"
  do
    if [[ -f "$cand" ]]; then
      cp -f "$cand" "$out_dir/"
      echo "Copied $(basename "$cand") -> $out_dir"
      return 0
    fi
  done
  echo "[warn] Could not locate built binary '$tgt_name$exe_suf' under $src_dir" >&2
  return 1
}

build_llama() {
  local device="$1" os="$2" arch="$3" exe_suf="$4"
  local src
  src="$(resolve_src_dir llama "$LLAMA_SRC_CLI" LLAMA_SRC)"
  if [[ -z "$src" ]]; then
    echo "[error] llama.cpp source not found. Provide --llama-src or set LLAMA_SRC or place repo at ./llama.cpp or ./external/llama.cpp" >&2
    exit 2
  fi
  show_version_note "llama.cpp" "$src" "$LLAMA_EXPECT_REF" ""
  local bdir="$BUILD_DIR/llama.cpp/$device"
  if [[ $CLEAN -eq 1 ]]; then rm -rf "$bdir"; fi
  mkdir -p "$bdir"
  local vflag="-DGGML_VULKAN=OFF" cuflag="-DGGML_CUDA=OFF" ocflag="-DGGML_OPENCL=OFF"
  local NATIVE_EXTRA=""
  local extra_cmake=("${EVA_COMMON_CMAKE_ARGS[@]}" "${LLAMA_CMAKE_ARGS_ARR[@]}")
  case "$device" in
    vulkan) vflag="-DGGML_VULKAN=ON" ;;
    cuda)   cuflag="-DGGML_CUDA=ON"; NATIVE_EXTRA="-DGGML_NATIVE=OFF" ;;
    opencl) ocflag="-DGGML_OPENCL=ON" ;;
  esac
  cmake -S "$src" -B "$bdir" $(cmake_gen) \
    -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=ON \
    $vflag $cuflag $ocflag $NATIVE_EXTRA -DCMAKE_BUILD_TYPE=Release \
    "${extra_cmake[@]}"
  # Determine available targets and build only those
  local help_out
  help_out=$(cmake --build "$bdir" --config Release --target help 2>/dev/null || true)
  local targets=()
  for t in llama-server llama-quantize; do
    if echo "$help_out" | grep -q "$t"; then targets+=("$t"); fi
  done
  if [[ "${#targets[@]}" -eq 0 ]]; then
    echo "[warn] No expected llama.cpp targets found; building default ALL" >&2
    cmake --build "$bdir" $(cmake_jobs_flag) --config Release
  else
    cmake --build "$bdir" $(cmake_jobs_flag) --config Release --target ${targets[@]}
  fi
  local out="$OUT_DIR/$arch/$os/$device/llama.cpp"
  copy_bin "$bdir" llama-server "$out" "$exe_suf" || true
  copy_bin "$bdir" llama-quantize "$out" "$exe_suf" || true
}

build_whisper() {
  local device="$1" os="$2" arch="$3" exe_suf="$4"
  local src
  src="$(resolve_src_dir whisper "$WHISPER_SRC_CLI" WHISPER_SRC)"
  if [[ -z "$src" ]]; then
    echo "[error] whisper.cpp source not found. Provide --whisper-src or set WHISPER_SRC or place repo at ./whisper.cpp or ./external/whisper.cpp" >&2
    exit 2
  fi
  show_version_note "whisper.cpp" "$src" "" "$WHISPER_EXPECT_TAG"
  local bdir="$BUILD_DIR/whisper.cpp/$device"
  if [[ $CLEAN -eq 1 ]]; then rm -rf "$bdir"; fi
  mkdir -p "$bdir"
  local vflag="-DGGML_VULKAN=OFF" cuflag="-DGGML_CUDA=OFF" ocflag="-DGGML_OPENCL=OFF"
  local NATIVE_EXTRA=""
  local extra_cmake=("${EVA_COMMON_CMAKE_ARGS[@]}" "${WHISPER_CMAKE_ARGS_ARR[@]}")
  case "$device" in
    vulkan) vflag="-DGGML_VULKAN=ON" ;;
    cuda)   cuflag="-DGGML_CUDA=ON"; NATIVE_EXTRA="-DGGML_NATIVE=OFF" ;;
    opencl) ocflag="-DGGML_OPENCL=ON" ;;
  esac
  cmake -S "$src" -B "$bdir" $(cmake_gen) \
    -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=ON \
    $vflag $cuflag $ocflag $NATIVE_EXTRA -DCMAKE_BUILD_TYPE=Release \
    "${extra_cmake[@]}"
  cmake --build "$bdir" $(cmake_jobs_flag) --config Release --target whisper-cli
  local out="$OUT_DIR/$arch/$os/$device/whisper.cpp"
  copy_bin "$bdir" whisper-cli "$out" "$exe_suf" || true
}

build_sd() {
  local device="$1" os="$2" arch="$3" exe_suf="$4"
  local src
  src="$(resolve_src_dir sd "$SD_SRC_CLI" SD_SRC)"
  if [[ -z "$src" ]]; then
    echo "[error] stable-diffusion.cpp source not found. Provide --sd-src or set SD_SRC or place repo at ./stable-diffusion.cpp or ./external/stable-diffusion.cpp" >&2
    exit 2
  fi
  show_version_note "stable-diffusion.cpp" "$src" "$SD_EXPECT_REF" ""
  local bdir="$BUILD_DIR/stable-diffusion.cpp/$device"
  if [[ $CLEAN -eq 1 ]]; then rm -rf "$bdir"; fi
  mkdir -p "$bdir"
  local vflag="-DGGML_VULKAN=OFF" cuflag="-DGGML_CUDA=OFF" ocflag="-DGGML_OPENCL=OFF"
  local SD_EXTRA=""
  local NATIVE_EXTRA=""
  local extra_cmake=("${EVA_COMMON_CMAKE_ARGS[@]}" "${SD_CMAKE_ARGS_ARR[@]}")
  case "$device" in
    vulkan) vflag="-DGGML_VULKAN=ON"; SD_EXTRA="-DSD_VULKAN=ON";;
    cuda)   cuflag="-DGGML_CUDA=ON";   SD_EXTRA="-DSD_CUDA=ON"; NATIVE_EXTRA="-DGGML_NATIVE=OFF";;
    opencl) ocflag="-DGGML_OPENCL=ON"; SD_EXTRA="-DSD_OPENCL=ON";;
  esac
  cmake -S "$src" -B "$bdir" $(cmake_gen) \
    -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    $vflag $cuflag $ocflag $SD_EXTRA $NATIVE_EXTRA -DCMAKE_BUILD_TYPE=Release \
    "${extra_cmake[@]}"
  local help_out targets=()
  help_out=$(cmake --build "$bdir" --config Release --target help 2>/dev/null || true)
  for t in sd-cli sd-server sd; do
    if echo "$help_out" | grep -Eq "(^|[[:space:]])$t([[:space:]]|$)"; then
      targets+=("$t")
    fi
  done
  if [[ "${#targets[@]}" -eq 0 ]]; then
    cmake --build "$bdir" $(cmake_jobs_flag) --config Release
    targets=(sd sd-cli sd-server)
  else
    cmake --build "$bdir" $(cmake_jobs_flag) --config Release --target "${targets[@]}"
  fi
  local out="$OUT_DIR/$arch/$os/$device/stable-diffusion.cpp"
  for t in "${targets[@]}"; do
    copy_bin "$bdir" "$t" "$out" "$exe_suf" || true
  done
  if [[ ! -f "$out/sd$exe_suf" && ! -f "$out/sd" ]]; then
    if [[ -f "$out/sd-cli$exe_suf" ]]; then
      cp -f "$out/sd-cli$exe_suf" "$out/sd$exe_suf"
      echo "Copied sd-cli$exe_suf -> $out/sd$exe_suf"
    elif [[ -f "$out/sd-cli" ]]; then
      cp -f "$out/sd-cli" "$out/sd"
      echo "Copied sd-cli -> $out/sd"
    fi
  fi
}

build_tts() {
  local device="$1" os="$2" arch="$3" exe_suf="$4"
  local project="tts.cpp"
  local bin_name="tts-cli$exe_suf"

  # For now, ship Vulkan build in CUDA folder as a drop-in replacement.
  if [[ "$device" == "cuda" ]]; then
    if ! can_vulkan; then
      echo "[warn] tts.cpp: Vulkan not detected; falling back to CPU binary for device 'cuda'" >&2
      build_tts cpu "$os" "$arch" "$exe_suf"
      local cpu_out="$OUT_DIR/$arch/$os/cpu/$project"
      local cpu_bin="$cpu_out/$bin_name"
      if [[ -f "$cpu_bin" ]]; then
        local cuda_out="$OUT_DIR/$arch/$os/cuda/$project"
        mkdir -p "$cuda_out"
        cp -f "$cpu_bin" "$cuda_out/"
        echo "Copied $(basename "$cpu_bin") -> $cuda_out"
      else
        echo "[warn] tts.cpp: CPU binary unavailable; cannot populate device 'cuda'" >&2
      fi
      return 0
    fi

    local vk_out="$OUT_DIR/$arch/$os/vulkan/$project"
    local vk_bin="$vk_out/$bin_name"
    if [[ $CLEAN -eq 1 || ! -f "$vk_bin" ]]; then
      build_tts vulkan "$os" "$arch" "$exe_suf"
    fi
    local cuda_out="$OUT_DIR/$arch/$os/cuda/$project"
    mkdir -p "$cuda_out"
    if [[ -f "$vk_bin" ]]; then
      cp -f "$vk_bin" "$cuda_out/"
      echo "Copied $(basename "$vk_bin") -> $cuda_out"
    else
      echo "[warn] tts.cpp: Vulkan binary unavailable; cannot populate device 'cuda'" >&2
    fi
    return 0
  fi

  # Keep OpenCL builds CPU-only for now (populate OpenCL dir from CPU build).
  if [[ "$device" == "opencl" ]]; then
    build_tts cpu "$os" "$arch" "$exe_suf"
    local cpu_out="$OUT_DIR/$arch/$os/cpu/$project"
    local cpu_bin="$cpu_out/$bin_name"
    local target_dir="$OUT_DIR/$arch/$os/$device/$project"
    if [[ -f "$cpu_bin" ]]; then
      mkdir -p "$target_dir"
      cp -f "$cpu_bin" "$target_dir/"
      echo "Copied $(basename "$cpu_bin") -> $target_dir"
    else
      echo "[warn] tts.cpp: CPU binary unavailable; cannot populate device '$device'" >&2
    fi
    return 0
  fi

  local src
  src="$(resolve_src_dir tts "$TTS_SRC_CLI" TTS_SRC)"
  if [[ -z "$src" ]]; then
    echo "[error] tts.cpp source not found. Provide --tts-src or set TTS_SRC or place repo at ./tts.cpp or ./external/tts.cpp" >&2
    exit 2
  fi
  show_version_note "tts.cpp" "$src" "$TTS_EXPECT_REF" ""
  local bdir="$BUILD_DIR/tts.cpp/$device"
  if [[ $CLEAN -eq 1 ]]; then rm -rf "$bdir"; fi
  local desired_cc="${TTS_CC:-}"
  local desired_cxx="${TTS_CXX:-}"
  mkdir -p "$bdir"
  if [[ -f "$bdir/CMakeCache.txt" && -n "$desired_cxx" ]]; then
    local cache_cxx
    cache_cxx=$(grep -E '^CMAKE_CXX_COMPILER:FILEPATH=' "$bdir/CMakeCache.txt" 2>/dev/null | cut -d= -f2 || true)
    if [[ -n "$cache_cxx" && "$cache_cxx" != *"$(basename "$desired_cxx")" ]]; then
      echo "[info] tts.cpp: switching compiler to $desired_cxx; clearing previous cache ($cache_cxx)"
      rm -rf "$bdir"
      mkdir -p "$bdir"
    fi
  fi
  if [[ -n "$desired_cc" ]] && ! have "$desired_cc"; then
    echo "[error] tts.cpp: requested compiler '$desired_cc' not found in PATH" >&2
    exit 3
  fi
  if [[ -n "$desired_cxx" ]] && ! have "$desired_cxx"; then
    echo "[error] tts.cpp: requested compiler '$desired_cxx' not found in PATH" >&2
    exit 3
  fi
  local cmake_env=()
  if [[ -n "$desired_cc" ]]; then cmake_env+=(CC="$desired_cc"); fi
  if [[ -n "$desired_cxx" ]]; then cmake_env+=(CXX="$desired_cxx"); fi
  if (( ${#cmake_env[@]} )); then
    echo "[info] tts.cpp: using CC='$desired_cc' CXX='$desired_cxx'"
  fi
  local host_uname
  host_uname=$(uname -s)
  local cmake_flag_args=()
  local cxx_flags=()
  local link_flags=()
  local use_libcxx_raw="${TTS_USE_LIBCXX:-}"
  local use_libcxx_lc
  use_libcxx_lc=$(printf '%s' "$use_libcxx_raw" | tr '[:upper:]' '[:lower:]')
  local use_libcxx=""
  case "$use_libcxx_lc" in
    1|true|yes|on) use_libcxx="1";;
    0|false|no|off) use_libcxx="0";;
    auto|"") if [[ "$host_uname" == "Darwin" ]]; then use_libcxx="1"; else use_libcxx="0"; fi;;
    *) use_libcxx="0";;
  esac
  if [[ "$use_libcxx" == "1" ]]; then
    echo "[info] tts.cpp: linking with libc++ runtime"
    cxx_flags+=("$TTS_LIBCXX_FLAG")
    link_flags+=("$TTS_LIBCXX_FLAG")
  else
    echo "[info] tts.cpp: linking with libstdc++ runtime"
  fi
  local static_stdlib_raw="${TTS_STATIC_STDLIB:-auto}"
  local static_stdlib_lc
  static_stdlib_lc=$(printf '%s' "$static_stdlib_raw" | tr '[:upper:]' '[:lower:]')
  local static_stdlib=""
  case "$static_stdlib_lc" in
    1|true|yes|on) static_stdlib="1";;
    0|false|no|off) static_stdlib="0";;
    auto|"")
      if [[ "$os" == "win" ]]; then
        static_stdlib="0"
      elif [[ "$use_libcxx" == "1" ]]; then
        static_stdlib="0"
      else
        static_stdlib="1"
      fi
      ;;
    *) static_stdlib="0";;
  esac
  if [[ "$static_stdlib" == "1" ]]; then
    if [[ "$use_libcxx" == "1" ]]; then
      echo "[warn] tts.cpp: static stdlib requested but libc++ selected; resulting binary may still depend on libc++.so" >&2
    else
      link_flags+=("-static-libstdc++" "-static-libgcc")
      echo "[info] tts.cpp: enabling static libstdc++/libgcc linking"
    fi
  fi
  if [[ -n "${TTS_EXTRA_CXX_FLAGS:-}" ]]; then
    cxx_flags+=("${TTS_EXTRA_CXX_FLAGS}")
  fi
  if [[ -n "${TTS_EXTRA_LINKER_FLAGS:-}" ]]; then
    link_flags+=("${TTS_EXTRA_LINKER_FLAGS}")
  fi
  if (( ${#cxx_flags[@]} )); then
    local cxx_flags_str
    printf -v cxx_flags_str '%s ' "${cxx_flags[@]}"
    cxx_flags_str="${cxx_flags_str% }"
    cmake_flag_args+=(-DCMAKE_CXX_FLAGS="$cxx_flags_str")
  fi
  if (( ${#link_flags[@]} )); then
    local link_flags_str
    printf -v link_flags_str '%s ' "${link_flags[@]}"
    link_flags_str="${link_flags_str% }"
    cmake_flag_args+=(-DCMAKE_EXE_LINKER_FLAGS="$link_flags_str")
  fi
  local vflag="-DGGML_VULKAN=OFF" cuflag="-DGGML_CUDA=OFF" ocflag="-DGGML_OPENCL=OFF"
  local native_extra=""
  case "$device" in
    vulkan) vflag="-DGGML_VULKAN=ON" ;;
  esac
  local extra_cmake=("${EVA_COMMON_CMAKE_ARGS[@]}" "${TTS_CMAKE_ARGS_ARR[@]}")
  "${cmake_env[@]}" cmake -S "$src" -B "$bdir" $(cmake_gen) \
    -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DTTS_BUILD_EXAMPLES=ON \
    $vflag $cuflag $ocflag $native_extra -DCMAKE_BUILD_TYPE=Release \
    "${extra_cmake[@]}" \
    "${cmake_flag_args[@]}"
  cmake --build "$bdir" $(cmake_jobs_flag) --config Release --target tts-cli
  local out="$OUT_DIR/$arch/$os/$device/$project"
  copy_bin "$bdir" tts-cli "$out" "$exe_suf" || true

  if [[ "$device" == "vulkan" ]]; then
    local vk_bin_path="$out/$bin_name"
    if [[ -f "$vk_bin_path" ]]; then
      local cuda_dir="$OUT_DIR/$arch/$os/cuda/$project"
      mkdir -p "$cuda_dir"
      cp -f "$vk_bin_path" "$cuda_dir/"
      echo "Copied $(basename "$vk_bin_path") -> $cuda_dir"
    else
      echo "[warn] tts.cpp: Vulkan binary missing after build; cannot populate CUDA dir" >&2
    fi
  fi
}

main() {
  parse_args "$@"
  local OS=$(os_id) ARCH=$(arch_id)
  local EXE_SUF=""; if [[ "$OS" == "win" ]]; then EXE_SUF=".exe"; fi
  BUILD_DIR="$ROOT_DIR/build-$ARCH-$OS"

  IFS=' ' read -r -a DEV_ARR <<< "$(resolve_devices)"
  ALL_DEVICES=("${DEV_ARR[@]}")

  # Resolve project list
  local PROJ_ARR=()
  if [[ "$PROJECTS" == "all" ]]; then
    PROJ_ARR=(llama whisper sd tts)
  else
    IFS=',' read -r -a PROJ_ARR <<< "$PROJECTS"
  fi

  echo "==> OS=$OS ARCH=$ARCH DEVICES=${DEV_ARR[*]} PROJECTS=${PROJ_ARR[*]}"

  for dev in "${DEV_ARR[@]}"; do
    for proj in "${PROJ_ARR[@]}"; do
      echo "--- Building $proj [$dev] ---"
      case "$proj" in
        llama)   build_llama   "$dev" "$OS" "$ARCH" "$EXE_SUF";;
        whisper) build_whisper "$dev" "$OS" "$ARCH" "$EXE_SUF";;
        sd)      build_sd      "$dev" "$OS" "$ARCH" "$EXE_SUF";;
        tts)     build_tts     "$dev" "$OS" "$ARCH" "$EXE_SUF";;
        *) echo "Unknown project: $proj"; exit 1;;
      esac
    done
  done

  echo "Done. Artifacts under: $OUT_DIR/$ARCH/$OS"
}

main "$@"
