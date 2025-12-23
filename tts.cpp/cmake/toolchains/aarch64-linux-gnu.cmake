# 用于在 x86_64 Linux（GitHub Actions Ubuntu 20.04）上交叉编译到 aarch64 的工具链文件。
# 目标：生成 ubuntu20.04 aarch64 的 tts-cli（二进制），供 nightly 发布使用。

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# 交叉编译器（来自 gcc-aarch64-linux-gnu / g++-aarch64-linux-gnu）
set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# 交叉编译时需要注意：
# - 程序（如 glslc / python 等宿主机工具）应在宿主机查找；
# - 库/头文件等目标依赖应交由编译器/链接器的默认搜索规则处理。
#
# 这里显式要求 find_program 仅在宿主机搜索，避免误从目标前缀中寻找可执行文件。
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

