# 仓库指南

- tts.cpp 是一个基于 ggml 的 文字转语音推理库，用于在本地设备上运行开源的文本转语音（TTS）模型
- 以kokoro模型为准，支持中/英
- 语言标准：C++17，尽量不引入第三方库
- 注释要求：使用中文对代码进行详细注释
- 每次完成任务，将实现的内容简洁写入到docs/功能迭代.md中，格式为 xxxx年-xx月-xx日：xxx，新内容记录在文档顶部。
- 每次修改推理流程或模型结构等后，都应自行运行 build\bin\tts-cli.exe --model-path models\Kokoro-82M-v1_1-zh_F16.gguf --voice "zf_001" --bench 生成音频，并使用 python scripts/verify_audio.py --ref bench_zf_001.wav  --test TTS.cpp.wav 进行验证，确保音质不下降。
- 不要尝试创建git分支或提交，这些交给用户
- 为了方便更新，尽可能不要改动ggml的代码
