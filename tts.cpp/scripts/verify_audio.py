#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比两份 WAV 音频的差异，输出关键指标并根据阈值决定是否失败。
说明：
- 仅依赖 Python 标准库，适合在本项目中直接使用。
- 主要用于回归验证：对比“参考音频”和“新生成音频”的质量差异。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import wave
from array import array
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class WavData:
    # WAV 基本信息
    sample_rate: int
    channels: int
    sample_width: int
    # 通道数据（浮点，范围约 [-1, 1]）
    samples_by_channel: List[List[float]]


@dataclass
class Metrics:
    # 对齐后的样本数
    samples: int
    # 绝对差相关指标
    mae: float
    rmse: float
    max_abs_diff: float
    # 信噪比（以参考音频能量为基准）
    snr_db: float
    # 相关系数（越接近 1 越相似）
    corr: float


def _decode_pcm(data: bytes, sample_width: int) -> List[float]:
    """解码 PCM 数据为浮点数组（范围约 [-1, 1]）。"""
    if sample_width == 1:
        # 8-bit PCM 为无符号
        arr = array("B")
        arr.frombytes(data)
        return [(v - 128) / 128.0 for v in arr]
    if sample_width == 2:
        arr = array("h")
        arr.frombytes(data)
        if sys.byteorder != "little":
            arr.byteswap()
        return [v / 32768.0 for v in arr]
    if sample_width == 3:
        # 24-bit PCM：手动解析并符号扩展
        out: List[float] = []
        for i in range(0, len(data), 3):
            b0 = data[i]
            b1 = data[i + 1]
            b2 = data[i + 2]
            val = b0 | (b1 << 8) | (b2 << 16)
            if val & 0x800000:
                val -= 1 << 24
            out.append(val / 8388608.0)
        return out
    if sample_width == 4:
        arr = array("i")
        arr.frombytes(data)
        if sys.byteorder != "little":
            arr.byteswap()
        return [v / 2147483648.0 for v in arr]
    raise ValueError(f"不支持的 sample_width: {sample_width}")


def _split_channels(samples: List[float], channels: int) -> List[List[float]]:
    """把交错的 PCM 样本拆为按通道排列。"""
    if channels <= 0:
        raise ValueError("channels 必须大于 0")
    if channels == 1:
        return [samples]
    frames = len(samples) // channels
    out = [[] for _ in range(channels)]
    for i in range(frames):
        base = i * channels
        for ch in range(channels):
            out[ch].append(samples[base + ch])
    return out


def read_wav(path: str) -> WavData:
    """读取 WAV 文件并返回浮点样本。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with wave.open(path, "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frames = wf.getnframes()
        raw = wf.readframes(frames)
    samples = _decode_pcm(raw, sample_width)
    samples_by_channel = _split_channels(samples, channels)
    return WavData(
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        samples_by_channel=samples_by_channel,
    )


def _trim_silence(samples_by_channel: List[List[float]], threshold_db: float) -> List[List[float]]:
    """按阈值裁剪首尾静音，返回裁剪后的通道数据。"""
    if threshold_db is None:
        return samples_by_channel
    threshold = 10 ** (threshold_db / 20.0)
    if threshold <= 0:
        return samples_by_channel
    n = min(len(ch) for ch in samples_by_channel)
    if n == 0:
        return samples_by_channel
    start = 0
    end = n - 1
    while start < n:
        if any(abs(ch[start]) > threshold for ch in samples_by_channel):
            break
        start += 1
    while end > start:
        if any(abs(ch[end]) > threshold for ch in samples_by_channel):
            break
        end -= 1
    return [ch[start : end + 1] for ch in samples_by_channel]


def _find_best_shift(ref: List[float], test: List[float], max_shift: int, window: int) -> Tuple[int, float]:
    """在 [-max_shift, +max_shift] 内找相关系数最高的对齐偏移。"""
    if max_shift <= 0:
        return 0, 1.0
    window = min(window, len(ref), len(test))
    if window <= 0:
        return 0, 0.0
    best_shift = 0
    best_corr = -1.0
    eps = 1e-12
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            if shift + window > len(test):
                continue
            r_start = 0
            t_start = shift
        else:
            if -shift + window > len(ref):
                continue
            r_start = -shift
            t_start = 0
        sum_ref = 0.0
        sum_test = 0.0
        sum_prod = 0.0
        for i in range(window):
            rv = ref[r_start + i]
            tv = test[t_start + i]
            sum_ref += rv * rv
            sum_test += tv * tv
            sum_prod += rv * tv
        denom = math.sqrt(sum_ref * sum_test) + eps
        corr = sum_prod / denom
        if corr > best_corr:
            best_corr = corr
            best_shift = shift
    return best_shift, best_corr


def _apply_shift(ref: List[float], test: List[float], shift: int) -> Tuple[List[float], List[float]]:
    """按偏移量裁剪两段序列，返回对齐后的重叠区间。"""
    if shift >= 0:
        ref_aligned = ref
        test_aligned = test[shift:]
    else:
        ref_aligned = ref[-shift:]
        test_aligned = test
    n = min(len(ref_aligned), len(test_aligned))
    return ref_aligned[:n], test_aligned[:n]


def _compute_metrics(ref: List[float], test: List[float]) -> Metrics:
    """计算对齐样本的差异指标。"""
    n = min(len(ref), len(test))
    if n <= 0:
        return Metrics(samples=0, mae=0.0, rmse=0.0, max_abs_diff=0.0, snr_db=float("-inf"), corr=0.0)
    sum_ref = 0.0
    sum_test = 0.0
    sum_diff = 0.0
    sum_abs = 0.0
    sum_prod = 0.0
    max_abs = 0.0
    for i in range(n):
        rv = ref[i]
        tv = test[i]
        diff = rv - tv
        sum_ref += rv * rv
        sum_test += tv * tv
        sum_prod += rv * tv
        sum_diff += diff * diff
        ad = abs(diff)
        sum_abs += ad
        if ad > max_abs:
            max_abs = ad
    rmse = math.sqrt(sum_diff / n)
    mae = sum_abs / n
    denom = math.sqrt(sum_ref * sum_test) + 1e-12
    corr = sum_prod / denom
    rms_ref = math.sqrt(sum_ref / n)
    snr_db = float("inf") if rmse <= 0 else 20.0 * math.log10((rms_ref + 1e-12) / rmse)
    return Metrics(samples=n, mae=mae, rmse=rmse, max_abs_diff=max_abs, snr_db=snr_db, corr=corr)


def _format_ms(samples: int, sample_rate: int) -> float:
    return 0.0 if sample_rate <= 0 else samples * 1000.0 / sample_rate


def main() -> int:
    parser = argparse.ArgumentParser(description="对比两份 WAV 音频差异并给出质量评估。")
    parser.add_argument("--ref", required=True, help="参考音频路径（基准）。")
    parser.add_argument("--test", required=True, help="待比较音频路径。")
    parser.add_argument("--trim-silence-db", type=float, default=-60.0,
                        help="裁剪首尾静音阈值（dBFS，负值）。默认 -60。")
    parser.add_argument("--max-shift-ms", type=float, default=20.0,
                        help="允许的最大对齐偏移（毫秒）。默认 20。")
    parser.add_argument("--align-window-ms", type=float, default=500.0,
                        help="对齐时使用的窗口长度（毫秒）。")
    parser.add_argument("--max-length-diff-ms", type=float, default=50.0,
                        help="允许的最大时长差异（毫秒）。默认 50。")
    parser.add_argument("--snr-min", type=float, default=0.0, help="最小可接受 SNR（dB），默认 0。")
    parser.add_argument("--rmse-max", type=float, default=0.08, help="最大可接受 RMSE，默认 0.08。")
    parser.add_argument("--mae-max", type=float, default=0.04, help="最大可接受 MAE，默认 0.04。")
    parser.add_argument("--max-abs-diff", type=float, default=1.1, help="最大可接受峰值差异，默认 1.1。")
    parser.add_argument("--allow-format-mismatch", action="store_true",
                        help="允许采样率/通道数/位深不同（不推荐）。")
    parser.add_argument("--per-channel", action="store_true", help="输出每个声道的详细指标。")
    args = parser.parse_args()

    ref = read_wav(args.ref)
    test = read_wav(args.test)

    format_ok = (ref.sample_rate == test.sample_rate and
                 ref.channels == test.channels and
                 ref.sample_width == test.sample_width)
    if not format_ok and not args.allow_format_mismatch:
        print("格式不一致：")
        print(f"  ref : {ref.sample_rate}Hz, {ref.channels}ch, {ref.sample_width * 8}bit")
        print(f"  test: {test.sample_rate}Hz, {test.channels}ch, {test.sample_width * 8}bit")
        return 2

    ref_samples = ref.samples_by_channel
    test_samples = test.samples_by_channel

    if args.trim_silence_db is not None:
        ref_samples = _trim_silence(ref_samples, args.trim_silence_db)
        test_samples = _trim_silence(test_samples, args.trim_silence_db)

    # 时长差异检查（使用裁剪后的长度）
    ref_len = min(len(ch) for ch in ref_samples)
    test_len = min(len(ch) for ch in test_samples)
    length_diff_ms = abs(_format_ms(ref_len, ref.sample_rate) - _format_ms(test_len, test.sample_rate))
    if length_diff_ms > args.max_length_diff_ms:
        print(f"时长差异过大：{length_diff_ms:.2f} ms > {args.max_length_diff_ms:.2f} ms")
        return 3

    # 对齐（基于第一个声道）
    max_shift_samples = int(args.max_shift_ms * ref.sample_rate / 1000.0)
    align_window_samples = int(args.align_window_ms * ref.sample_rate / 1000.0)
    shift, corr = _find_best_shift(ref_samples[0], test_samples[0], max_shift_samples, align_window_samples)

    metrics_list: List[Metrics] = []
    for ch in range(min(len(ref_samples), len(test_samples))):
        r_aligned, t_aligned = _apply_shift(ref_samples[ch], test_samples[ch], shift)
        metrics_list.append(_compute_metrics(r_aligned, t_aligned))

    worst_snr = min(m.snr_db for m in metrics_list)
    worst_rmse = max(m.rmse for m in metrics_list)
    worst_mae = max(m.mae for m in metrics_list)
    worst_max = max(m.max_abs_diff for m in metrics_list)
    worst_corr = min(m.corr for m in metrics_list)

    print("对比结果：")
    print(f"  对齐偏移: {shift} samples ({_format_ms(shift, ref.sample_rate):.2f} ms), corr={corr:.6f}")
    print(f"  样本数  : {metrics_list[0].samples} (以对齐后的重叠区间为准)")
    print(f"  SNR     : {worst_snr:.2f} dB")
    print(f"  RMSE    : {worst_rmse:.6f}")
    print(f"  MAE     : {worst_mae:.6f}")
    print(f"  MaxDiff : {worst_max:.6f}")
    print(f"  Corr    : {worst_corr:.6f}")

    if args.per_channel:
        for i, m in enumerate(metrics_list):
            print(f"  [ch{i}] SNR={m.snr_db:.2f}dB RMSE={m.rmse:.6f} MAE={m.mae:.6f} Max={m.max_abs_diff:.6f} Corr={m.corr:.6f}")

    failed = False
    if worst_snr < args.snr_min:
        print(f"FAIL: SNR {worst_snr:.2f} dB < {args.snr_min:.2f} dB")
        failed = True
    if worst_rmse > args.rmse_max:
        print(f"FAIL: RMSE {worst_rmse:.6f} > {args.rmse_max:.6f}")
        failed = True
    if worst_mae > args.mae_max:
        print(f"FAIL: MAE {worst_mae:.6f} > {args.mae_max:.6f}")
        failed = True
    if worst_max > args.max_abs_diff:
        print(f"FAIL: MaxDiff {worst_max:.6f} > {args.max_abs_diff:.6f}")
        failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
