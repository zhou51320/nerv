import ast
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import gguf
import numpy as np
import torch

from .tts_encoder import TTSEncoder

COSYVOICE3_ARCHITECTURE = "cosyvoice3"


def _safe_torch_load(path: Path) -> Dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # 兼容旧版 torch（不支持 weights_only）
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        return state["state_dict"]
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        return state["model"]
    if not isinstance(state, dict):
        raise ValueError(f"Expected state dict at {path}, got {type(state)}")
    return state


def _extract_cosyvoice3_special_tokens(tokenizer_py: Path) -> List[str]:
    text = tokenizer_py.read_text(encoding="utf-8")
    mod = ast.parse(text)

    class Finder(ast.NodeVisitor):
        def __init__(self) -> None:
            self.tokens: Optional[List[str]] = None

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            if node.name != "CosyVoice3Tokenizer":
                return
            for stmt in node.body:
                if not isinstance(stmt, ast.FunctionDef) or stmt.name != "__init__":
                    continue
                for inner in stmt.body:
                    if not isinstance(inner, ast.Assign):
                        continue
                    for tgt in inner.targets:
                        if not isinstance(tgt, ast.Name) or tgt.id != "special_tokens":
                            continue
                        if not isinstance(inner.value, ast.Dict):
                            continue
                        for k, v in zip(inner.value.keys, inner.value.values):
                            if isinstance(k, ast.Constant) and k.value == "additional_special_tokens":
                                if isinstance(v, (ast.List, ast.Tuple)):
                                    self.tokens = [elt.value for elt in v.elts]

    finder = Finder()
    finder.visit(mod)
    if finder.tokens is None:
        raise ValueError(f"Failed to find CosyVoice3Tokenizer.additional_special_tokens in {tokenizer_py}")
    return finder.tokens


def _load_qwen2_config(vocab_dir: Path) -> Dict[str, object]:
    config_path = vocab_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing Qwen2 config.json in {vocab_dir}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _load_bpe_merges(merges_path: Path) -> List[str]:
    merges = []
    for line in merges_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        merges.append(line.strip())
    return merges


def _build_token_list(
    vocab_dir: Path,
    additional_special_tokens: List[str],
    target_vocab_size: int,
    logger: logging.Logger,
) -> List[str]:
    special_tokens = {
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "additional_special_tokens": additional_special_tokens,
    }

    tokens: List[str] = []
    try:
        from transformers import AutoTokenizer  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(str(vocab_dir))
        tokenizer.add_special_tokens(special_tokens)
        vocab = tokenizer.get_vocab()
        max_id = max(vocab.values())
        tokens = [""] * (max_id + 1)
        for tok, idx in vocab.items():
            tokens[idx] = tok
    except Exception as exc:
        # 兜底路径：按 vocab.json + tokenizer_config.json 组合，并拼接 special tokens
        logger.warning("AutoTokenizer 不可用或加载失败，使用简化 BPE 词表构建：%s", exc)
        vocab_path = vocab_dir / "vocab.json"
        if not vocab_path.exists():
            raise FileNotFoundError(f"Missing vocab.json in {vocab_dir}")
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        tokens = [""] * len(vocab)
        for tok, idx in vocab.items():
            tokens[idx] = tok

        config_path = vocab_dir / "tokenizer_config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            added = config.get("added_tokens_decoder", {})
            for _, info in sorted(added.items(), key=lambda kv: int(kv[0])):
                content = info.get("content", "")
                if content and content not in tokens:
                    tokens.append(content)

        for token in special_tokens["additional_special_tokens"]:
            if token not in tokens:
                tokens.append(token)

        if special_tokens["eos_token"] not in tokens:
            tokens.append(special_tokens["eos_token"])

    if len(tokens) < target_vocab_size:
        # 说明：Qwen2 的 vocab_size 大于当前 tokenizer 的词表长度，补齐占位 token 保证索引对齐。
        existing = set(tokens)
        while len(tokens) < target_vocab_size:
            candidate = f"<|extra_token_{len(tokens)}|>"
            if candidate in existing:
                candidate = f"<|extra_token_{len(tokens)}_pad|>"
            tokens.append(candidate)
            existing.add(candidate)
        logger.warning(
            "词表长度小于目标 vocab_size，已补齐占位 token：tokens=%d target=%d",
            len(tokens),
            target_vocab_size,
        )
    elif len(tokens) > target_vocab_size:
        raise ValueError(f"Token list length {len(tokens)} exceeds target vocab size {target_vocab_size}")
    return tokens


class CosyVoice3Encoder(TTSEncoder):
    def __init__(
        self,
        model_path: Path | str,
        model_dir: Path | str,
        voice_pack: Optional[Path | str] = None,
        voice_name: str = "default",
        vocab_dir: Optional[Path | str] = None,
        tokenizer_py: Optional[Path | str] = None,
    ) -> None:
        super().__init__(model_path=model_path, architecture=COSYVOICE3_ARCHITECTURE)
        self.model_dir = Path(model_dir)
        self.voice_pack = Path(voice_pack) if voice_pack else None
        self.voice_name = voice_name
        self.vocab_dir = Path(vocab_dir) if vocab_dir else self.model_dir / "CosyVoice-BlankEN"
        self.tokenizer_py = Path(tokenizer_py) if tokenizer_py else Path("参考项目/CosyVoice/CosyVoice/tokenizer/tokenizer.py")
        self._llm_state: Optional[Dict[str, torch.Tensor]] = None
        self._flow_state: Optional[Dict[str, torch.Tensor]] = None
        self._hift_state: Optional[Dict[str, torch.Tensor]] = None
        self._qwen2_config: Optional[Dict[str, object]] = None

    @property
    def llm_state(self) -> Dict[str, torch.Tensor]:
        if self._llm_state is None:
            self._llm_state = _safe_torch_load(self.model_dir / "llm.pt")
        return self._llm_state

    @property
    def flow_state(self) -> Dict[str, torch.Tensor]:
        if self._flow_state is None:
            self._flow_state = _safe_torch_load(self.model_dir / "flow.pt")
        return self._flow_state

    @property
    def hift_state(self) -> Dict[str, torch.Tensor]:
        if self._hift_state is None:
            self._hift_state = _safe_torch_load(self.model_dir / "hift.pt")
        return self._hift_state

    @property
    def qwen2_config(self) -> Dict[str, object]:
        if self._qwen2_config is None:
            self._qwen2_config = _load_qwen2_config(self.vocab_dir)
        return self._qwen2_config

    def prepare_tensors(self) -> None:
        self.prepare_llm_tensors()
        self.prepare_flow_tensors()
        self.prepare_hift_tensors()
        self.prepare_const_tensors()
        self.prepare_voice_pack()

    def prepare_llm_tensors(self) -> None:
        missing: List[str] = []
        for name, tensor in self.llm_state.items():
            if name.startswith("llm.model.model."):
                out_name = f"{COSYVOICE3_ARCHITECTURE}.llm.{name[len('llm.model.model.'):]}"
            elif name == "llm_decoder.weight":
                out_name = f"{COSYVOICE3_ARCHITECTURE}.llm.llm_decoder.weight"
            elif name == "speech_embedding.weight":
                out_name = f"{COSYVOICE3_ARCHITECTURE}.llm.speech_embedding.weight"
            else:
                missing.append(name)
                continue
            self.set_tensor(out_name, tensor)

        if missing:
            self.logger.warning("LLM 未处理权重数量=%d，示例=%s", len(missing), missing[:3])

    def prepare_flow_tensors(self) -> None:
        for name, tensor in self.flow_state.items():
            out_name = f"{COSYVOICE3_ARCHITECTURE}.flow.{name}"
            self.set_tensor(out_name, tensor)

    def prepare_hift_tensors(self) -> None:
        weight_norm_map: Dict[str, Dict[str, torch.Tensor]] = {}
        for name, tensor in self.hift_state.items():
            if ".parametrizations.weight.original0" in name:
                base = name.replace(".parametrizations.weight.original0", "")
                weight_norm_map.setdefault(base, {})["g"] = tensor
                continue
            if ".parametrizations.weight.original1" in name:
                base = name.replace(".parametrizations.weight.original1", "")
                weight_norm_map.setdefault(base, {})["v"] = tensor
                continue
            out_name = f"{COSYVOICE3_ARCHITECTURE}.hift.{name}"
            self.set_tensor(out_name, tensor)

        for base, parts in weight_norm_map.items():
            if "g" not in parts or "v" not in parts:
                raise ValueError(f"WeightNorm 参数不完整: {base}")
            # 说明：torch._weight_norm 的默认 dim=0，与 PyTorch weight_norm 在 Conv1d 上一致。
            weight = torch._weight_norm(parts["v"], parts["g"], 0)
            out_name = f"{COSYVOICE3_ARCHITECTURE}.hift.{base}.weight"
            self.set_tensor(out_name, weight)

    def prepare_const_tensors(self) -> None:
        # 固定噪声（CFM 推理用）
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            rand_noise = torch.randn((1, 80, 50 * 300), dtype=torch.float32)
        self.set_tensor(f"{COSYVOICE3_ARCHITECTURE}.flow.rand_noise", rand_noise)

        # STFT Hann 窗（HiFT ISTFT 用）
        n_fft = 16
        # 说明：Hann 窗需使用 fftbins=True 的“周期窗”形式（分母为 n_fft），避免与 PyTorch/Scipy 默认不一致。
        idx = np.arange(n_fft, dtype=np.float32)
        window = (0.5 - 0.5 * np.cos(2.0 * np.pi * idx / n_fft)).astype(np.float32)
        self.set_tensor(f"{COSYVOICE3_ARCHITECTURE}.hift.stft_window", window)

    def _load_voice_pack_npz(self, path: Path) -> Dict[str, np.ndarray]:
        if not path.exists():
            raise FileNotFoundError(f"Voice pack not found: {path}")
        data = dict(np.load(path, allow_pickle=False))
        return data

    def _normalize_voice_pack(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        def _take(name: str) -> np.ndarray:
            if name not in data:
                raise KeyError(f"Voice pack missing key: {name}")
            return data[name]

        prompt_text_token = _take("prompt_text_token").astype(np.int32).squeeze()
        prompt_speech_token = _take("prompt_speech_token").astype(np.int32).squeeze()
        prompt_speech_feat = _take("prompt_speech_feat").astype(np.float32)
        embedding = _take("embedding").astype(np.float32).squeeze()

        if prompt_text_token.ndim != 1:
            prompt_text_token = prompt_text_token.reshape(-1)
        if prompt_speech_token.ndim != 1:
            prompt_speech_token = prompt_speech_token.reshape(-1)
        if prompt_speech_feat.ndim == 3:
            prompt_speech_feat = prompt_speech_feat.squeeze(0)
        if prompt_speech_feat.ndim != 2:
            raise ValueError("prompt_speech_feat must be 2D (T, 80) or 3D (1, T, 80)")
        if embedding.ndim != 1:
            embedding = embedding.reshape(-1)

        prompt_text_token_len = np.array([prompt_text_token.shape[0]], dtype=np.int32)
        prompt_speech_token_len = np.array([prompt_speech_token.shape[0]], dtype=np.int32)
        prompt_speech_feat_len = np.array([prompt_speech_feat.shape[0]], dtype=np.int32)

        return {
            "prompt_text_token": prompt_text_token,
            "prompt_text_token_len": prompt_text_token_len,
            "prompt_speech_token": prompt_speech_token,
            "prompt_speech_token_len": prompt_speech_token_len,
            "prompt_speech_feat": prompt_speech_feat,
            "prompt_speech_feat_len": prompt_speech_feat_len,
            "embedding": embedding,
        }

    def prepare_voice_pack(self) -> None:
        if self.voice_pack is None:
            return
        if self.voice_pack.is_dir():
            voice_paths = sorted(self.voice_pack.glob("*.npz"))
            if not voice_paths:
                raise FileNotFoundError(f"No .npz voice packs in {self.voice_pack}")
            voice_names = [p.stem for p in voice_paths]
        else:
            voice_paths = [self.voice_pack]
            voice_names = [self.voice_name]

        self.gguf_writer.add_array(f"{COSYVOICE3_ARCHITECTURE}.voices", voice_names)

        for voice_path, voice_name in zip(voice_paths, voice_names):
            pack = self._normalize_voice_pack(self._load_voice_pack_npz(voice_path))
            base = f"{COSYVOICE3_ARCHITECTURE}.voice_tensors.{voice_name}"
            # 说明：token 相关张量使用 I32，mel/embedding 使用 F32。
            self.set_tensor(f"{base}.prompt_text_token", pack["prompt_text_token"],
                            dtype=np.int32, gguf_dtype=gguf.GGMLQuantizationType.I32)
            self.set_tensor(f"{base}.prompt_text_token_len", pack["prompt_text_token_len"],
                            dtype=np.int32, gguf_dtype=gguf.GGMLQuantizationType.I32)
            self.set_tensor(f"{base}.prompt_speech_token", pack["prompt_speech_token"],
                            dtype=np.int32, gguf_dtype=gguf.GGMLQuantizationType.I32)
            self.set_tensor(f"{base}.prompt_speech_token_len", pack["prompt_speech_token_len"],
                            dtype=np.int32, gguf_dtype=gguf.GGMLQuantizationType.I32)
            self.set_tensor(f"{base}.prompt_speech_feat", pack["prompt_speech_feat"])
            self.set_tensor(f"{base}.prompt_speech_feat_len", pack["prompt_speech_feat_len"],
                            dtype=np.int32, gguf_dtype=gguf.GGMLQuantizationType.I32)
            self.set_tensor(f"{base}.embedding", pack["embedding"])

    def prepare_metadata(self) -> None:
        total_params, shared_params, expert_params, expert_count = self.gguf_writer.get_total_parameter_count()
        self.metadata = gguf.Metadata.load(None, None, COSYVOICE3_ARCHITECTURE, total_params)

        if self.metadata.size_label is None and total_params > 0:
            self.metadata.size_label = gguf.size_label(total_params, shared_params, expert_params, expert_count)

        self.set_type()
        self.set_gguf_parameters()
        self.metadata.set_gguf_meta_model(self.gguf_writer)
        self.set_vocab()
        self.gguf_writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

    def set_type(self) -> None:
        self.gguf_writer.add_type(gguf.GGUFType.MODEL)

    def set_gguf_parameters(self) -> None:
        # ---- Global ----
        self.gguf_writer.add_uint32(f"{COSYVOICE3_ARCHITECTURE}.sample_rate", 24000)
        self.gguf_writer.add_uint32(f"{COSYVOICE3_ARCHITECTURE}.token_frame_rate", 25)
        self.gguf_writer.add_uint32(f"{COSYVOICE3_ARCHITECTURE}.token_mel_ratio", 2)
        self.gguf_writer.add_uint32(f"{COSYVOICE3_ARCHITECTURE}.pre_lookahead_len", 3)
        self.gguf_writer.add_uint32(f"{COSYVOICE3_ARCHITECTURE}.speech_token_size", 6561)
        self.gguf_writer.add_uint32(f"{COSYVOICE3_ARCHITECTURE}.speech_stop_token_count", 200)

        # ---- Qwen2 LLM ----
        cfg = self.qwen2_config
        self.gguf_writer.add_uint32(f"{COSYVOICE3_ARCHITECTURE}.llm.vocab_size", int(cfg["vocab_size"]))
        self.gguf_writer.add_uint32(f"{COSYVOICE3_ARCHITECTURE}.llm.hidden_size", int(cfg["hidden_size"]))
        self.gguf_writer.add_uint32(f"{COSYVOICE3_ARCHITECTURE}.llm.layers", int(cfg["num_hidden_layers"]))
        self.gguf_writer.add_uint32(f"{COSYVOICE3_ARCHITECTURE}.llm.attn_heads", int(cfg["num_attention_heads"]))
        self.gguf_writer.add_uint32(f"{COSYVOICE3_ARCHITECTURE}.llm.kv_heads", int(cfg["num_key_value_heads"]))
        head_dim = int(cfg["hidden_size"]) // int(cfg["num_attention_heads"])
        self.gguf_writer.add_uint32(f"{COSYVOICE3_ARCHITECTURE}.llm.head_dim", head_dim)
        self.gguf_writer.add_float32(f"{COSYVOICE3_ARCHITECTURE}.llm.rope_theta", float(cfg["rope_theta"]))
        self.gguf_writer.add_float32(f"{COSYVOICE3_ARCHITECTURE}.llm.rms_norm_eps", float(cfg["rms_norm_eps"]))
        self.gguf_writer.add_uint32(f"{COSYVOICE3_ARCHITECTURE}.llm.max_position_embeddings", int(cfg["max_position_embeddings"]))

        # ---- Flow (DiT + CFM) ----
        self.gguf_writer.add_uint32(f"{COSYVOICE3_ARCHITECTURE}.flow.dim", 1024)
        self.gguf_writer.add_uint32(f"{COSYVOICE3_ARCHITECTURE}.flow.depth", 22)
        self.gguf_writer.add_uint32(f"{COSYVOICE3_ARCHITECTURE}.flow.heads", 16)
        self.gguf_writer.add_uint32(f"{COSYVOICE3_ARCHITECTURE}.flow.dim_head", 64)
        self.gguf_writer.add_uint32(f"{COSYVOICE3_ARCHITECTURE}.flow.ff_mult", 2)
        self.gguf_writer.add_uint32(f"{COSYVOICE3_ARCHITECTURE}.flow.mel_dim", 80)
        self.gguf_writer.add_uint32(f"{COSYVOICE3_ARCHITECTURE}.flow.spk_dim", 80)
        self.gguf_writer.add_uint32(f"{COSYVOICE3_ARCHITECTURE}.flow.n_timesteps", 10)
        self.gguf_writer.add_float32(f"{COSYVOICE3_ARCHITECTURE}.flow.sigma_min", 1e-6)
        self.gguf_writer.add_float32(f"{COSYVOICE3_ARCHITECTURE}.flow.inference_cfg_rate", 0.7)
        self.gguf_writer.add_str(f"{COSYVOICE3_ARCHITECTURE}.flow.t_scheduler", "cosine")
        self.gguf_writer.add_uint32(f"{COSYVOICE3_ARCHITECTURE}.flow.static_chunk_size", 50)
        self.gguf_writer.add_int32(f"{COSYVOICE3_ARCHITECTURE}.flow.num_decoding_left_chunks", -1)

        # ---- HiFT ----
        self.gguf_writer.add_uint32(f"{COSYVOICE3_ARCHITECTURE}.hift.in_channels", 80)
        self.gguf_writer.add_uint32(f"{COSYVOICE3_ARCHITECTURE}.hift.base_channels", 512)
        self.gguf_writer.add_uint32(f"{COSYVOICE3_ARCHITECTURE}.hift.nb_harmonics", 8)
        self.gguf_writer.add_uint32(f"{COSYVOICE3_ARCHITECTURE}.hift.n_fft", 16)
        self.gguf_writer.add_uint32(f"{COSYVOICE3_ARCHITECTURE}.hift.hop_len", 4)
        self.gguf_writer.add_float32(f"{COSYVOICE3_ARCHITECTURE}.hift.audio_limit", 0.99)
        self.gguf_writer.add_uint32(f"{COSYVOICE3_ARCHITECTURE}.hift.conv_pre_look_right", 4)
        self.gguf_writer.add_float32(f"{COSYVOICE3_ARCHITECTURE}.hift.lrelu_slope", 0.1)
        self.gguf_writer.add_float32(f"{COSYVOICE3_ARCHITECTURE}.hift.nsf_alpha", 0.1)
        self.gguf_writer.add_float32(f"{COSYVOICE3_ARCHITECTURE}.hift.nsf_sigma", 0.003)
        self.gguf_writer.add_float32(f"{COSYVOICE3_ARCHITECTURE}.hift.nsf_voiced_threshold", 10)
        self.gguf_writer.add_array(f"{COSYVOICE3_ARCHITECTURE}.hift.upsample_rates", [8, 5, 3])
        self.gguf_writer.add_array(f"{COSYVOICE3_ARCHITECTURE}.hift.upsample_kernel_sizes", [16, 11, 7])
        self.gguf_writer.add_array(f"{COSYVOICE3_ARCHITECTURE}.hift.resblock_kernel_sizes", [3, 7, 11])
        self.gguf_writer.add_array(f"{COSYVOICE3_ARCHITECTURE}.hift.resblock_dilation_sizes.0", [1, 3, 5])
        self.gguf_writer.add_array(f"{COSYVOICE3_ARCHITECTURE}.hift.resblock_dilation_sizes.1", [1, 3, 5])
        self.gguf_writer.add_array(f"{COSYVOICE3_ARCHITECTURE}.hift.resblock_dilation_sizes.2", [1, 3, 5])
        self.gguf_writer.add_array(f"{COSYVOICE3_ARCHITECTURE}.hift.source_resblock_kernel_sizes", [7, 7, 11])
        self.gguf_writer.add_array(f"{COSYVOICE3_ARCHITECTURE}.hift.source_resblock_dilation_sizes.0", [1, 3, 5])
        self.gguf_writer.add_array(f"{COSYVOICE3_ARCHITECTURE}.hift.source_resblock_dilation_sizes.1", [1, 3, 5])
        self.gguf_writer.add_array(f"{COSYVOICE3_ARCHITECTURE}.hift.source_resblock_dilation_sizes.2", [1, 3, 5])

        # 文件类型标记（不影响实际推理精度，仅用于描述）
        self.gguf_writer.add_file_type(gguf.LlamaFileType.ALL_F32)

    def set_vocab(self) -> None:
        additional = _extract_cosyvoice3_special_tokens(self.tokenizer_py)
        llm_vocab_size = int(self.qwen2_config["vocab_size"])
        tokens = _build_token_list(self.vocab_dir, additional, llm_vocab_size, self.logger)
        merges = _load_bpe_merges(self.vocab_dir / "merges.txt")

        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_merges(merges)
        self.gguf_writer.add_tokenizer_model("bpe")
        self.gguf_writer.add_tokenizer_pre("qwen2")

        self.gguf_writer.add_bos_token_id(int(self.qwen2_config["bos_token_id"]))
        self.gguf_writer.add_eos_token_id(int(self.qwen2_config["eos_token_id"]))
        self.gguf_writer.add_pad_token_id(int(self.qwen2_config["bos_token_id"]))
