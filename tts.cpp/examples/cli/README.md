### Overview

This simple example cli tool can be used to generate speach from a text prompt and save it localy to a _.wav_ audio file.

### Requirements

* CLI and library must be built 
* A local GGUF file for parler tts mini

### Usage

For a concise user-facing help message, run:
```bash
./tts-cli --help
```

For the full list of advanced/tuning parameters, run:

```bash
./tts-cli --help-all
```

General usage should follow from these possible parameters. E.G. The following command will save generated speech to the `/tmp/test.wav` file.

```bash
./tts-cli --model-path /model/path/to/gguf_file.gguf --prompt "I am saying some words" --save-path /tmp/test.wav
```

Or, to run a built-in mixed zh/en benchmark prompt (ignores `--prompt` / `-p`):

```bash
./tts-cli --model-path /model/path/to/gguf_file.gguf --bench --save-path /tmp/bench_zf_001.wav
```

#### Dia and Orpheus Generation Arguments

Currently the default cli arguments are not aligned with Dia's or Orpheus' default sampling settings. Specifically the temperature and topk settings should be changed to  `1.3` and `35` respectively when generating with Dia like so:

```bash
./tts-cli --model-path /model/path/to/Dia.gguf --prompt "[S1] Hi, I am Dia, this is how I talk." --save-path /tmp/test.wav --topk 35 --temperature 1.3
```

and the voice, temperature, and repetition penalty setting should be changed to a valid voice (e.g. `leah`), `0.7`, and `1.1` respectively when generating with Orpheus like so:

```bash
./tts-cli --model-path /model/path/to/Orpheus.gguf --prompt "Hi, I am Orpheus, this is how I talk." --save-path /tmp/test.wav --voice leah --temperature 0.7 --repetition-penalty 1.1
```


#### Conditional Generation

Conditional generation is a Parler TTS specific behavior.

By default the Parler TTS model is saved to the GGUF format with a pre-encoded conditional prompt (i.e. a prompt used to determine how to generate speech), but if the text encoder model, the T5-Encoder model, is avaiable in gguf format (see the [python convertion scripts](../../py-gguf/README.md) for more information on how to prepare the T5-Encoder model) then a new conditional prompt can be used for generation like so:

```bash
./tts-cli --model-path /model/path/to/gguf_file.gguf --prompt "I am saying some words" --save-path /tmp/test.wav --text-encoder-path /model/path/to/t5_encoder_file.gguf --consditional-prompt "deep voice"
```

#### Distinct Voice Support

Kokoro and Orpheus both support voices which can be set via the `--voice` (`-v`) argument. Orpheus supports the following voices:

```
"zoe", "zac","jess", "leo", "mia", "julia", "leah"
```

and Kokoro supports the voices listedin the section below.

#### MultiLanguage Configuration

Kokoro supports multiple langauges with distinct voices, and, by default, the standard voices are encoded in the Kokoro gguf file. Below is a list of the available voices:

```
'af_alloy', 'af_aoede', 'af_bella', 'af_heart', 'af_jessica', 'af_kore', 'af_nicole',
'af_nova', 'af_river', 'af_sarah', 'af_sky', 'am_adam', 'am_echo', 'am_eric', 'am_fenrir',
'am_liam', 'am_michael', 'am_onyx', 'am_puck', 'am_santa', 'bf_alice', 'bf_emma',
'bf_isabella', 'bf_lily', 'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis', 'ef_dora',
'em_alex', 'em_santa', 'ff_siwis', 'hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi', 'if_sara',
'im_nicola', 'jf_alpha', 'jf_gongitsune', 'jf_nezumi', 'jf_tebukuro', 'jm_kumo', 'pf_dora',
'pm_alex', 'pm_santa', 'zf_xiaobei', 'zf_xiaoni', 'zf_xiaoxiao', 'zf_xiaoyi'
```

Each voice has a language assigned and gender assigned to it where the first letter of the pack represents the language and the second the gender (e.g. `af_alloy` is an American English Female voice; `a` corresponds to American Enlgish and `f` to Female). Below is a list of all currently supported langauges mapped to their respective codes:

```
# 🇺🇸 'a' => American English, 🇬🇧 'b' => British English
# 🇪🇸 'e' => Spanish es
# 🇫🇷 'f' => French fr-fr
# 🇮🇳 'h' => Hindi hi
# 🇮🇹 'i' => Italian it
# 🇯🇵 'j' => Japanese: pip install misaki[ja]
# 🇧🇷 'p' => Brazilian Portuguese pt-br
# 🇨🇳 'z' => Mandarin Chinese: pip install misaki[zh]
```

This fork does not rely on external phonemization libraries. Phonemization is handled as follows:

- CJK text (and `z*` voices) uses the built-in Chinese frontend (`zh_frontend`).
- Other text uses the built-in TTS.cpp phonemizer embedded in the GGUF.

Non zh/en languages are currently best-effort.

