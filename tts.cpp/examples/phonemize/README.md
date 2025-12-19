### Overview

This is a simple cli for running the built-in TTS.cpp phonemization (GGUF encoded rules + dictionary) on a plain text string.

### Requirements

* phonemize and library must be built 
* A local GGUF file containg TTS.cpp phonemization rules

### Usage

In order to get a detailed breakdown the functionality currently available you can call the cli with the `--help` parameter. This will return a breakdown of all parameters:
```commandline
./build/bin/phonemize --help

--phonemizer-path (-mp):
    (REQUIRED) The local path of the gguf phonemiser file for TTS.cpp phonemizer.
--prompt (-p):
    (REQUIRED) The text prompt to phonemize.
```

General usage should follow from these possible parameters. E.G. The following command will return the phonemized IPA text for the prompt via the TTS.cpp phonemizer.

```commandline
./build/bin/phonemize --phonemizer-path "/path/to/tts_phonemizer.gguf" --prompt "this is a test."
```
