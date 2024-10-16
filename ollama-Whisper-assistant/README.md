# ollama-voice-mac
A completely offline voice assistant using Qwen 7b via Ollama and Whisper speech recognition models. This Spired on the [excellent work of maudoin](https://github.com/maudoin/ollama-voice) by adding Mac compatibility with various improvements.

https://github.com/apeatling/ollama-voice-mac/assets/1464705/996abeb7-7e99-451b-8d3b-feb3fecbb82e

## Installing and running

1. Install [Ollama](https://ollama.ai) on your Mac.
2. Download the Qwen 7b model using the `ollama pull mistral` command.
3. Download an [OpenAI Whisper Model](https://github.com/openai/whisper/discussions/63#discussioncomment-3798552) (base.en works fine).
4. Clone this repo somewhere.
5. Place the Whisper model in a /whisper directory in the repo root folder.
6. Make sure you have [Python](https://www.python.org/downloads/macos/) and [Pip](https://pip.pypa.io/en/stable/installation/) installed.
7. For Apple silicon support of the PyAudio library you'll need to install [Homebrew](https://brew.sh) and run `brew install portaudio`.
8. Run `pip install -r requirements.txt` to install.
9. Run `python assistant_zh.py` to start the assistant.

## Other languages
You can set up support for other languages by editing `assistant.yaml`. Be sure to download a different Whisper model in your language and change the default `modelPath`.
