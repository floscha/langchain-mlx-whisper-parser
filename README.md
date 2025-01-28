# LangChain MLX Whisper Parser

LangChain parser to transcribe audio files with a local [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) model.

## ⚙️ Installation

Install with pip:
```
pip install langchain-mlx-whisper-parser@git+https://github.com/floscha/langchain-mlx-whisper-parser
```

## 🚀 Usage

Import and instantiate like below and then use it like a regular LangChain parser:

```python
from langchain_mlx_whisper_parser import MlxWhisperParser

parser = MlxWhisperParser()
```
