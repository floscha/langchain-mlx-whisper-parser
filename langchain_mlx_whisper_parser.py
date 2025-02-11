import logging
from pathlib import Path
from typing import Iterator

from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
import mlx_whisper

logger = logging.getLogger(__name__)


class MlxWhisperParser(BaseBlobParser):
    "Transcribe and parse audio files with a local MLX Whisper model."

    def __init__(self, model_path_or_hf_repo: str = "mlx-community/whisper-turbo", cache_transcriptions: bool = False):
        """Initialize the parser.

        Args:
            model_path_or_hf_repo: MLX Whisper model to use. Defaults to 'mlx-community/whisper-turbo'.
            cache_transcriptions: If True, stores transcriptions as .txt files locally for easy re-use.
        """
        self.model_path_or_hf_repo = model_path_or_hf_repo
        self.cache_transcriptions = cache_transcriptions
        logger.info("Using the following model: ", self.model_path_or_hf_repo)


    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        logger.info(f"Transcribing part {blob.path}!")
        transcription_path = Path(blob.path).with_suffix(".txt")

        if self.cache_transcriptions and transcription_path.exists():
            transcription_text = transcription_path.read_text()
        else:
            transcription_result = mlx_whisper.transcribe(str(blob.path), path_or_hf_repo=self.model_path_or_hf_repo)
            transcription_text = transcription_result["text"]

        if self.cache_transcriptions and not transcription_path.exists():
            transcription_path.write_text(transcription_text)

        yield Document(page_content=transcription_text, metadata={"source": blob.source})
