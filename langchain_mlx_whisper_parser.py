import logging
from typing import Iterator

from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
import mlx_whisper

logger = logging.getLogger(__name__)


class MlxWhisperParser(BaseBlobParser):
    "Transcribe and parse audio files with a local MLX Whisper model."

    def __init__(self, model_path_or_hf_repo: str = "mlx-community/whisper-turbo"):
        """Initialize the parser.

        Args:
            model_path_or_hf_repo: MLX Whisper model to use. Defaults to 'mlx-community/whisper-turbo'.
        """
        self.model_path_or_hf_repo = model_path_or_hf_repo
        print("Using the following model: ", self.model_path_or_hf_repo)


    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        print(f"Transcribing part {blob.path}!")
        transcription_result = mlx_whisper.transcribe(str(blob.path), path_or_hf_repo=self.model_path_or_hf_repo)
        yield Document(page_content=transcription_result["text"], metadata={"source": blob.source})
