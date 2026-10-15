# contracts/py/tinyllm/io/hf_whisper.pyi (L14.3): Hugging Face Whisper checkpoints
# chapter: content/courses/build-an-llm/p14-speech/03-whisper-decoder-and-weights.md
#
# An HF Whisper directory holds config.json (WhisperConfig.from_hf, L14.2)
# and model.safetensors (L0.6's reader; every dtype becomes float32). Its
# tensor names are the course's with a "model." prefix, plus two tensors
# the course model does not hold as parameters:
#
#   proj_out.weight                  the output projection, tied to
#                                    model.decoder.embed_tokens.weight (often
#                                    not stored at all)
#   model.encoder.embed_positions.weight
#                                    the sinusoid table, frozen
from typing import Mapping, Optional

from numpy.typing import NDArray

from tinyllm.speech.encoder import WhisperConfig

def whisper_state_from_hf(tensors: Mapping[str, NDArray], cfg: WhisperConfig) -> tuple[dict[str, NDArray], Optional[NDArray]]:
    """(course state dict, stored sinusoid table or None). Names lose
    "model."; proj_out.weight is dropped after checking it equals
    decoder.embed_tokens.weight; the encoder table is checked against
    whisper_sinusoids(n_audio_ctx, d) within 1e-4 (5e-4 for a table
    exactly representable in float16) and returned separately.
    ValueError for a name without the prefix (other than proj_out.weight),
    an untied proj_out, or a table that is not Whisper's sinusoids."""

def read_whisper(dir: str) -> tuple[WhisperConfig, dict[str, NDArray], Optional[NDArray]]:
    """WhisperConfig.from_hf(dir) and whisper_state_from_hf of every tensor
    in dir/model.safetensors (or the shards a model.safetensors.index.json
    lists). FileNotFoundError when there is no weight file."""
