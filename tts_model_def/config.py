# tts_model_def/config.py

# This file defines the configuration class for the DSM-TTS model.
# It's necessary to instantiate the model architecture before loading the weights.

from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class DSMTTSConfig:
    # Model dimensions
    dim: int = 1024
    # Transformer layers
    n_heads: int = 16
    n_layers: int = 16
    # Conformer settings
    conformer_n_layers: int = 4
    # Other architectural details
    codebook_size: int = 8192
    n_codebooks: int = 32
    max_text_len: int = 512
    max_speech_len: int = 4096
    # These values are based on the standard configuration for the kyutai/tts-1.6b-en_fr model
