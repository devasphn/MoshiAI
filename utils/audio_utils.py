import numpy as np
import torch
import librosa
from typing import Optional

def preprocess_audio(audio_data: np.ndarray, target_sr: int = 16000) -> torch.Tensor:
    """Preprocess audio data for model input"""
    if len(audio_data) == 0:
        return torch.zeros(1, 1, target_sr)
    
    # Ensure mono
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Resample if needed
    if target_sr != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=16000, target_sr=target_sr)
    
    # Normalize
    audio_data = audio_data / np.max(np.abs(audio_data) + 1e-9)
    
    # Convert to tensor
    audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0).unsqueeze(0)
    
    return audio_tensor

def postprocess_audio(audio_tensor: torch.Tensor, target_sr: int = 24000) -> np.ndarray:
    """Postprocess audio tensor to numpy array"""
    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor.squeeze()
    
    audio_np = audio_tensor.cpu().numpy()
    
    # Ensure reasonable amplitude
    max_val = np.max(np.abs(audio_np))
    if max_val > 0:
        audio_np = audio_np / max_val * 0.7
    
    return audio_np.astype(np.float32)

def detect_speech_activity(audio_data: np.ndarray, threshold: float = 0.01) -> bool:
    """Simple voice activity detection"""
    if len(audio_data) == 0:
        return False
    
    energy = np.mean(audio_data ** 2)
    return energy > threshold
