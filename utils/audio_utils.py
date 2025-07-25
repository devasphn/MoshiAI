import numpy as np
import torch
import librosa
from typing import Optional

def preprocess_audio(audio_data: np.ndarray, target_sr: int = 24000) -> torch.Tensor:
    """Preprocess audio data for Kyutai models with proper tensor shapes"""
    if len(audio_data) == 0:
        return torch.zeros(1, 1, target_sr)
    
    # Ensure mono
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Resample if needed
    if target_sr != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=16000, target_sr=target_sr)
    
    # Normalize
    audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-9)
    
    # Convert to tensor with proper shape [B, C, T]
    audio_tensor = torch.from_numpy(audio_data).float()
    
    # CRITICAL: Ensure proper shape for Kyutai models
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # [T] -> [1, 1, T]
    elif audio_tensor.dim() == 2:
        audio_tensor = audio_tensor.unsqueeze(1)  # [B, T] -> [B, 1, T]
    
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

def detect_speech_activity(audio_data: np.ndarray, threshold: float = 0.005) -> bool:
    """Enhanced voice activity detection with lower threshold"""
    if len(audio_data) == 0:
        return False
    
    # Energy-based detection with lower threshold
    energy = np.mean(audio_data ** 2)
    
    if energy < threshold:
        return False
    
    # Spectral-based detection
    try:
        # Check if audio has reasonable frequency content
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=16000)
        avg_centroid = np.mean(spectral_centroid)
        
        # Check for zero crossing rate (voice has specific patterns)
        zcr = librosa.feature.zero_crossing_rate(audio_data)
        avg_zcr = np.mean(zcr)
        
        # Combined detection with relaxed thresholds
        has_energy = energy > threshold
        has_spectral_content = avg_centroid > 200  # Lowered from 500
        has_reasonable_zcr = 0.01 < avg_zcr < 0.8
        
        return has_energy and has_spectral_content and has_reasonable_zcr
        
    except Exception as e:
        # Fallback to simple energy detection
        return energy > threshold

def fix_audio_shape_for_mimi(audio_tensor: torch.Tensor) -> torch.Tensor:
    """Fix audio tensor shape for Mimi compression model"""
    if audio_tensor.dim() == 1:
        # [T] -> [1, 1, T]
        return audio_tensor.unsqueeze(0).unsqueeze(0)
    elif audio_tensor.dim() == 2:
        # [B, T] -> [B, 1, T]
        return audio_tensor.unsqueeze(1)
    else:
        # Already correct shape [B, C, T]
        return audio_tensor
