"""
Audio Preprocessing Pipeline for EchoTwin

This module handles audio loading, preprocessing, and basic transformations
optimized for voice analysis and disorder detection.
"""

import librosa
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import butter, filtfilt
import warnings
from typing import Tuple, Optional, Union
import os

warnings.filterwarnings('ignore')


class AudioProcessor:
    """
    Comprehensive audio preprocessing pipeline for voice analysis.
    
    Features:
    - Audio loading and resampling
    - Noise reduction and filtering
    - Voice activity detection
    - Audio normalization
    - Segment extraction
    """
    
    def __init__(self, 
                 target_sr: int = 16000,
                 frame_length: int = 2048,
                 hop_length: int = 512):
        """
        Initialize AudioProcessor with default parameters.
        
        Args:
            target_sr: Target sampling rate for audio processing
            frame_length: Frame length for STFT analysis
            hop_length: Hop length for STFT analysis
        """
        self.target_sr = target_sr
        self.frame_length = frame_length
        self.hop_length = hop_length
        
    def load_audio(self, 
                   audio_path: Union[str, np.ndarray], 
                   sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        Load audio file or process numpy array.
        
        Args:
            audio_path: Path to audio file or numpy array
            sr: Sampling rate (if audio_path is numpy array)
            
        Returns:
            Tuple of (audio_data, sampling_rate)
        """
        if isinstance(audio_path, str):
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Load audio file
            audio, original_sr = librosa.load(audio_path, sr=None)
            
            # Resample if necessary
            if original_sr != self.target_sr:
                audio = librosa.resample(audio, 
                                       orig_sr=original_sr, 
                                       target_sr=self.target_sr)
            
            return audio, self.target_sr
            
        elif isinstance(audio_path, np.ndarray):
            audio = audio_path
            if sr is None:
                sr = self.target_sr
            
            # Resample if necessary
            if sr != self.target_sr:
                audio = librosa.resample(audio, 
                                       orig_sr=sr, 
                                       target_sr=self.target_sr)
            
            return audio, self.target_sr
        else:
            raise ValueError("audio_path must be string path or numpy array")
    
    def normalize_audio(self, audio: np.ndarray, method: str = 'peak') -> np.ndarray:
        """
        Normalize audio signal.
        
        Args:
            audio: Input audio signal
            method: Normalization method ('peak', 'rms', 'lufs')
            
        Returns:
            Normalized audio signal
        """
        if method == 'peak':
            # Peak normalization
            peak = np.max(np.abs(audio))
            if peak > 0:
                return audio / peak
            return audio
            
        elif method == 'rms':
            # RMS normalization
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                return audio / rms * 0.1  # Scale to reasonable level
            return audio
            
        elif method == 'lufs':
            # Simple LUFS-inspired normalization
            # This is a simplified version - full LUFS requires more complex filtering
            squared = audio**2
            mean_square = np.mean(squared)
            if mean_square > 0:
                target_lufs = -23  # Target LUFS level
                current_lufs = -0.691 + 10 * np.log10(mean_square)
                gain_db = target_lufs - current_lufs
                gain_linear = 10**(gain_db / 20)
                return audio * gain_linear
            return audio
        
        else:
            raise ValueError("method must be 'peak', 'rms', or 'lufs'")
    
    def apply_bandpass_filter(self, 
                             audio: np.ndarray, 
                             low_freq: float = 80, 
                             high_freq: float = 8000) -> np.ndarray:
        """
        Apply bandpass filter to focus on speech frequencies.
        
        Args:
            audio: Input audio signal
            low_freq: Low cutoff frequency
            high_freq: High cutoff frequency
            
        Returns:
            Filtered audio signal
        """
        nyquist = self.target_sr / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Design butterworth bandpass filter
        b, a = butter(4, [low, high], btype='band')
        
        # Apply filter
        filtered_audio = filtfilt(b, a, audio)
        
        return filtered_audio
    
    def remove_silence(self, 
                      audio: np.ndarray, 
                      top_db: int = 20,
                      frame_length: int = 2048,
                      hop_length: int = 512) -> np.ndarray:
        """
        Remove silence from audio using energy-based detection.
        
        Args:
            audio: Input audio signal
            top_db: Threshold below reference for silence
            frame_length: Frame length for analysis
            hop_length: Hop length for analysis
            
        Returns:
            Audio with silence removed
        """
        # Use librosa's built-in silence removal
        audio_trimmed, _ = librosa.effects.trim(
            audio, 
            top_db=top_db,
            frame_length=frame_length,
            hop_length=hop_length
        )
        
        return audio_trimmed
    
    def detect_voice_activity(self, 
                             audio: np.ndarray,
                             frame_length: int = 2048,
                             hop_length: int = 512,
                             energy_threshold: float = 0.01) -> np.ndarray:
        """
        Simple voice activity detection based on energy.
        
        Args:
            audio: Input audio signal
            frame_length: Frame length for analysis
            hop_length: Hop length for analysis
            energy_threshold: Energy threshold for voice detection
            
        Returns:
            Boolean array indicating voice activity
        """
        # Calculate frame-wise energy
        frames = librosa.util.frame(audio, 
                                  frame_length=frame_length, 
                                  hop_length=hop_length)
        energy = np.sum(frames**2, axis=0)
        
        # Normalize energy
        energy = energy / np.max(energy) if np.max(energy) > 0 else energy
        
        # Apply threshold
        voice_activity = energy > energy_threshold
        
        return voice_activity
    
    def extract_voiced_segments(self, 
                               audio: np.ndarray,
                               min_duration: float = 0.1,
                               max_duration: float = 10.0) -> list:
        """
        Extract voiced segments from audio.
        
        Args:
            audio: Input audio signal
            min_duration: Minimum segment duration in seconds
            max_duration: Maximum segment duration in seconds
            
        Returns:
            List of voiced audio segments
        """
        # Detect voice activity
        vad = self.detect_voice_activity(audio)
        
        # Convert to sample indices
        hop_length = self.hop_length
        min_samples = int(min_duration * self.target_sr)
        max_samples = int(max_duration * self.target_sr)
        
        segments = []
        start_idx = None
        
        for i, is_voice in enumerate(vad):
            sample_idx = i * hop_length
            
            if is_voice and start_idx is None:
                start_idx = sample_idx
            elif not is_voice and start_idx is not None:
                end_idx = sample_idx
                segment_length = end_idx - start_idx
                
                if min_samples <= segment_length <= max_samples:
                    segment = audio[start_idx:end_idx]
                    segments.append(segment)
                
                start_idx = None
        
        # Handle case where audio ends with voice
        if start_idx is not None:
            segment = audio[start_idx:]
            if len(segment) >= min_samples:
                segments.append(segment)
        
        return segments
    
    def add_noise(self, 
                  audio: np.ndarray, 
                  noise_level: float = 0.005,
                  noise_type: str = 'gaussian') -> np.ndarray:
        """
        Add noise to audio for data augmentation.
        
        Args:
            audio: Input audio signal
            noise_level: Noise level (standard deviation for gaussian)
            noise_type: Type of noise ('gaussian', 'uniform')
            
        Returns:
            Audio with added noise
        """
        if noise_type == 'gaussian':
            noise = np.random.normal(0, noise_level, len(audio))
        elif noise_type == 'uniform':
            noise = np.random.uniform(-noise_level, noise_level, len(audio))
        else:
            raise ValueError("noise_type must be 'gaussian' or 'uniform'")
        
        return audio + noise
    
    def time_stretch(self, audio: np.ndarray, rate: float) -> np.ndarray:
        """
        Time stretch audio without changing pitch.
        
        Args:
            audio: Input audio signal
            rate: Stretch rate (>1 = faster, <1 = slower)
            
        Returns:
            Time-stretched audio
        """
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def pitch_shift(self, audio: np.ndarray, n_steps: float) -> np.ndarray:
        """
        Shift pitch without changing duration.
        
        Args:
            audio: Input audio signal
            n_steps: Number of semitones to shift
            
        Returns:
            Pitch-shifted audio
        """
        return librosa.effects.pitch_shift(audio, 
                                         sr=self.target_sr, 
                                         n_steps=n_steps)
    
    def save_audio(self, 
                   audio: np.ndarray, 
                   output_path: str, 
                   sr: Optional[int] = None) -> None:
        """
        Save audio to file.
        
        Args:
            audio: Audio signal to save
            output_path: Output file path
            sr: Sampling rate (uses self.target_sr if None)
        """
        if sr is None:
            sr = self.target_sr
        
        sf.write(output_path, audio, sr)
    
    def get_audio_info(self, audio: np.ndarray) -> dict:
        """
        Get basic information about audio signal.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Dictionary with audio information
        """
        return {
            'duration': len(audio) / self.target_sr,
            'samples': len(audio),
            'sampling_rate': self.target_sr,
            'channels': 1,  # Assuming mono
            'peak_amplitude': np.max(np.abs(audio)),
            'rms_amplitude': np.sqrt(np.mean(audio**2)),
            'dynamic_range': np.max(audio) - np.min(audio)
        }


# Example usage and testing
if __name__ == "__main__":
    # Create processor instance
    processor = AudioProcessor()
    
    # Example with synthetic audio
    duration = 3.0  # seconds
    t = np.linspace(0, duration, int(duration * processor.target_sr))
    # Create a simple sine wave as test audio
    test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    # Test preprocessing pipeline
    print("Testing AudioProcessor...")
    
    # Normalize
    normalized = processor.normalize_audio(test_audio, method='peak')
    print(f"Normalized peak: {np.max(np.abs(normalized)):.3f}")
    
    # Filter
    filtered = processor.apply_bandpass_filter(normalized)
    print(f"Filtered audio shape: {filtered.shape}")
    
    # Get info
    info = processor.get_audio_info(filtered)
    print(f"Audio info: {info}")
    
    print("AudioProcessor test completed successfully!")