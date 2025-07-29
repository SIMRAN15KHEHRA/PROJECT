"""
Feature Extraction Module for EchoTwin

This module extracts comprehensive voice features for disorder detection:
- MFCC (Mel-Frequency Cepstral Coefficients)
- Pitch-related features (F0, jitter, shimmer)
- Spectral features
- Prosodic features
- Voice quality measures
"""

import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy import stats
from scipy.signal import find_peaks
import warnings
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

warnings.filterwarnings('ignore')


class FeatureExtractor:
    """
    Comprehensive feature extraction for voice analysis.
    
    Extracts multiple types of features:
    - Spectral features (MFCC, spectral centroid, rolloff, etc.)
    - Pitch features (F0, jitter, shimmer)
    - Voice quality features (HNR, NHR, etc.)
    - Prosodic features (rhythm, timing)
    """
    
    def __init__(self, sr: int = 16000):
        """
        Initialize FeatureExtractor.
        
        Args:
            sr: Sampling rate for audio processing
        """
        self.sr = sr
        
    def extract_mfcc_features(self, 
                             audio: np.ndarray, 
                             n_mfcc: int = 13,
                             n_fft: int = 2048,
                             hop_length: int = 512,
                             n_mels: int = 128) -> Dict[str, np.ndarray]:
        """
        Extract MFCC features and related statistics.
        
        Args:
            audio: Input audio signal
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel bands
            
        Returns:
            Dictionary containing MFCC features and statistics
        """
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
        # Calculate statistics for each MFCC coefficient
        mfcc_features = {
            'mfcc_raw': mfccs,
            'mfcc_mean': np.mean(mfccs, axis=1),
            'mfcc_std': np.std(mfccs, axis=1),
            'mfcc_skew': stats.skew(mfccs, axis=1),
            'mfcc_kurtosis': stats.kurtosis(mfccs, axis=1),
            'mfcc_median': np.median(mfccs, axis=1),
            'mfcc_min': np.min(mfccs, axis=1),
            'mfcc_max': np.max(mfccs, axis=1),
            'mfcc_range': np.max(mfccs, axis=1) - np.min(mfccs, axis=1)
        }
        
        # Delta and delta-delta features
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        
        mfcc_features.update({
            'mfcc_delta_mean': np.mean(mfcc_delta, axis=1),
            'mfcc_delta_std': np.std(mfcc_delta, axis=1),
            'mfcc_delta2_mean': np.mean(mfcc_delta2, axis=1),
            'mfcc_delta2_std': np.std(mfcc_delta2, axis=1)
        })
        
        return mfcc_features
    
    def extract_spectral_features(self, 
                                 audio: np.ndarray,
                                 n_fft: int = 2048,
                                 hop_length: int = 512) -> Dict[str, float]:
        """
        Extract spectral features from audio.
        
        Args:
            audio: Input audio signal
            n_fft: FFT window size
            hop_length: Hop length for STFT
            
        Returns:
            Dictionary containing spectral features
        """
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio, sr=self.sr, hop_length=hop_length)[0]
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sr, hop_length=hop_length)[0]
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sr, hop_length=hop_length)[0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            audio, frame_length=n_fft, hop_length=hop_length)[0]
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio, sr=self.sr, hop_length=hop_length)
        
        # Spectral flatness
        spectral_flatness = librosa.feature.spectral_flatness(
            y=audio, hop_length=hop_length)[0]
        
        # RMS energy
        rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        
        features = {
            'spectral_centroid_mean': np.mean(spectral_centroids),
            'spectral_centroid_std': np.std(spectral_centroids),
            'spectral_rolloff_mean': np.mean(spectral_rolloff),
            'spectral_rolloff_std': np.std(spectral_rolloff),
            'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
            'spectral_bandwidth_std': np.std(spectral_bandwidth),
            'zcr_mean': np.mean(zcr),
            'zcr_std': np.std(zcr),
            'spectral_contrast_mean': np.mean(spectral_contrast, axis=1),
            'spectral_contrast_std': np.std(spectral_contrast, axis=1),
            'spectral_flatness_mean': np.mean(spectral_flatness),
            'spectral_flatness_std': np.std(spectral_flatness),
            'rms_mean': np.mean(rms),
            'rms_std': np.std(rms)
        }
        
        return features
    
    def extract_pitch_features(self, 
                              audio: np.ndarray,
                              f0_min: float = 75,
                              f0_max: float = 600) -> Dict[str, float]:
        """
        Extract pitch-related features using Parselmouth/Praat.
        
        Args:
            audio: Input audio signal
            f0_min: Minimum F0 frequency
            f0_max: Maximum F0 frequency
            
        Returns:
            Dictionary containing pitch features
        """
        try:
            # Create Parselmouth Sound object
            sound = parselmouth.Sound(audio, sampling_frequency=self.sr)
            
            # Extract pitch
            pitch = call(sound, "To Pitch", 0.0, f0_min, f0_max)
            
            # Get F0 values
            f0_values = call(pitch, "List values in all frames", "Hertz")
            f0_values = [f for f in f0_values if f != 0]  # Remove unvoiced frames
            
            if len(f0_values) == 0:
                return self._get_empty_pitch_features()
            
            f0_array = np.array(f0_values)
            
            # Basic F0 statistics
            features = {
                'f0_mean': np.mean(f0_array),
                'f0_std': np.std(f0_array),
                'f0_min': np.min(f0_array),
                'f0_max': np.max(f0_array),
                'f0_range': np.max(f0_array) - np.min(f0_array),
                'f0_median': np.median(f0_array),
                'f0_iqr': np.percentile(f0_array, 75) - np.percentile(f0_array, 25),
                'f0_skew': stats.skew(f0_array),
                'f0_kurtosis': stats.kurtosis(f0_array)
            }
            
            # Calculate jitter and shimmer
            jitter_features = self._calculate_jitter(sound, pitch)
            shimmer_features = self._calculate_shimmer(sound, pitch)
            
            features.update(jitter_features)
            features.update(shimmer_features)
            
            # Voice quality measures
            voice_quality = self._calculate_voice_quality(sound, pitch)
            features.update(voice_quality)
            
            return features
            
        except Exception as e:
            print(f"Error extracting pitch features: {e}")
            return self._get_empty_pitch_features()
    
    def _calculate_jitter(self, sound, pitch) -> Dict[str, float]:
        """Calculate various jitter measures."""
        try:
            # Local jitter
            jitter_local = call(pitch, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            
            # Absolute jitter
            jitter_absolute = call(pitch, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
            
            # RAP jitter
            jitter_rap = call(pitch, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            
            # PPQ5 jitter
            jitter_ppq5 = call(pitch, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
            
            return {
                'jitter_local': jitter_local,
                'jitter_absolute': jitter_absolute,
                'jitter_rap': jitter_rap,
                'jitter_ppq5': jitter_ppq5
            }
        except:
            return {
                'jitter_local': 0.0,
                'jitter_absolute': 0.0,
                'jitter_rap': 0.0,
                'jitter_ppq5': 0.0
            }
    
    def _calculate_shimmer(self, sound, pitch) -> Dict[str, float]:
        """Calculate various shimmer measures."""
        try:
            # Local shimmer
            shimmer_local = call([sound, pitch], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            
            # Absolute shimmer
            shimmer_absolute = call([sound, pitch], "Get shimmer (local, dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            
            # APQ3 shimmer
            shimmer_apq3 = call([sound, pitch], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            
            # APQ5 shimmer
            shimmer_apq5 = call([sound, pitch], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            
            return {
                'shimmer_local': shimmer_local,
                'shimmer_absolute': shimmer_absolute,
                'shimmer_apq3': shimmer_apq3,
                'shimmer_apq5': shimmer_apq5
            }
        except:
            return {
                'shimmer_local': 0.0,
                'shimmer_absolute': 0.0,
                'shimmer_apq3': 0.0,
                'shimmer_apq5': 0.0
            }
    
    def _calculate_voice_quality(self, sound, pitch) -> Dict[str, float]:
        """Calculate voice quality measures."""
        try:
            # Harmonics-to-Noise Ratio
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
            
            # Noise-to-Harmonics Ratio
            nhr = 1 / (10**(hnr/10)) if hnr > 0 else float('inf')
            
            return {
                'hnr': hnr,
                'nhr': nhr
            }
        except:
            return {
                'hnr': 0.0,
                'nhr': 0.0
            }
    
    def _get_empty_pitch_features(self) -> Dict[str, float]:
        """Return empty pitch features when extraction fails."""
        return {
            'f0_mean': 0.0, 'f0_std': 0.0, 'f0_min': 0.0, 'f0_max': 0.0,
            'f0_range': 0.0, 'f0_median': 0.0, 'f0_iqr': 0.0,
            'f0_skew': 0.0, 'f0_kurtosis': 0.0,
            'jitter_local': 0.0, 'jitter_absolute': 0.0,
            'jitter_rap': 0.0, 'jitter_ppq5': 0.0,
            'shimmer_local': 0.0, 'shimmer_absolute': 0.0,
            'shimmer_apq3': 0.0, 'shimmer_apq5': 0.0,
            'hnr': 0.0, 'nhr': 0.0
        }
    
    def extract_prosodic_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract prosodic features related to rhythm and timing.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Dictionary containing prosodic features
        """
        # Voice activity detection for rhythm analysis
        frame_length = 2048
        hop_length = 512
        
        # Energy-based segmentation
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        energy = np.sum(frames**2, axis=0)
        
        # Normalize energy
        energy_norm = energy / np.max(energy) if np.max(energy) > 0 else energy
        
        # Find voiced segments
        threshold = 0.1
        voiced_frames = energy_norm > threshold
        
        # Calculate rhythm features
        features = {}
        
        if np.any(voiced_frames):
            # Voice activity ratio
            features['voice_activity_ratio'] = np.sum(voiced_frames) / len(voiced_frames)
            
            # Find voice segments
            voice_segments = []
            in_segment = False
            start_idx = 0
            
            for i, is_voiced in enumerate(voiced_frames):
                if is_voiced and not in_segment:
                    start_idx = i
                    in_segment = True
                elif not is_voiced and in_segment:
                    voice_segments.append(i - start_idx)
                    in_segment = False
            
            if in_segment:  # Handle case where audio ends with voice
                voice_segments.append(len(voiced_frames) - start_idx)
            
            if voice_segments:
                segment_durations = np.array(voice_segments) * hop_length / self.sr
                features.update({
                    'mean_segment_duration': np.mean(segment_durations),
                    'std_segment_duration': np.std(segment_durations),
                    'num_segments': len(voice_segments)
                })
            else:
                features.update({
                    'mean_segment_duration': 0.0,
                    'std_segment_duration': 0.0,
                    'num_segments': 0
                })
        else:
            features.update({
                'voice_activity_ratio': 0.0,
                'mean_segment_duration': 0.0,
                'std_segment_duration': 0.0,
                'num_segments': 0
            })
        
        # Energy dynamics
        if len(energy_norm) > 1:
            energy_diff = np.diff(energy_norm)
            features.update({
                'energy_dynamics_mean': np.mean(np.abs(energy_diff)),
                'energy_dynamics_std': np.std(energy_diff)
            })
        else:
            features.update({
                'energy_dynamics_mean': 0.0,
                'energy_dynamics_std': 0.0
            })
        
        return features
    
    def extract_formant_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract formant frequencies using Parselmouth.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Dictionary containing formant features
        """
        try:
            # Create Parselmouth Sound object
            sound = parselmouth.Sound(audio, sampling_frequency=self.sr)
            
            # Extract formants
            formants = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
            
            # Get formant values for first 3 formants
            f1_values = []
            f2_values = []
            f3_values = []
            
            # Sample formants at regular intervals
            duration = call(sound, "Get total duration")
            time_step = 0.01  # 10ms steps
            
            for t in np.arange(0, duration, time_step):
                try:
                    f1 = call(formants, "Get value at time", 1, t, "Hertz", "Linear")
                    f2 = call(formants, "Get value at time", 2, t, "Hertz", "Linear")
                    f3 = call(formants, "Get value at time", 3, t, "Hertz", "Linear")
                    
                    if not np.isnan(f1) and f1 > 0:
                        f1_values.append(f1)
                    if not np.isnan(f2) and f2 > 0:
                        f2_values.append(f2)
                    if not np.isnan(f3) and f3 > 0:
                        f3_values.append(f3)
                except:
                    continue
            
            features = {}
            
            # F1 statistics
            if f1_values:
                f1_array = np.array(f1_values)
                features.update({
                    'f1_mean': np.mean(f1_array),
                    'f1_std': np.std(f1_array),
                    'f1_median': np.median(f1_array)
                })
            else:
                features.update({'f1_mean': 0.0, 'f1_std': 0.0, 'f1_median': 0.0})
            
            # F2 statistics
            if f2_values:
                f2_array = np.array(f2_values)
                features.update({
                    'f2_mean': np.mean(f2_array),
                    'f2_std': np.std(f2_array),
                    'f2_median': np.median(f2_array)
                })
            else:
                features.update({'f2_mean': 0.0, 'f2_std': 0.0, 'f2_median': 0.0})
            
            # F3 statistics
            if f3_values:
                f3_array = np.array(f3_values)
                features.update({
                    'f3_mean': np.mean(f3_array),
                    'f3_std': np.std(f3_array),
                    'f3_median': np.median(f3_array)
                })
            else:
                features.update({'f3_mean': 0.0, 'f3_std': 0.0, 'f3_median': 0.0})
            
            # Formant ratios and relationships
            if f1_values and f2_values:
                f1_mean = features['f1_mean']
                f2_mean = features['f2_mean']
                if f1_mean > 0:
                    features['f2_f1_ratio'] = f2_mean / f1_mean
                else:
                    features['f2_f1_ratio'] = 0.0
            else:
                features['f2_f1_ratio'] = 0.0
            
            return features
            
        except Exception as e:
            print(f"Error extracting formant features: {e}")
            return {
                'f1_mean': 0.0, 'f1_std': 0.0, 'f1_median': 0.0,
                'f2_mean': 0.0, 'f2_std': 0.0, 'f2_median': 0.0,
                'f3_mean': 0.0, 'f3_std': 0.0, 'f3_median': 0.0,
                'f2_f1_ratio': 0.0
            }
    
    def extract_all_features(self, audio: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
        """
        Extract all available features from audio.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Dictionary containing all extracted features
        """
        all_features = {}
        
        print("Extracting MFCC features...")
        mfcc_features = self.extract_mfcc_features(audio)
        all_features.update(mfcc_features)
        
        print("Extracting spectral features...")
        spectral_features = self.extract_spectral_features(audio)
        all_features.update(spectral_features)
        
        print("Extracting pitch features...")
        pitch_features = self.extract_pitch_features(audio)
        all_features.update(pitch_features)
        
        print("Extracting prosodic features...")
        prosodic_features = self.extract_prosodic_features(audio)
        all_features.update(prosodic_features)
        
        print("Extracting formant features...")
        formant_features = self.extract_formant_features(audio)
        all_features.update(formant_features)
        
        return all_features
    
    def features_to_vector(self, features: Dict[str, Union[float, np.ndarray]], 
                          exclude_raw: bool = True) -> np.ndarray:
        """
        Convert feature dictionary to feature vector.
        
        Args:
            features: Dictionary of extracted features
            exclude_raw: Whether to exclude raw features (like mfcc_raw)
            
        Returns:
            Feature vector as numpy array
        """
        feature_vector = []
        
        for key, value in features.items():
            if exclude_raw and 'raw' in key:
                continue
                
            if isinstance(value, np.ndarray):
                if value.ndim == 1:
                    feature_vector.extend(value)
                else:
                    # Flatten multi-dimensional arrays
                    feature_vector.extend(value.flatten())
            else:
                feature_vector.append(value)
        
        return np.array(feature_vector)
    
    def get_feature_names(self, features: Dict[str, Union[float, np.ndarray]], 
                         exclude_raw: bool = True) -> List[str]:
        """
        Get names of features in the same order as features_to_vector.
        
        Args:
            features: Dictionary of extracted features
            exclude_raw: Whether to exclude raw features
            
        Returns:
            List of feature names
        """
        feature_names = []
        
        for key, value in features.items():
            if exclude_raw and 'raw' in key:
                continue
                
            if isinstance(value, np.ndarray):
                if value.ndim == 1:
                    for i in range(len(value)):
                        feature_names.append(f"{key}_{i}")
                else:
                    # Handle multi-dimensional arrays
                    flat_value = value.flatten()
                    for i in range(len(flat_value)):
                        feature_names.append(f"{key}_{i}")
            else:
                feature_names.append(key)
        
        return feature_names


# Example usage and testing
if __name__ == "__main__":
    # Create feature extractor
    extractor = FeatureExtractor()
    
    # Create synthetic test audio
    duration = 2.0
    t = np.linspace(0, duration, int(duration * extractor.sr))
    # Complex test signal with multiple frequencies
    test_audio = (0.3 * np.sin(2 * np.pi * 220 * t) + 
                  0.2 * np.sin(2 * np.pi * 440 * t) + 
                  0.1 * np.sin(2 * np.pi * 880 * t))
    
    print("Testing FeatureExtractor...")
    
    # Test individual feature extraction
    print("Testing MFCC extraction...")
    mfcc_features = extractor.extract_mfcc_features(test_audio)
    print(f"MFCC features extracted: {len(mfcc_features)} types")
    
    print("Testing spectral features...")
    spectral_features = extractor.extract_spectral_features(test_audio)
    print(f"Spectral features: {len(spectral_features)}")
    
    print("Testing pitch features...")
    pitch_features = extractor.extract_pitch_features(test_audio)
    print(f"Pitch features: {len(pitch_features)}")
    
    # Test complete feature extraction
    print("Testing complete feature extraction...")
    all_features = extractor.extract_all_features(test_audio)
    print(f"Total features extracted: {len(all_features)}")
    
    # Test feature vector conversion
    feature_vector = extractor.features_to_vector(all_features)
    feature_names = extractor.get_feature_names(all_features)
    
    print(f"Feature vector shape: {feature_vector.shape}")
    print(f"Number of feature names: {len(feature_names)}")
    
    print("FeatureExtractor test completed successfully!")