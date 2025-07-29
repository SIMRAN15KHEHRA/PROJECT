"""
Voice Simulation Module for EchoTwin

This module simulates various voice disorders and damage effects:
- Vocal fatigue simulation
- Vocal nodule effects
- Vocal fold paralysis
- Breathiness and roughness
- Pitch instability (jitter/shimmer)
- Formant modifications
- Harmonic structure alterations
"""

import numpy as np
import librosa
import scipy.signal as signal
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import parselmouth
from parselmouth.praat import call
import warnings
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


class VoiceSimulator:
    """
    Comprehensive voice disorder simulation system.
    
    Simulates various voice pathologies by modifying:
    - Fundamental frequency (F0) and its stability
    - Harmonic structure and noise components
    - Formant frequencies and bandwidths
    - Amplitude modulations
    - Spectral characteristics
    """
    
    def __init__(self, sr: int = 16000):
        """
        Initialize VoiceSimulator.
        
        Args:
            sr: Sampling rate for audio processing
        """
        self.sr = sr
        
    def simulate_vocal_fatigue(self, 
                              audio: np.ndarray, 
                              severity: float = 0.5,
                              f0_drop: float = 0.1,
                              noise_level: float = 0.05,
                              tremor_freq: float = 4.0) -> np.ndarray:
        """
        Simulate vocal fatigue effects.
        
        Vocal fatigue typically causes:
        - Gradual F0 drop throughout speech
        - Increased breathiness/noise
        - Slight tremor
        - Reduced vocal intensity
        
        Args:
            audio: Input audio signal
            severity: Severity of fatigue (0.0 to 1.0)
            f0_drop: Amount of F0 drop as percentage
            noise_level: Added noise level
            tremor_freq: Tremor frequency in Hz
            
        Returns:
            Simulated fatigued voice
        """
        # Create time vector
        duration = len(audio) / self.sr
        t = np.linspace(0, duration, len(audio))
        
        # Simulate gradual F0 drop
        f0_modulation = 1.0 - (f0_drop * severity * t / duration)
        
        # Add slight tremor
        tremor = 1.0 + 0.02 * severity * np.sin(2 * np.pi * tremor_freq * t)
        f0_modulation *= tremor
        
        # Apply pitch modulation
        modified_audio = self._apply_pitch_modulation(audio, f0_modulation)
        
        # Add breathiness (noise)
        noise = np.random.normal(0, noise_level * severity, len(audio))
        modified_audio = modified_audio + noise
        
        # Reduce overall amplitude to simulate reduced vocal effort
        amplitude_reduction = 1.0 - 0.2 * severity
        modified_audio *= amplitude_reduction
        
        return np.clip(modified_audio, -1.0, 1.0)
    
    def simulate_vocal_nodules(self, 
                              audio: np.ndarray, 
                              severity: float = 0.5,
                              roughness: float = 0.3,
                              breathiness: float = 0.2,
                              f0_instability: float = 0.1) -> np.ndarray:
        """
        Simulate vocal nodule effects.
        
        Vocal nodules typically cause:
        - Increased roughness and breathiness
        - F0 instability (jitter)
        - Reduced harmonic clarity
        - Diplophonia (double voicing) in severe cases
        
        Args:
            audio: Input audio signal
            severity: Severity of nodules (0.0 to 1.0)
            roughness: Amount of roughness to add
            breathiness: Amount of breathiness to add
            f0_instability: Amount of F0 instability
            
        Returns:
            Simulated voice with nodule effects
        """
        # Add roughness through amplitude modulation
        rough_audio = self._add_roughness(audio, roughness * severity)
        
        # Add breathiness (turbulent noise)
        breathy_audio = self._add_breathiness(rough_audio, breathiness * severity)
        
        # Add F0 instability (jitter)
        unstable_audio = self._add_f0_instability(breathy_audio, f0_instability * severity)
        
        # Reduce harmonic clarity
        degraded_audio = self._degrade_harmonics(unstable_audio, severity * 0.3)
        
        # For severe cases, add diplophonia
        if severity > 0.7:
            diplophonic_audio = self._add_diplophonia(degraded_audio, severity * 0.2)
            return np.clip(diplophonic_audio, -1.0, 1.0)
        
        return np.clip(degraded_audio, -1.0, 1.0)
    
    def simulate_vocal_paralysis(self, 
                                audio: np.ndarray, 
                                severity: float = 0.5,
                                side: str = 'unilateral',
                                breathiness: float = 0.4,
                                weakness: float = 0.3) -> np.ndarray:
        """
        Simulate vocal fold paralysis effects.
        
        Vocal paralysis typically causes:
        - Severe breathiness
        - Reduced vocal intensity
        - Possible diplophonia
        - Aspiration noise
        
        Args:
            audio: Input audio signal
            severity: Severity of paralysis (0.0 to 1.0)
            side: Type of paralysis ('unilateral' or 'bilateral')
            breathiness: Amount of breathiness
            weakness: Amount of vocal weakness
            
        Returns:
            Simulated voice with paralysis effects
        """
        # Severe breathiness
        breathy_audio = self._add_breathiness(audio, breathiness * severity, aspiration=True)
        
        # Vocal weakness (amplitude reduction)
        weak_audio = breathy_audio * (1.0 - weakness * severity)
        
        # Add aspiration noise
        aspiration_noise = self._generate_aspiration_noise(len(audio), severity * 0.1)
        aspirated_audio = weak_audio + aspiration_noise
        
        # For bilateral paralysis, effects are more severe
        if side == 'bilateral':
            # More severe breathiness and weakness
            aspirated_audio = self._add_breathiness(aspirated_audio, breathiness * severity * 0.5, aspiration=True)
            aspirated_audio *= (1.0 - weakness * severity * 0.3)
        
        return np.clip(aspirated_audio, -1.0, 1.0)
    
    def simulate_spasmodic_dysphonia(self, 
                                    audio: np.ndarray, 
                                    severity: float = 0.5,
                                    spasm_type: str = 'adductor',
                                    spasm_freq: float = 3.0) -> np.ndarray:
        """
        Simulate spasmodic dysphonia effects.
        
        Args:
            audio: Input audio signal
            severity: Severity of dysphonia (0.0 to 1.0)
            spasm_type: Type of spasms ('adductor' or 'abductor')
            spasm_freq: Frequency of spasms in Hz
            
        Returns:
            Simulated voice with spasmodic dysphonia
        """
        duration = len(audio) / self.sr
        t = np.linspace(0, duration, len(audio))
        
        if spasm_type == 'adductor':
            # Adductor spasms cause voice breaks and strain
            spasm_pattern = np.abs(np.sin(2 * np.pi * spasm_freq * t))
            strain_modulation = 1.0 + severity * 0.5 * spasm_pattern
            
            # Apply strain and occasional voice breaks
            strained_audio = audio * strain_modulation
            
            # Add voice breaks (sudden amplitude drops)
            break_pattern = (spasm_pattern > 0.8) * severity
            strained_audio *= (1.0 - break_pattern * 0.8)
            
        else:  # abductor
            # Abductor spasms cause breathy interruptions
            spasm_pattern = np.abs(np.sin(2 * np.pi * spasm_freq * t))
            
            # Add breathiness during spasms
            breathy_noise = np.random.normal(0, 0.05, len(audio))
            breathy_modulation = severity * spasm_pattern
            
            strained_audio = audio + breathy_noise * breathy_modulation
            
            # Reduce voice during spasms
            strained_audio *= (1.0 - breathy_modulation * 0.3)
        
        return np.clip(strained_audio, -1.0, 1.0)
    
    def simulate_presbyphonia(self, 
                             audio: np.ndarray, 
                             severity: float = 0.5,
                             tremor_freq: float = 5.0,
                             breathiness: float = 0.2) -> np.ndarray:
        """
        Simulate presbyphonia (age-related voice changes).
        
        Presbyphonia typically causes:
        - Voice tremor
        - Increased breathiness
        - Reduced vocal intensity
        - Pitch instability
        
        Args:
            audio: Input audio signal
            severity: Severity of aging effects (0.0 to 1.0)
            tremor_freq: Tremor frequency in Hz
            breathiness: Amount of breathiness
            
        Returns:
            Simulated aged voice
        """
        duration = len(audio) / self.sr
        t = np.linspace(0, duration, len(audio))
        
        # Add voice tremor
        tremor = 1.0 + severity * 0.05 * np.sin(2 * np.pi * tremor_freq * t)
        tremor_audio = audio * tremor
        
        # Add breathiness
        breathy_audio = self._add_breathiness(tremor_audio, breathiness * severity)
        
        # Add pitch instability
        unstable_audio = self._add_f0_instability(breathy_audio, severity * 0.08)
        
        # Reduce overall intensity
        aged_audio = unstable_audio * (1.0 - severity * 0.15)
        
        return np.clip(aged_audio, -1.0, 1.0)
    
    def _apply_pitch_modulation(self, audio: np.ndarray, modulation: np.ndarray) -> np.ndarray:
        """Apply pitch modulation to audio signal."""
        try:
            # Use librosa's pitch shifting with time-varying factors
            # This is a simplified approach - more sophisticated methods exist
            modified_audio = np.zeros_like(audio)
            
            # Process in overlapping windows
            window_size = 2048
            hop_size = 512
            
            for i in range(0, len(audio) - window_size, hop_size):
                window = audio[i:i + window_size]
                mod_factor = np.mean(modulation[i:i + window_size])
                
                # Convert modulation factor to semitones
                semitones = 12 * np.log2(mod_factor) if mod_factor > 0 else 0
                
                # Apply pitch shift
                shifted_window = librosa.effects.pitch_shift(
                    window, sr=self.sr, n_steps=semitones
                )
                
                # Overlap-add
                end_idx = min(i + window_size, len(modified_audio))
                modified_audio[i:end_idx] += shifted_window[:end_idx - i]
            
            return modified_audio
            
        except Exception:
            # Fallback to simple amplitude modulation if pitch shifting fails
            return audio * modulation
    
    def _add_roughness(self, audio: np.ndarray, amount: float) -> np.ndarray:
        """Add roughness through low-frequency amplitude modulation."""
        duration = len(audio) / self.sr
        t = np.linspace(0, duration, len(audio))
        
        # Multiple modulation frequencies for realistic roughness
        roughness_mod = (
            0.4 * np.sin(2 * np.pi * 30 * t) +
            0.3 * np.sin(2 * np.pi * 50 * t) +
            0.3 * np.sin(2 * np.pi * 80 * t)
        )
        
        modulation = 1.0 + amount * roughness_mod
        return audio * modulation
    
    def _add_breathiness(self, audio: np.ndarray, amount: float, aspiration: bool = False) -> np.ndarray:
        """Add breathiness through filtered noise."""
        # Generate noise
        noise = np.random.normal(0, 1, len(audio))
        
        # Filter noise to match speech spectrum
        # High-pass filter for breathiness
        b, a = signal.butter(4, 1000 / (self.sr / 2), btype='high')
        filtered_noise = signal.filtfilt(b, a, noise)
        
        if aspiration:
            # Add aspiration noise (higher frequency content)
            b_asp, a_asp = signal.butter(4, [2000, 8000] / (self.sr / 2), btype='band')
            aspiration_noise = signal.filtfilt(b_asp, a_asp, noise)
            filtered_noise += 0.5 * aspiration_noise
        
        # Scale noise
        noise_amplitude = amount * np.std(audio)
        scaled_noise = filtered_noise * noise_amplitude
        
        return audio + scaled_noise
    
    def _add_f0_instability(self, audio: np.ndarray, amount: float) -> np.ndarray:
        """Add F0 instability (jitter) through frequency modulation."""
        duration = len(audio) / self.sr
        t = np.linspace(0, duration, len(audio))
        
        # Random F0 variations
        jitter_freq = 10  # Hz
        jitter = amount * 0.02 * np.random.normal(0, 1, len(audio))
        
        # Smooth the jitter
        jitter_smooth = gaussian_filter1d(jitter, sigma=self.sr / jitter_freq)
        
        # Apply as frequency modulation
        modulation = 1.0 + jitter_smooth
        
        return self._apply_pitch_modulation(audio, modulation)
    
    def _degrade_harmonics(self, audio: np.ndarray, amount: float) -> np.ndarray:
        """Degrade harmonic structure by adding inharmonic components."""
        # Get STFT
        stft = librosa.stft(audio, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Add random noise to magnitude spectrum
        noise_magnitude = amount * 0.1 * np.random.normal(0, 1, magnitude.shape)
        degraded_magnitude = magnitude + noise_magnitude * magnitude
        
        # Reconstruct
        degraded_stft = degraded_magnitude * np.exp(1j * phase)
        degraded_audio = librosa.istft(degraded_stft, hop_length=512)
        
        return degraded_audio
    
    def _add_diplophonia(self, audio: np.ndarray, amount: float) -> np.ndarray:
        """Add diplophonia (double voicing) effect."""
        # Create a slightly pitch-shifted version
        semitones = amount * 2  # Small pitch difference
        shifted_audio = librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=semitones)
        
        # Mix with original
        diplophonic_audio = audio + amount * shifted_audio
        
        return diplophonic_audio
    
    def _generate_aspiration_noise(self, length: int, amplitude: float) -> np.ndarray:
        """Generate aspiration noise."""
        # High-frequency noise
        noise = np.random.normal(0, amplitude, length)
        
        # Filter to aspiration frequency range
        b, a = signal.butter(4, [1500, 6000] / (self.sr / 2), btype='band')
        aspiration = signal.filtfilt(b, a, noise)
        
        return aspiration
    
    def simulate_custom_disorder(self, 
                                audio: np.ndarray, 
                                parameters: Dict[str, float]) -> np.ndarray:
        """
        Simulate custom voice disorder with specified parameters.
        
        Args:
            audio: Input audio signal
            parameters: Dictionary of disorder parameters
                - 'breathiness': 0.0 to 1.0
                - 'roughness': 0.0 to 1.0
                - 'f0_instability': 0.0 to 1.0
                - 'amplitude_reduction': 0.0 to 1.0
                - 'tremor_freq': Tremor frequency in Hz
                - 'tremor_amplitude': 0.0 to 1.0
                - 'noise_level': 0.0 to 1.0
                
        Returns:
            Simulated voice with custom disorder
        """
        modified_audio = audio.copy()
        
        # Apply breathiness
        if 'breathiness' in parameters:
            modified_audio = self._add_breathiness(modified_audio, parameters['breathiness'])
        
        # Apply roughness
        if 'roughness' in parameters:
            modified_audio = self._add_roughness(modified_audio, parameters['roughness'])
        
        # Apply F0 instability
        if 'f0_instability' in parameters:
            modified_audio = self._add_f0_instability(modified_audio, parameters['f0_instability'])
        
        # Apply tremor
        if 'tremor_freq' in parameters and 'tremor_amplitude' in parameters:
            duration = len(audio) / self.sr
            t = np.linspace(0, duration, len(audio))
            tremor = 1.0 + parameters['tremor_amplitude'] * np.sin(2 * np.pi * parameters['tremor_freq'] * t)
            modified_audio *= tremor
        
        # Apply amplitude reduction
        if 'amplitude_reduction' in parameters:
            modified_audio *= (1.0 - parameters['amplitude_reduction'])
        
        # Add general noise
        if 'noise_level' in parameters:
            noise = np.random.normal(0, parameters['noise_level'], len(audio))
            modified_audio += noise
        
        return np.clip(modified_audio, -1.0, 1.0)
    
    def create_progression_simulation(self, 
                                     audio: np.ndarray, 
                                     disorder_type: str,
                                     progression_steps: int = 5) -> List[np.ndarray]:
        """
        Create a progression of voice disorder from mild to severe.
        
        Args:
            audio: Input audio signal
            disorder_type: Type of disorder to simulate
            progression_steps: Number of progression steps
            
        Returns:
            List of audio samples showing disorder progression
        """
        progression = []
        severities = np.linspace(0.1, 0.9, progression_steps)
        
        for severity in severities:
            if disorder_type == 'vocal_fatigue':
                simulated = self.simulate_vocal_fatigue(audio, severity=severity)
            elif disorder_type == 'vocal_nodules':
                simulated = self.simulate_vocal_nodules(audio, severity=severity)
            elif disorder_type == 'vocal_paralysis':
                simulated = self.simulate_vocal_paralysis(audio, severity=severity)
            elif disorder_type == 'spasmodic_dysphonia':
                simulated = self.simulate_spasmodic_dysphonia(audio, severity=severity)
            elif disorder_type == 'presbyphonia':
                simulated = self.simulate_presbyphonia(audio, severity=severity)
            else:
                raise ValueError(f"Unknown disorder type: {disorder_type}")
            
            progression.append(simulated)
        
        return progression
    
    def get_available_disorders(self) -> List[str]:
        """Get list of available disorder simulations."""
        return [
            'vocal_fatigue',
            'vocal_nodules', 
            'vocal_paralysis',
            'spasmodic_dysphonia',
            'presbyphonia'
        ]
    
    def get_disorder_parameters(self, disorder_type: str) -> Dict[str, Dict]:
        """
        Get parameter descriptions for a specific disorder.
        
        Args:
            disorder_type: Type of disorder
            
        Returns:
            Dictionary of parameter descriptions
        """
        parameters = {
            'vocal_fatigue': {
                'severity': {'range': [0.0, 1.0], 'description': 'Overall severity of fatigue'},
                'f0_drop': {'range': [0.0, 0.3], 'description': 'Amount of F0 drop'},
                'noise_level': {'range': [0.0, 0.1], 'description': 'Added noise level'},
                'tremor_freq': {'range': [2.0, 8.0], 'description': 'Tremor frequency in Hz'}
            },
            'vocal_nodules': {
                'severity': {'range': [0.0, 1.0], 'description': 'Overall severity of nodules'},
                'roughness': {'range': [0.0, 0.5], 'description': 'Amount of roughness'},
                'breathiness': {'range': [0.0, 0.4], 'description': 'Amount of breathiness'},
                'f0_instability': {'range': [0.0, 0.2], 'description': 'F0 instability amount'}
            },
            'vocal_paralysis': {
                'severity': {'range': [0.0, 1.0], 'description': 'Overall severity of paralysis'},
                'side': {'options': ['unilateral', 'bilateral'], 'description': 'Type of paralysis'},
                'breathiness': {'range': [0.0, 0.6], 'description': 'Amount of breathiness'},
                'weakness': {'range': [0.0, 0.5], 'description': 'Amount of vocal weakness'}
            },
            'spasmodic_dysphonia': {
                'severity': {'range': [0.0, 1.0], 'description': 'Overall severity of dysphonia'},
                'spasm_type': {'options': ['adductor', 'abductor'], 'description': 'Type of spasms'},
                'spasm_freq': {'range': [1.0, 8.0], 'description': 'Frequency of spasms in Hz'}
            },
            'presbyphonia': {
                'severity': {'range': [0.0, 1.0], 'description': 'Overall severity of aging'},
                'tremor_freq': {'range': [3.0, 8.0], 'description': 'Tremor frequency in Hz'},
                'breathiness': {'range': [0.0, 0.4], 'description': 'Amount of breathiness'}
            }
        }
        
        return parameters.get(disorder_type, {})


# Example usage and testing
if __name__ == "__main__":
    # Create simulator
    simulator = VoiceSimulator()
    
    # Create synthetic test audio
    duration = 2.0
    t = np.linspace(0, duration, int(duration * simulator.sr))
    # Create a more complex test signal
    test_audio = (0.5 * np.sin(2 * np.pi * 220 * t) + 
                  0.2 * np.sin(2 * np.pi * 440 * t) + 
                  0.1 * np.sin(2 * np.pi * 660 * t))
    
    print("Testing VoiceSimulator...")
    
    # Test different disorder simulations
    print("Testing vocal fatigue simulation...")
    fatigued = simulator.simulate_vocal_fatigue(test_audio, severity=0.5)
    print(f"Fatigued audio shape: {fatigued.shape}")
    
    print("Testing vocal nodule simulation...")
    nodules = simulator.simulate_vocal_nodules(test_audio, severity=0.6)
    print(f"Nodules audio shape: {nodules.shape}")
    
    print("Testing vocal paralysis simulation...")
    paralysis = simulator.simulate_vocal_paralysis(test_audio, severity=0.7)
    print(f"Paralysis audio shape: {paralysis.shape}")
    
    print("Testing progression simulation...")
    progression = simulator.create_progression_simulation(
        test_audio, 'vocal_fatigue', progression_steps=3
    )
    print(f"Progression steps: {len(progression)}")
    
    # Test custom disorder
    print("Testing custom disorder simulation...")
    custom_params = {
        'breathiness': 0.3,
        'roughness': 0.2,
        'f0_instability': 0.1,
        'tremor_freq': 5.0,
        'tremor_amplitude': 0.05
    }
    custom_disorder = simulator.simulate_custom_disorder(test_audio, custom_params)
    print(f"Custom disorder audio shape: {custom_disorder.shape}")
    
    # Show available disorders
    print(f"Available disorders: {simulator.get_available_disorders()}")
    
    print("VoiceSimulator test completed successfully!")