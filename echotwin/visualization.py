"""
Visualization Module for EchoTwin

This module provides comprehensive visualization tools for voice analysis:
- Spectrograms and waveform plots
- Pitch and F0 visualization
- MFCC and feature visualizations
- Voice health metrics plots
- Comparison visualizations
- Interactive plots for analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from scipy import signal
import parselmouth
from parselmouth.praat import call
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
import io
import base64

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class VoiceVisualizer:
    """
    Comprehensive visualization system for voice analysis.
    
    Provides static and interactive visualizations for:
    - Audio waveforms and spectrograms
    - Voice features and health metrics
    - Disorder progression and comparisons
    - Analysis results and predictions
    """
    
    def __init__(self, sr: int = 16000, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize VoiceVisualizer.
        
        Args:
            sr: Sampling rate for audio processing
            figsize: Default figure size for matplotlib plots
        """
        self.sr = sr
        self.figsize = figsize
        
    def plot_waveform(self, 
                     audio: np.ndarray, 
                     title: str = "Audio Waveform",
                     interactive: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        Plot audio waveform.
        
        Args:
            audio: Audio signal
            title: Plot title
            interactive: Whether to create interactive plot
            
        Returns:
            Matplotlib or Plotly figure
        """
        time = np.linspace(0, len(audio) / self.sr, len(audio))
        
        if interactive:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=time, y=audio,
                mode='lines',
                name='Waveform',
                line=dict(color='blue', width=1)
            ))
            fig.update_layout(
                title=title,
                xaxis_title='Time (s)',
                yaxis_title='Amplitude',
                hovermode='x unified'
            )
            return fig
        else:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.plot(time, audio, color='blue', linewidth=0.8)
            ax.set_title(title)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig
    
    def plot_spectrogram(self, 
                        audio: np.ndarray,
                        title: str = "Spectrogram",
                        hop_length: int = 512,
                        n_fft: int = 2048,
                        interactive: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        Plot spectrogram of audio signal.
        
        Args:
            audio: Audio signal
            title: Plot title
            hop_length: Hop length for STFT
            n_fft: FFT window size
            interactive: Whether to create interactive plot
            
        Returns:
            Matplotlib or Plotly figure
        """
        # Compute spectrogram
        stft = librosa.stft(audio, hop_length=hop_length, n_fft=n_fft)
        magnitude_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        
        if interactive:
            # Create time and frequency axes
            times = librosa.frames_to_time(np.arange(magnitude_db.shape[1]), 
                                         sr=self.sr, hop_length=hop_length)
            frequencies = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)
            
            fig = go.Figure(data=go.Heatmap(
                z=magnitude_db,
                x=times,
                y=frequencies,
                colorscale='Viridis',
                colorbar=dict(title='Magnitude (dB)')
            ))
            fig.update_layout(
                title=title,
                xaxis_title='Time (s)',
                yaxis_title='Frequency (Hz)'
            )
            return fig
        else:
            fig, ax = plt.subplots(figsize=self.figsize)
            librosa.display.specshow(
                magnitude_db, 
                sr=self.sr, 
                hop_length=hop_length,
                x_axis='time', 
                y_axis='hz',
                ax=ax,
                cmap='viridis'
            )
            ax.set_title(title)
            plt.colorbar(ax.collections[0], ax=ax, label='Magnitude (dB)')
            plt.tight_layout()
            return fig
    
    def plot_pitch_contour(self, 
                          audio: np.ndarray,
                          title: str = "Pitch Contour",
                          f0_min: float = 75,
                          f0_max: float = 600,
                          interactive: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        Plot F0/pitch contour.
        
        Args:
            audio: Audio signal
            title: Plot title
            f0_min: Minimum F0 frequency
            f0_max: Maximum F0 frequency
            interactive: Whether to create interactive plot
            
        Returns:
            Matplotlib or Plotly figure
        """
        try:
            # Extract pitch using Parselmouth
            sound = parselmouth.Sound(audio, sampling_frequency=self.sr)
            pitch = call(sound, "To Pitch", 0.0, f0_min, f0_max)
            
            # Get pitch values and times
            pitch_values = pitch.selected_array['frequency']
            pitch_times = pitch.xs()
            
            # Remove unvoiced frames (0 Hz)
            voiced_mask = pitch_values > 0
            voiced_times = pitch_times[voiced_mask]
            voiced_pitch = pitch_values[voiced_mask]
            
            if interactive:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=voiced_times, y=voiced_pitch,
                    mode='lines+markers',
                    name='F0',
                    line=dict(color='red', width=2),
                    marker=dict(size=3)
                ))
                fig.update_layout(
                    title=title,
                    xaxis_title='Time (s)',
                    yaxis_title='Frequency (Hz)',
                    hovermode='x unified'
                )
                return fig
            else:
                fig, ax = plt.subplots(figsize=self.figsize)
                ax.plot(voiced_times, voiced_pitch, 'r-', linewidth=2, marker='o', markersize=2)
                ax.set_title(title)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Frequency (Hz)')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                return fig
                
        except Exception as e:
            print(f"Error extracting pitch: {e}")
            # Return empty plot
            if interactive:
                fig = go.Figure()
                fig.update_layout(title=f"{title} - Error extracting pitch")
                return fig
            else:
                fig, ax = plt.subplots(figsize=self.figsize)
                ax.set_title(f"{title} - Error extracting pitch")
                return fig
    
    def plot_mfcc(self, 
                  audio: np.ndarray,
                  title: str = "MFCC Features",
                  n_mfcc: int = 13,
                  interactive: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        Plot MFCC features.
        
        Args:
            audio: Audio signal
            title: Plot title
            n_mfcc: Number of MFCC coefficients
            interactive: Whether to create interactive plot
            
        Returns:
            Matplotlib or Plotly figure
        """
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=n_mfcc)
        
        if interactive:
            times = librosa.frames_to_time(np.arange(mfccs.shape[1]), sr=self.sr)
            
            fig = go.Figure(data=go.Heatmap(
                z=mfccs,
                x=times,
                y=list(range(1, n_mfcc + 1)),
                colorscale='RdBu_r',
                colorbar=dict(title='MFCC Value')
            ))
            fig.update_layout(
                title=title,
                xaxis_title='Time (s)',
                yaxis_title='MFCC Coefficient'
            )
            return fig
        else:
            fig, ax = plt.subplots(figsize=self.figsize)
            librosa.display.specshow(
                mfccs, 
                sr=self.sr,
                x_axis='time',
                ax=ax,
                cmap='RdBu_r'
            )
            ax.set_title(title)
            ax.set_ylabel('MFCC Coefficient')
            plt.colorbar(ax.collections[0], ax=ax, label='MFCC Value')
            plt.tight_layout()
            return fig
    
    def plot_feature_comparison(self, 
                               features1: Dict[str, float],
                               features2: Dict[str, float],
                               labels: Tuple[str, str] = ("Original", "Modified"),
                               title: str = "Feature Comparison",
                               interactive: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        Compare two sets of voice features.
        
        Args:
            features1: First set of features
            features2: Second set of features
            labels: Labels for the two sets
            title: Plot title
            interactive: Whether to create interactive plot
            
        Returns:
            Matplotlib or Plotly figure
        """
        # Find common features
        common_features = set(features1.keys()) & set(features2.keys())
        common_features = [f for f in common_features if isinstance(features1[f], (int, float))]
        
        if not common_features:
            print("No common numerical features found")
            return None
        
        # Prepare data
        feature_names = list(common_features)
        values1 = [features1[f] for f in feature_names]
        values2 = [features2[f] for f in feature_names]
        
        if interactive:
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name=labels[0],
                x=feature_names,
                y=values1,
                marker_color='lightblue'
            ))
            
            fig.add_trace(go.Bar(
                name=labels[1],
                x=feature_names,
                y=values2,
                marker_color='lightcoral'
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title='Feature',
                yaxis_title='Value',
                barmode='group',
                xaxis_tickangle=-45
            )
            return fig
        else:
            fig, ax = plt.subplots(figsize=(max(12, len(feature_names) * 0.8), 8))
            
            x = np.arange(len(feature_names))
            width = 0.35
            
            ax.bar(x - width/2, values1, width, label=labels[0], color='lightblue')
            ax.bar(x + width/2, values2, width, label=labels[1], color='lightcoral')
            
            ax.set_title(title)
            ax.set_xlabel('Feature')
            ax.set_ylabel('Value')
            ax.set_xticks(x)
            ax.set_xticklabels(feature_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig
    
    def plot_voice_health_dashboard(self, 
                                   features: Dict[str, float],
                                   classification_result: Optional[str] = None,
                                   confidence: Optional[float] = None,
                                   interactive: bool = True) -> Union[plt.Figure, go.Figure]:
        """
        Create a comprehensive voice health dashboard.
        
        Args:
            features: Extracted voice features
            classification_result: Classification result
            confidence: Classification confidence
            interactive: Whether to create interactive plot
            
        Returns:
            Matplotlib or Plotly figure
        """
        if interactive:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Pitch Statistics', 'Voice Quality', 'Spectral Features', 'Classification'),
                specs=[[{"type": "bar"}, {"type": "indicator"}],
                       [{"type": "bar"}, {"type": "pie"}]]
            )
            
            # Pitch statistics
            pitch_features = ['f0_mean', 'f0_std', 'f0_range']
            pitch_values = [features.get(f, 0) for f in pitch_features]
            
            fig.add_trace(go.Bar(
                x=pitch_features,
                y=pitch_values,
                name='Pitch',
                marker_color='lightblue'
            ), row=1, col=1)
            
            # Voice quality gauge
            hnr = features.get('hnr', 0)
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=hnr,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "HNR (dB)"},
                gauge={'axis': {'range': [None, 30]},
                      'bar': {'color': "darkblue"},
                      'steps': [{'range': [0, 10], 'color': "lightgray"},
                               {'range': [10, 20], 'color': "gray"}],
                      'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 15}}
            ), row=1, col=2)
            
            # Spectral features
            spectral_features = ['spectral_centroid_mean', 'spectral_rolloff_mean', 'zcr_mean']
            spectral_values = [features.get(f, 0) for f in spectral_features]
            
            fig.add_trace(go.Bar(
                x=spectral_features,
                y=spectral_values,
                name='Spectral',
                marker_color='lightgreen'
            ), row=2, col=1)
            
            # Classification result
            if classification_result and confidence:
                fig.add_trace(go.Pie(
                    labels=[classification_result, 'Uncertainty'],
                    values=[confidence, 1-confidence],
                    name="Classification"
                ), row=2, col=2)
            
            fig.update_layout(height=800, showlegend=False, title_text="Voice Health Dashboard")
            return fig
            
        else:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Voice Health Dashboard', fontsize=16)
            
            # Pitch statistics
            pitch_features = ['f0_mean', 'f0_std', 'f0_range']
            pitch_values = [features.get(f, 0) for f in pitch_features]
            axes[0, 0].bar(pitch_features, pitch_values, color='lightblue')
            axes[0, 0].set_title('Pitch Statistics')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Voice quality
            quality_features = ['hnr', 'jitter_local', 'shimmer_local']
            quality_values = [features.get(f, 0) for f in quality_features]
            axes[0, 1].bar(quality_features, quality_values, color='lightcoral')
            axes[0, 1].set_title('Voice Quality Measures')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Spectral features
            spectral_features = ['spectral_centroid_mean', 'spectral_rolloff_mean', 'zcr_mean']
            spectral_values = [features.get(f, 0) for f in spectral_features]
            axes[1, 0].bar(spectral_features, spectral_values, color='lightgreen')
            axes[1, 0].set_title('Spectral Features')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Classification result
            if classification_result and confidence:
                labels = [classification_result, 'Uncertainty']
                sizes = [confidence, 1-confidence]
                axes[1, 1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                axes[1, 1].set_title('Classification Result')
            else:
                axes[1, 1].text(0.5, 0.5, 'No Classification\nResult Available', 
                              ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Classification Result')
            
            plt.tight_layout()
            return fig
    
    def plot_disorder_progression(self, 
                                 progression_audio: List[np.ndarray],
                                 disorder_type: str,
                                 interactive: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        Visualize disorder progression through multiple stages.
        
        Args:
            progression_audio: List of audio samples showing progression
            disorder_type: Type of disorder
            interactive: Whether to create interactive plot
            
        Returns:
            Matplotlib or Plotly figure
        """
        n_stages = len(progression_audio)
        
        if interactive:
            fig = make_subplots(
                rows=n_stages, cols=1,
                subplot_titles=[f'Stage {i+1}' for i in range(n_stages)],
                vertical_spacing=0.05
            )
            
            for i, audio in enumerate(progression_audio):
                time = np.linspace(0, len(audio) / self.sr, len(audio))
                fig.add_trace(go.Scatter(
                    x=time, y=audio,
                    mode='lines',
                    name=f'Stage {i+1}',
                    line=dict(width=1)
                ), row=i+1, col=1)
            
            fig.update_layout(
                height=200*n_stages,
                title_text=f'{disorder_type.replace("_", " ").title()} Progression',
                showlegend=False
            )
            return fig
            
        else:
            fig, axes = plt.subplots(n_stages, 1, figsize=(12, 3*n_stages))
            if n_stages == 1:
                axes = [axes]
            
            fig.suptitle(f'{disorder_type.replace("_", " ").title()} Progression', fontsize=16)
            
            for i, audio in enumerate(progression_audio):
                time = np.linspace(0, len(audio) / self.sr, len(audio))
                axes[i].plot(time, audio, linewidth=0.8)
                axes[i].set_title(f'Stage {i+1}')
                axes[i].set_xlabel('Time (s)')
                axes[i].set_ylabel('Amplitude')
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
    
    def plot_feature_importance(self, 
                               feature_names: List[str],
                               importance_scores: List[float],
                               title: str = "Feature Importance",
                               top_n: int = 20,
                               interactive: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        Plot feature importance scores.
        
        Args:
            feature_names: List of feature names
            importance_scores: Corresponding importance scores
            title: Plot title
            top_n: Number of top features to show
            interactive: Whether to create interactive plot
            
        Returns:
            Matplotlib or Plotly figure
        """
        # Sort by importance
        sorted_indices = np.argsort(importance_scores)[::-1]
        top_indices = sorted_indices[:top_n]
        
        top_names = [feature_names[i] for i in top_indices]
        top_scores = [importance_scores[i] for i in top_indices]
        
        if interactive:
            fig = go.Figure(go.Bar(
                x=top_scores,
                y=top_names,
                orientation='h',
                marker_color='skyblue'
            ))
            fig.update_layout(
                title=title,
                xaxis_title='Importance Score',
                yaxis_title='Feature',
                height=max(400, len(top_names) * 25)
            )
            return fig
        else:
            fig, ax = plt.subplots(figsize=(10, max(8, len(top_names) * 0.4)))
            y_pos = np.arange(len(top_names))
            
            ax.barh(y_pos, top_scores, color='skyblue')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_names)
            ax.invert_yaxis()
            ax.set_xlabel('Importance Score')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig
    
    def create_comparison_plot(self, 
                              original_audio: np.ndarray,
                              modified_audio: np.ndarray,
                              labels: Tuple[str, str] = ("Original", "Modified"),
                              interactive: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        Create side-by-side comparison of two audio signals.
        
        Args:
            original_audio: Original audio signal
            modified_audio: Modified audio signal
            labels: Labels for the two signals
            interactive: Whether to create interactive plot
            
        Returns:
            Matplotlib or Plotly figure
        """
        if interactive:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(f'{labels[0]} Waveform', f'{labels[1]} Waveform',
                              f'{labels[0]} Spectrogram', f'{labels[1]} Spectrogram'),
                specs=[[{"type": "scatter"}, {"type": "scatter"}],
                       [{"type": "heatmap"}, {"type": "heatmap"}]]
            )
            
            # Waveforms
            time1 = np.linspace(0, len(original_audio) / self.sr, len(original_audio))
            time2 = np.linspace(0, len(modified_audio) / self.sr, len(modified_audio))
            
            fig.add_trace(go.Scatter(x=time1, y=original_audio, mode='lines', name=labels[0]), row=1, col=1)
            fig.add_trace(go.Scatter(x=time2, y=modified_audio, mode='lines', name=labels[1]), row=1, col=2)
            
            # Spectrograms
            stft1 = librosa.stft(original_audio)
            stft2 = librosa.stft(modified_audio)
            mag1_db = librosa.amplitude_to_db(np.abs(stft1), ref=np.max)
            mag2_db = librosa.amplitude_to_db(np.abs(stft2), ref=np.max)
            
            times1 = librosa.frames_to_time(np.arange(mag1_db.shape[1]), sr=self.sr)
            times2 = librosa.frames_to_time(np.arange(mag2_db.shape[1]), sr=self.sr)
            freqs = librosa.fft_frequencies(sr=self.sr)
            
            fig.add_trace(go.Heatmap(z=mag1_db, x=times1, y=freqs, colorscale='Viridis'), row=2, col=1)
            fig.add_trace(go.Heatmap(z=mag2_db, x=times2, y=freqs, colorscale='Viridis'), row=2, col=2)
            
            fig.update_layout(height=800, showlegend=False, title_text="Audio Comparison")
            return fig
            
        else:
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle('Audio Comparison', fontsize=16)
            
            # Waveforms
            time1 = np.linspace(0, len(original_audio) / self.sr, len(original_audio))
            time2 = np.linspace(0, len(modified_audio) / self.sr, len(modified_audio))
            
            axes[0, 0].plot(time1, original_audio, color='blue', linewidth=0.8)
            axes[0, 0].set_title(f'{labels[0]} Waveform')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Amplitude')
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].plot(time2, modified_audio, color='red', linewidth=0.8)
            axes[0, 1].set_title(f'{labels[1]} Waveform')
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Amplitude')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Spectrograms
            stft1 = librosa.stft(original_audio)
            stft2 = librosa.stft(modified_audio)
            mag1_db = librosa.amplitude_to_db(np.abs(stft1), ref=np.max)
            mag2_db = librosa.amplitude_to_db(np.abs(stft2), ref=np.max)
            
            librosa.display.specshow(mag1_db, sr=self.sr, x_axis='time', y_axis='hz', ax=axes[1, 0], cmap='viridis')
            axes[1, 0].set_title(f'{labels[0]} Spectrogram')
            
            librosa.display.specshow(mag2_db, sr=self.sr, x_axis='time', y_axis='hz', ax=axes[1, 1], cmap='viridis')
            axes[1, 1].set_title(f'{labels[1]} Spectrogram')
            
            plt.tight_layout()
            return fig
    
    def save_plot(self, fig: plt.Figure, filename: str, dpi: int = 300) -> None:
        """
        Save matplotlib figure to file.
        
        Args:
            fig: Matplotlib figure
            filename: Output filename
            dpi: Resolution in DPI
        """
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {filename}")
    
    def fig_to_base64(self, fig: plt.Figure) -> str:
        """
        Convert matplotlib figure to base64 string for web display.
        
        Args:
            fig: Matplotlib figure
            
        Returns:
            Base64 encoded string
        """
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        return image_base64


# Example usage and testing
if __name__ == "__main__":
    # Create visualizer
    visualizer = VoiceVisualizer()
    
    # Create synthetic test audio
    duration = 2.0
    t = np.linspace(0, duration, int(duration * visualizer.sr))
    test_audio = 0.5 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.random.normal(0, 1, len(t))
    
    print("Testing VoiceVisualizer...")
    
    # Test waveform plot
    print("Testing waveform plot...")
    waveform_fig = visualizer.plot_waveform(test_audio)
    print("Waveform plot created successfully")
    
    # Test spectrogram
    print("Testing spectrogram...")
    spec_fig = visualizer.plot_spectrogram(test_audio)
    print("Spectrogram created successfully")
    
    # Test pitch contour
    print("Testing pitch contour...")
    pitch_fig = visualizer.plot_pitch_contour(test_audio)
    print("Pitch contour created successfully")
    
    # Test MFCC plot
    print("Testing MFCC plot...")
    mfcc_fig = visualizer.plot_mfcc(test_audio)
    print("MFCC plot created successfully")
    
    # Test feature comparison
    print("Testing feature comparison...")
    features1 = {'f0_mean': 220, 'f0_std': 15, 'hnr': 12}
    features2 = {'f0_mean': 200, 'f0_std': 20, 'hnr': 8}
    comparison_fig = visualizer.plot_feature_comparison(features1, features2)
    print("Feature comparison created successfully")
    
    # Test voice health dashboard
    print("Testing voice health dashboard...")
    features = {
        'f0_mean': 220, 'f0_std': 15, 'f0_range': 50,
        'hnr': 12, 'jitter_local': 0.01, 'shimmer_local': 0.05,
        'spectral_centroid_mean': 2000, 'spectral_rolloff_mean': 4000, 'zcr_mean': 0.1
    }
    dashboard_fig = visualizer.plot_voice_health_dashboard(
        features, classification_result='healthy', confidence=0.85, interactive=False
    )
    print("Voice health dashboard created successfully")
    
    plt.close('all')  # Close all figures to save memory
    print("VoiceVisualizer test completed successfully!")