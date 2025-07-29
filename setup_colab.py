"""
Google Colab Setup Script for EchoTwin

This script sets up the EchoTwin environment in Google Colab and provides
a simplified demonstration of the voice analysis capabilities.

Run this in a Colab cell to get started with EchoTwin!
"""

# Install required packages
def install_dependencies():
    """Install all required dependencies for EchoTwin."""
    import subprocess
    import sys
    
    packages = [
        'librosa>=0.10.0',
        'soundfile>=0.12.1',
        'scipy>=1.10.0',
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'plotly>=5.15.0',
        'scikit-learn>=1.3.0',
        'torch>=2.0.0',
        'torchaudio>=2.0.0',
        'gradio>=3.40.0',
        'ipywidgets>=8.0.0',
        'praat-parselmouth>=0.4.3'
    ]
    
    print("🔧 Installing EchoTwin dependencies...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
    
    print("🎉 Installation completed!")

def create_sample_audio():
    """Create sample audio for demonstration."""
    import numpy as np
    
    def generate_voice_sample(duration=3.0, sr=16000, voice_type='clean'):
        """Generate synthetic voice samples."""
        t = np.linspace(0, duration, int(duration * sr))
        
        if voice_type == 'clean':
            # Clean voice-like signal
            audio = (
                0.3 * np.sin(2 * np.pi * 220 * t) +  # F0
                0.2 * np.sin(2 * np.pi * 440 * t) +  # First harmonic
                0.1 * np.sin(2 * np.pi * 660 * t) +  # Second harmonic
                0.05 * np.random.normal(0, 1, len(t))  # Small amount of noise
            )
        elif voice_type == 'rough':
            # Rough voice with amplitude modulation
            modulation = 1 + 0.3 * np.sin(2 * np.pi * 30 * t)
            audio = (
                0.3 * np.sin(2 * np.pi * 180 * t) * modulation +
                0.2 * np.sin(2 * np.pi * 360 * t) * modulation +
                0.1 * np.random.normal(0, 1, len(t))
            )
        elif voice_type == 'breathy':
            # Breathy voice with more noise
            audio = (
                0.2 * np.sin(2 * np.pi * 200 * t) +
                0.15 * np.sin(2 * np.pi * 400 * t) +
                0.3 * np.random.normal(0, 1, len(t))
            )
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8
        return audio, sr
    
    return generate_voice_sample

def create_simple_classifier():
    """Create a simple voice classifier for demonstration."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    # Generate synthetic training data
    np.random.seed(42)
    n_samples = 1000
    
    # Features for healthy voices
    healthy_features = np.random.normal([200, 15, 2000, 0.1, 0.15], 
                                       [30, 5, 300, 0.02, 0.03], 
                                       (n_samples//2, 5))
    
    # Features for disordered voices
    disorder_features = np.random.normal([180, 25, 1800, 0.15, 0.12], 
                                        [40, 8, 400, 0.04, 0.05], 
                                        (n_samples//2, 5))
    
    # Combine data
    X = np.vstack([healthy_features, disorder_features])
    y = np.array(['healthy'] * (n_samples//2) + ['disorder'] * (n_samples//2))
    
    # Train classifier
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_scaled, y)
    
    return classifier, scaler

def extract_basic_features(audio, sr):
    """Extract basic voice features."""
    import librosa
    import numpy as np
    
    # MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    rms = librosa.feature.rms(y=audio)[0]
    
    # Basic pitch estimation
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    f0_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            f0_values.append(pitch)
    
    f0_mean = np.mean(f0_values) if f0_values else 0
    f0_std = np.std(f0_values) if f0_values else 0
    
    features = [
        f0_mean,
        f0_std,
        np.mean(spectral_centroids),
        np.mean(zcr),
        np.mean(rms)
    ]
    
    return np.array(features)

def create_echotwin_demo():
    """Create a complete EchoTwin demonstration."""
    import numpy as np
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display
    from IPython.display import Audio, display, HTML
    import gradio as gr
    import tempfile
    import soundfile as sf
    
    print("🎙️ Creating EchoTwin Demo...")
    
    # Setup
    generate_audio = create_sample_audio()
    classifier, scaler = create_simple_classifier()
    
    # Create sample audio
    clean_audio, sr = generate_audio(voice_type='clean')
    rough_audio, sr = generate_audio(voice_type='rough')
    breathy_audio, sr = generate_audio(voice_type='breathy')
    
    print("✅ Sample audio created!")
    
    # Voice simulation functions
    def simulate_vocal_fatigue(audio, severity=0.5, sr=16000):
        """Simulate vocal fatigue."""
        duration = len(audio) / sr
        t = np.linspace(0, duration, len(audio))
        
        # F0 drop and tremor
        f0_modulation = 1.0 - (0.1 * severity * t / duration)
        tremor = 1.0 + 0.02 * severity * np.sin(2 * np.pi * 4 * t)
        modulation = f0_modulation * tremor
        
        # Apply effects
        modified_audio = audio * modulation
        noise = np.random.normal(0, 0.05 * severity, len(audio))
        modified_audio = modified_audio + noise
        modified_audio *= (1.0 - 0.2 * severity)
        
        return np.clip(modified_audio, -1.0, 1.0)
    
    def simulate_vocal_nodules(audio, severity=0.5, sr=16000):
        """Simulate vocal nodules."""
        duration = len(audio) / sr
        t = np.linspace(0, duration, len(audio))
        
        # Roughness modulation
        roughness_mod = 0.4 * np.sin(2 * np.pi * 30 * t)
        modulation = 1.0 + 0.3 * severity * roughness_mod
        rough_audio = audio * modulation
        
        # Add breathiness
        noise = np.random.normal(0, 0.2 * severity, len(audio))
        return np.clip(rough_audio + noise, -1.0, 1.0)
    
    # Analysis function
    def analyze_voice(audio_file):
        """Analyze uploaded voice."""
        if audio_file is None:
            return "No audio uploaded", None
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_file, sr=16000)
            
            # Extract features
            features = extract_basic_features(audio, sr)
            
            # Classify
            features_scaled = scaler.transform(features.reshape(1, -1))
            prediction = classifier.predict(features_scaled)[0]
            confidence = np.max(classifier.predict_proba(features_scaled)[0])
            
            # Create visualization
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # Waveform
            time = np.linspace(0, len(audio) / sr, len(audio))
            axes[0].plot(time, audio, linewidth=0.8)
            axes[0].set_title('Audio Waveform')
            axes[0].set_xlabel('Time (s)')
            axes[0].set_ylabel('Amplitude')
            axes[0].grid(True, alpha=0.3)
            
            # Spectrogram
            D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=axes[1])
            axes[1].set_title('Spectrogram')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = tempfile.mktemp(suffix='.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            result_text = f"""
            🔍 **Voice Analysis Results:**
            
            **Classification:** {prediction.title()}
            **Confidence:** {confidence:.2%}
            
            **Audio Duration:** {len(audio) / sr:.2f} seconds
            **Sampling Rate:** {sr} Hz
            """
            
            return result_text, plot_path
            
        except Exception as e:
            return f"Error analyzing audio: {str(e)}", None
    
    def simulate_disorder(audio_file, disorder_type, severity):
        """Simulate voice disorder."""
        if audio_file is None:
            return "No audio uploaded", None
        
        try:
            audio, sr = librosa.load(audio_file, sr=16000)
            
            if disorder_type == "Vocal Fatigue":
                simulated = simulate_vocal_fatigue(audio, severity=severity, sr=sr)
            elif disorder_type == "Vocal Nodules":
                simulated = simulate_vocal_nodules(audio, severity=severity, sr=sr)
            else:
                return "Unknown disorder type", None
            
            # Save simulated audio
            output_path = tempfile.mktemp(suffix='.wav')
            sf.write(output_path, simulated, sr)
            
            result_text = f"✅ Successfully simulated {disorder_type} with severity {severity:.1f}"
            return result_text, output_path
            
        except Exception as e:
            return f"Error simulating disorder: {str(e)}", None
    
    # Create Gradio interface
    with gr.Blocks(title="EchoTwin Demo", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🎙️ EchoTwin - Voice Analysis Demo
        
        Upload an audio file to analyze voice health or simulate voice disorders.
        
        ⚠️ **Note:** This is a demonstration with synthetic models.
        """)
        
        with gr.Tabs():
            with gr.TabItem("🔍 Voice Analysis"):
                with gr.Row():
                    with gr.Column():
                        audio_input = gr.Audio(label="Upload Audio", type="filepath")
                        analyze_btn = gr.Button("Analyze Voice", variant="primary")
                    with gr.Column():
                        analysis_result = gr.Markdown(label="Results")
                
                analysis_plot = gr.Image(label="Analysis")
                
                analyze_btn.click(
                    fn=analyze_voice,
                    inputs=[audio_input],
                    outputs=[analysis_result, analysis_plot]
                )
            
            with gr.TabItem("🎭 Voice Simulation"):
                with gr.Row():
                    with gr.Column():
                        sim_audio_input = gr.Audio(label="Upload Audio", type="filepath")
                        disorder_type = gr.Dropdown(
                            choices=["Vocal Fatigue", "Vocal Nodules"],
                            label="Disorder Type",
                            value="Vocal Fatigue"
                        )
                        severity = gr.Slider(0.1, 0.9, 0.5, step=0.1, label="Severity")
                        simulate_btn = gr.Button("Simulate", variant="primary")
                    with gr.Column():
                        sim_result = gr.Markdown(label="Results")
                        sim_audio_output = gr.Audio(label="Simulated Audio")
                
                simulate_btn.click(
                    fn=simulate_disorder,
                    inputs=[sim_audio_input, disorder_type, severity],
                    outputs=[sim_result, sim_audio_output]
                )
    
    print("🚀 Launching EchoTwin Demo Interface...")
    return interface

# Main execution
if __name__ == "__main__":
    print("🎙️ EchoTwin Google Colab Setup")
    print("=" * 50)
    
    # Install dependencies
    install_dependencies()
    
    # Create and launch demo
    demo = create_echotwin_demo()
    demo.launch(share=True, debug=True)
    
    print("✅ EchoTwin setup completed!")
    print("🌐 Use the public link to access the interface from anywhere!")