"""
Gradio UI Module for EchoTwin

This module provides a comprehensive web interface for the EchoTwin system:
- Audio upload and preprocessing
- Voice feature extraction and analysis
- Voice disorder classification
- Voice simulation and damage modeling
- Interactive visualizations and comparisons
- Results export and sharing
"""

import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import soundfile as sf
import io
import base64
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
import tempfile
import os

# Import EchoTwin modules
from .audio_processor import AudioProcessor
from .feature_extractor import FeatureExtractor
from .voice_classifier import VoiceClassifier
from .voice_simulator import VoiceSimulator
from .visualization import VoiceVisualizer

warnings.filterwarnings('ignore')


class EchoTwinUI:
    """
    Comprehensive Gradio UI for the EchoTwin voice analysis system.
    
    Provides an intuitive web interface for:
    - Audio upload and analysis
    - Voice disorder detection
    - Voice simulation and comparison
    - Interactive visualizations
    - Results export
    """
    
    def __init__(self):
        """Initialize the EchoTwin UI with all necessary components."""
        self.audio_processor = AudioProcessor()
        self.feature_extractor = FeatureExtractor()
        self.voice_classifier = VoiceClassifier(model_type='cnn')
        self.voice_simulator = VoiceSimulator()
        self.visualizer = VoiceVisualizer()
        
        # State variables
        self.current_audio = None
        self.current_features = None
        self.current_sr = 16000
        
        # Pre-train a simple model with synthetic data for demo
        self._initialize_demo_model()
    
    def _initialize_demo_model(self):
        """Initialize a demo model with synthetic data."""
        print("Initializing demo model...")
        
        # Generate synthetic training data
        n_samples = 1000
        n_features = 50  # Reduced for demo
        
        # Create synthetic features
        X = np.random.randn(n_samples, n_features)
        # Create synthetic labels (healthy vs disorder)
        y = np.random.choice(['healthy', 'disorder'], n_samples)
        
        # Add some structure to make it more realistic
        disorder_mask = y == 'disorder'
        X[disorder_mask] += np.random.normal(0.5, 0.2, (np.sum(disorder_mask), n_features))
        
        try:
            # Train the classifier
            self.voice_classifier.train(X, y, epochs=20, verbose=False)
            print("Demo model initialized successfully!")
        except Exception as e:
            print(f"Warning: Could not initialize demo model: {e}")
    
    def process_audio_upload(self, audio_file) -> Tuple[str, str, str]:
        """
        Process uploaded audio file.
        
        Args:
            audio_file: Uploaded audio file
            
        Returns:
            Tuple of (status_message, audio_info, waveform_plot)
        """
        try:
            if audio_file is None:
                return "No audio file uploaded.", "", ""
            
            # Load audio
            audio_data, sr = self.audio_processor.load_audio(audio_file)
            self.current_audio = audio_data
            self.current_sr = sr
            
            # Get audio info
            info = self.audio_processor.get_audio_info(audio_data)
            info_text = f"""
            **Audio Information:**
            - Duration: {info['duration']:.2f} seconds
            - Samples: {info['samples']:,}
            - Sampling Rate: {info['sampling_rate']} Hz
            - Peak Amplitude: {info['peak_amplitude']:.3f}
            - RMS Amplitude: {info['rms_amplitude']:.3f}
            """
            
            # Create waveform plot
            fig = self.visualizer.plot_waveform(audio_data, title="Uploaded Audio Waveform")
            
            # Convert to base64 for display
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            
            return "Audio uploaded and processed successfully!", info_text, f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            return f"Error processing audio: {str(e)}", "", ""
    
    def extract_and_analyze_features(self) -> Tuple[str, str, str]:
        """
        Extract features from current audio and create visualizations.
        
        Returns:
            Tuple of (status_message, features_table, analysis_plots)
        """
        try:
            if self.current_audio is None:
                return "No audio loaded. Please upload an audio file first.", "", ""
            
            # Extract features
            print("Extracting features...")
            features = self.feature_extractor.extract_all_features(self.current_audio)
            self.current_features = features
            
            # Create features table
            feature_data = []
            for key, value in features.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    feature_data.append([key, f"{value:.4f}"])
                elif isinstance(value, np.ndarray) and value.ndim == 1:
                    feature_data.append([key, f"Array shape: {value.shape}"])
            
            features_df = pd.DataFrame(feature_data, columns=["Feature", "Value"])
            features_table = features_df.to_html(index=False, table_id="features-table")
            
            # Create analysis plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Voice Feature Analysis', fontsize=16)
            
            # Spectrogram
            stft = self.audio_processor.sr
            spec_fig = self.visualizer.plot_spectrogram(self.current_audio, interactive=False)
            
            # MFCC
            mfcc_fig = self.visualizer.plot_mfcc(self.current_audio, interactive=False)
            
            # Pitch contour
            pitch_fig = self.visualizer.plot_pitch_contour(self.current_audio, interactive=False)
            
            # Voice health dashboard
            dashboard_fig = self.visualizer.plot_voice_health_dashboard(
                features, interactive=False
            )
            
            # Convert to base64
            buffer = io.BytesIO()
            dashboard_fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Close figures to save memory
            plt.close('all')
            
            return "Features extracted successfully!", features_table, f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            return f"Error extracting features: {str(e)}", "", ""
    
    def classify_voice(self) -> Tuple[str, str]:
        """
        Classify the current audio for voice disorders.
        
        Returns:
            Tuple of (classification_result, confidence_info)
        """
        try:
            if self.current_features is None:
                return "No features available. Please extract features first.", ""
            
            if not self.voice_classifier.trained:
                return "Classifier not trained. Using demo mode with synthetic predictions.", ""
            
            # Convert features to vector (simplified for demo)
            feature_vector = []
            for key, value in self.current_features.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    feature_vector.append(value)
            
            # Pad or truncate to match expected size
            if len(feature_vector) > 50:
                feature_vector = feature_vector[:50]
            elif len(feature_vector) < 50:
                feature_vector.extend([0.0] * (50 - len(feature_vector)))
            
            feature_array = np.array(feature_vector).reshape(1, -1)
            
            try:
                # Make prediction
                prediction = self.voice_classifier.predict(feature_array)[0]
                probabilities = self.voice_classifier.predict_proba(feature_array)[0]
                confidence = np.max(probabilities)
                
                result_text = f"""
                **Classification Result:**
                - Prediction: **{prediction}**
                - Confidence: **{confidence:.2%}**
                """
                
                # Create confidence visualization
                labels = self.voice_classifier.label_encoder.classes_
                conf_text = "**Class Probabilities:**\n"
                for label, prob in zip(labels, probabilities):
                    conf_text += f"- {label}: {prob:.2%}\n"
                
                return result_text, conf_text
                
            except Exception as e:
                # Fallback to demo prediction
                demo_prediction = np.random.choice(['healthy', 'disorder'])
                demo_confidence = np.random.uniform(0.6, 0.9)
                
                result_text = f"""
                **Demo Classification Result:**
                - Prediction: **{demo_prediction}**
                - Confidence: **{demo_confidence:.2%}**
                - Note: This is a demo prediction with synthetic model
                """
                
                conf_text = f"""
                **Demo Probabilities:**
                - healthy: {demo_confidence if demo_prediction == 'healthy' else 1-demo_confidence:.2%}
                - disorder: {1-demo_confidence if demo_prediction == 'healthy' else demo_confidence:.2%}
                """
                
                return result_text, conf_text
                
        except Exception as e:
            return f"Error in classification: {str(e)}", ""
    
    def simulate_voice_disorder(self, disorder_type: str, severity: float) -> Tuple[str, str, Any]:
        """
        Simulate voice disorder on current audio.
        
        Args:
            disorder_type: Type of disorder to simulate
            severity: Severity level (0.0 to 1.0)
            
        Returns:
            Tuple of (status_message, audio_comparison, simulated_audio)
        """
        try:
            if self.current_audio is None:
                return "No audio loaded. Please upload an audio file first.", "", None
            
            # Simulate disorder
            if disorder_type == "Vocal Fatigue":
                simulated_audio = self.voice_simulator.simulate_vocal_fatigue(
                    self.current_audio, severity=severity
                )
            elif disorder_type == "Vocal Nodules":
                simulated_audio = self.voice_simulator.simulate_vocal_nodules(
                    self.current_audio, severity=severity
                )
            elif disorder_type == "Vocal Paralysis":
                simulated_audio = self.voice_simulator.simulate_vocal_paralysis(
                    self.current_audio, severity=severity
                )
            elif disorder_type == "Spasmodic Dysphonia":
                simulated_audio = self.voice_simulator.simulate_spasmodic_dysphonia(
                    self.current_audio, severity=severity
                )
            elif disorder_type == "Presbyphonia":
                simulated_audio = self.voice_simulator.simulate_presbyphonia(
                    self.current_audio, severity=severity
                )
            else:
                return f"Unknown disorder type: {disorder_type}", "", None
            
            # Create comparison visualization
            comparison_fig = self.visualizer.create_comparison_plot(
                self.current_audio, simulated_audio, 
                labels=("Original", f"{disorder_type} (Severity: {severity:.1f})"),
                interactive=False
            )
            
            # Convert to base64
            buffer = io.BytesIO()
            comparison_fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(comparison_fig)
            
            # Save simulated audio to temp file for playback
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                sf.write(tmp_file.name, simulated_audio, self.current_sr)
                temp_path = tmp_file.name
            
            status_msg = f"Successfully simulated {disorder_type} with severity {severity:.1f}"
            
            return status_msg, f"data:image/png;base64,{img_base64}", temp_path
            
        except Exception as e:
            return f"Error simulating disorder: {str(e)}", "", None
    
    def create_progression_simulation(self, disorder_type: str, steps: int) -> Tuple[str, str]:
        """
        Create disorder progression simulation.
        
        Args:
            disorder_type: Type of disorder
            steps: Number of progression steps
            
        Returns:
            Tuple of (status_message, progression_plot)
        """
        try:
            if self.current_audio is None:
                return "No audio loaded. Please upload an audio file first.", ""
            
            # Create progression
            disorder_key = disorder_type.lower().replace(" ", "_")
            progression = self.voice_simulator.create_progression_simulation(
                self.current_audio, disorder_key, progression_steps=steps
            )
            
            # Create visualization
            progression_fig = self.visualizer.plot_disorder_progression(
                progression, disorder_key, interactive=False
            )
            
            # Convert to base64
            buffer = io.BytesIO()
            progression_fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(progression_fig)
            
            status_msg = f"Created {steps}-step progression for {disorder_type}"
            
            return status_msg, f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            return f"Error creating progression: {str(e)}", ""
    
    def export_results(self) -> Tuple[str, str]:
        """
        Export analysis results to file.
        
        Returns:
            Tuple of (status_message, download_link)
        """
        try:
            if self.current_features is None:
                return "No analysis results to export.", ""
            
            # Create results dictionary
            results = {
                'audio_info': self.audio_processor.get_audio_info(self.current_audio) if self.current_audio is not None else {},
                'features': {k: v for k, v in self.current_features.items() if isinstance(v, (int, float, str))},
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
                import json
                json.dump(results, tmp_file, indent=2, default=str)
                temp_path = tmp_file.name
            
            return "Results exported successfully!", temp_path
            
        except Exception as e:
            return f"Error exporting results: {str(e)}", ""
    
    def create_interface(self) -> gr.Blocks:
        """
        Create the main Gradio interface.
        
        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(title="EchoTwin - Voice Analysis & Simulation", theme=gr.themes.Soft()) as interface:
            
            # Header
            gr.Markdown("""
            # 🎙️ EchoTwin - AI-Powered Voice Analysis & Simulation
            
            **Digital Twin for Voice Health Analysis**
            
            Upload your voice recording to analyze voice health, detect potential disorders, 
            and simulate various voice conditions for research and clinical applications.
            """)
            
            # Main tabs
            with gr.Tabs():
                
                # Tab 1: Audio Upload & Analysis
                with gr.TabItem("📤 Audio Upload & Analysis"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            audio_input = gr.Audio(
                                label="Upload Audio File",
                                type="filepath",
                                sources=["upload", "microphone"]
                            )
                            
                            upload_btn = gr.Button("Process Audio", variant="primary")
                            
                        with gr.Column(scale=2):
                            upload_status = gr.Textbox(
                                label="Status",
                                interactive=False
                            )
                            
                            audio_info = gr.Markdown(label="Audio Information")
                            
                    waveform_plot = gr.Image(label="Waveform")
                    
                    # Feature extraction section
                    gr.Markdown("## 🔍 Feature Extraction")
                    
                    with gr.Row():
                        extract_btn = gr.Button("Extract Features", variant="secondary")
                        
                    with gr.Row():
                        with gr.Column(scale=1):
                            extract_status = gr.Textbox(
                                label="Extraction Status",
                                interactive=False
                            )
                            
                        with gr.Column(scale=2):
                            features_table = gr.HTML(label="Extracted Features")
                    
                    analysis_plots = gr.Image(label="Feature Analysis")
                
                # Tab 2: Voice Classification
                with gr.TabItem("🔬 Voice Disorder Detection"):
                    gr.Markdown("""
                    ## Voice Health Classification
                    Analyze the uploaded audio for potential voice disorders using AI classification.
                    """)
                    
                    classify_btn = gr.Button("Classify Voice", variant="primary")
                    
                    with gr.Row():
                        with gr.Column():
                            classification_result = gr.Markdown(label="Classification Result")
                            
                        with gr.Column():
                            confidence_info = gr.Markdown(label="Confidence Information")
                
                # Tab 3: Voice Simulation
                with gr.TabItem("🎭 Voice Simulation"):
                    gr.Markdown("""
                    ## Voice Disorder Simulation
                    Simulate various voice disorders to understand their effects on speech.
                    """)
                    
                    with gr.Row():
                        with gr.Column():
                            disorder_type = gr.Dropdown(
                                choices=["Vocal Fatigue", "Vocal Nodules", "Vocal Paralysis", 
                                        "Spasmodic Dysphonia", "Presbyphonia"],
                                label="Disorder Type",
                                value="Vocal Fatigue"
                            )
                            
                            severity_slider = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.5,
                                step=0.1,
                                label="Severity Level"
                            )
                            
                            simulate_btn = gr.Button("Simulate Disorder", variant="primary")
                            
                        with gr.Column():
                            simulation_status = gr.Textbox(
                                label="Simulation Status",
                                interactive=False
                            )
                            
                            simulated_audio = gr.Audio(
                                label="Simulated Audio",
                                type="filepath"
                            )
                    
                    comparison_plot = gr.Image(label="Original vs Simulated Comparison")
                
                # Tab 4: Progression Analysis
                with gr.TabItem("📈 Disorder Progression"):
                    gr.Markdown("""
                    ## Disorder Progression Simulation
                    Visualize how voice disorders progress from mild to severe stages.
                    """)
                    
                    with gr.Row():
                        with gr.Column():
                            prog_disorder_type = gr.Dropdown(
                                choices=["Vocal Fatigue", "Vocal Nodules", "Vocal Paralysis", 
                                        "Spasmodic Dysphonia", "Presbyphonia"],
                                label="Disorder Type",
                                value="Vocal Fatigue"
                            )
                            
                            prog_steps = gr.Slider(
                                minimum=3,
                                maximum=10,
                                value=5,
                                step=1,
                                label="Progression Steps"
                            )
                            
                            progression_btn = gr.Button("Create Progression", variant="primary")
                            
                        with gr.Column():
                            progression_status = gr.Textbox(
                                label="Status",
                                interactive=False
                            )
                    
                    progression_plot = gr.Image(label="Disorder Progression")
                
                # Tab 5: Results & Export
                with gr.TabItem("💾 Results & Export"):
                    gr.Markdown("""
                    ## Export Analysis Results
                    Download your voice analysis results for further study or clinical use.
                    """)
                    
                    export_btn = gr.Button("Export Results", variant="secondary")
                    
                    with gr.Row():
                        export_status = gr.Textbox(
                            label="Export Status",
                            interactive=False
                        )
                        
                        download_file = gr.File(
                            label="Download Results",
                            visible=False
                        )
            
            # Event handlers
            upload_btn.click(
                fn=self.process_audio_upload,
                inputs=[audio_input],
                outputs=[upload_status, audio_info, waveform_plot]
            )
            
            extract_btn.click(
                fn=self.extract_and_analyze_features,
                inputs=[],
                outputs=[extract_status, features_table, analysis_plots]
            )
            
            classify_btn.click(
                fn=self.classify_voice,
                inputs=[],
                outputs=[classification_result, confidence_info]
            )
            
            simulate_btn.click(
                fn=self.simulate_voice_disorder,
                inputs=[disorder_type, severity_slider],
                outputs=[simulation_status, comparison_plot, simulated_audio]
            )
            
            progression_btn.click(
                fn=self.create_progression_simulation,
                inputs=[prog_disorder_type, prog_steps],
                outputs=[progression_status, progression_plot]
            )
            
            export_btn.click(
                fn=self.export_results,
                inputs=[],
                outputs=[export_status, download_file]
            )
            
            # Footer
            gr.Markdown("""
            ---
            **EchoTwin** - Developed for voice health research and clinical applications.
            
            ⚠️ **Disclaimer**: This tool is for research and educational purposes. 
            Always consult healthcare professionals for medical diagnosis and treatment.
            """)
        
        return interface
    
    def launch(self, 
               share: bool = False, 
               server_name: str = "127.0.0.1", 
               server_port: int = 7860,
               debug: bool = False) -> None:
        """
        Launch the Gradio interface.
        
        Args:
            share: Whether to create a public link
            server_name: Server hostname
            server_port: Server port
            debug: Enable debug mode
        """
        interface = self.create_interface()
        
        print("🎙️ Starting EchoTwin Voice Analysis System...")
        print(f"📡 Server will be available at: http://{server_name}:{server_port}")
        
        if share:
            print("🌐 Creating public link...")
        
        interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            debug=debug,
            show_error=True
        )


# Standalone launch function
def launch_echotwin_ui(**kwargs):
    """
    Standalone function to launch EchoTwin UI.
    
    Args:
        **kwargs: Arguments to pass to launch method
    """
    ui = EchoTwinUI()
    ui.launch(**kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Create and launch the UI
    ui = EchoTwinUI()
    ui.launch(debug=True)