import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

# Your project modules
from config import get_experiment
from models.registry import build_model, load_checkpoint

warnings.filterwarnings("ignore")

# --- SETTINGS & CONSTANTS ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["CD", "HYP", "MI", "NORM", "STTC"]
CHECKPOINT_DIR = Path("./checkpoints")

# --- MODEL CONFIGURATION ---
MODELS_CONFIG = {
    "🌟 InceptionTime 1D (100Hz)": {
        "preset": "inceptiontime_100hz",
        "pt_file": "inceptiontime_100hz.pt",
        "thr_file": "inceptiontime_100hz_thresholds.npy"
    },
    "⚡ SE-ResNet 1D (100Hz)": {
        "preset": "seresnet1d_100hz",
        "pt_file": "seresnet1d_100hz.pt",
        "thr_file": "seresnet1d_100hz_thresholds.npy"
    },
    "🚀 ScratchResNet18 2D CWT (100Hz)": {
        "preset": "scratchresnet18_cwt224_100hz",
        "pt_file": "scratchresnet18_cwt224_100hz.pt",
        "thr_file": "scratchresnet18_cwt224_100hz_thresholds.npy" 
    },
    "🧠 ResNet50 Pretrained 2D CWT (100Hz)": {
        "preset": "resnet2d_cwt224_partial_ft_100hz",
        "pt_file": "resnet2d_cwt224_partial_ft_100hz.pt",
        "thr_file": "resnet2d_cwt224_partial_ft_100hz_thresholds.npy"
    }
}

loaded_models = {}
loaded_thresholds = {}

def load_model_system(model_display_name):
    config = MODELS_CONFIG[model_display_name]
    preset_name = config["preset"]
    
    if preset_name in loaded_models:
        return loaded_models[preset_name], loaded_thresholds[preset_name]
    
    print(f"Loading {preset_name} into memory...")
    cfg = get_experiment(preset_name)
    model = build_model(cfg.model)
    
    pt_path = CHECKPOINT_DIR / config["pt_file"]
    thr_path = CHECKPOINT_DIR / config["thr_file"]
    
    if not pt_path.exists():
        raise FileNotFoundError(f"Missing weights file: {pt_path}")
        
    load_checkpoint(model, pt_path, device=DEVICE)
    model.to(DEVICE)
    model.eval()
    
    if thr_path.exists():
        thresholds = np.load(thr_path)
    else:
        thresholds = np.array([0.5] * 5)
        
    loaded_models[preset_name] = model
    loaded_thresholds[preset_name] = thresholds
    return model, thresholds

# --- DATA PREPROCESSING ---
def preprocess_1d(ecg_signal, strategy):
    """Dynamic normalization to fix 1D Model Bias."""
    ecg_signal = ecg_signal.astype(np.float32)
    eps = 1e-6
    
    if strategy == "Raw (No Normalization)":
        normalized = ecg_signal
    elif strategy == "Z-Score (Per Lead)":
        means = np.mean(ecg_signal, axis=1, keepdims=True)
        stds = np.std(ecg_signal, axis=1, keepdims=True)
        normalized = (ecg_signal - means) / (stds + eps)
    elif strategy == "Z-Score (Global)":
        mean = np.mean(ecg_signal)
        std = np.std(ecg_signal)
        normalized = (ecg_signal - mean) / (std + eps)
    elif strategy == "Min-Max (0 to 1)":
        min_val = np.min(ecg_signal, axis=1, keepdims=True)
        max_val = np.max(ecg_signal, axis=1, keepdims=True)
        normalized = (ecg_signal - min_val) / (max_val - min_val + eps)
        
    return torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)

def preprocess_2d_cwt(ecg_signal):
    if ecg_signal.ndim != 3 or ecg_signal.shape[1:] != (224, 224):
        raise ValueError(f"Shape Mismatch: Expected (12, 224, 224) for 2D models.")
    
    eps = 1e-6
    means = np.mean(ecg_signal, axis=(1, 2), keepdims=True)
    stds = np.std(ecg_signal, axis=(1, 2), keepdims=True)
    normalized = (ecg_signal - means) / (stds + eps)

    return torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)

# --- VISUALIZATION ---
def create_dashboard_plot(probs, thresholds, true_labels, model_name):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')
    
    colors = ['#2b6cb0' if c in true_labels else '#cbd5e0' for c in CLASSES]
    percentages = [p * 100 for p in probs]
    
    bars = ax.barh(CLASSES, percentages, color=colors, edgecolor='none', height=0.6)
    
    for i, bar in enumerate(bars):
        thr_percent = thresholds[i] * 100
        ax.plot([thr_percent, thr_percent], [bar.get_y() - 0.1, bar.get_y() + bar.get_height() + 0.1], 
                color='#e53e3e', linewidth=3, solid_capstyle='round', zorder=5)
        
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                va='center', fontsize=10, fontweight='bold', color='#2d3748')

    ax.set_xlim(0, 100)
    ax.set_xlabel('AI Confidence Level (%)', fontsize=11, fontweight='bold', color='#4a5568')
    ax.set_title(f'📊 Diagnostic Confidence: {model_name}\nRed Line: Tuned Decision Threshold | Blue Bar: Ground Truth', 
                 fontsize=12, fontweight='bold', color='#2d3748', pad=15)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig

# --- MAIN INFERENCE PIPELINE ---
def analyze_ecg(file_info, model_choice, norm_strategy):
    if file_info is None:
        return None, "Please upload an ECG data file."
    
    try:
        model, thresholds = load_model_system(model_choice)
        filename = Path(file_info.name).stem.upper()
        true_labels = [c for c in CLASSES if c in filename]
        
        ecg_data = np.load(file_info.name, allow_pickle=True)
        
        if ecg_data.dtype == object and ecg_data.ndim == 1:
            ecg_data = np.stack(ecg_data).astype(np.float32)
            
        if len(ecg_data.shape) == 2 and ecg_data.shape[0] != 12 and ecg_data.shape[1] == 12:
            ecg_data = ecg_data.T
            
        if ecg_data.shape[0] != 12:
            return None, f"Error: Uploaded file has {ecg_data.shape[0]} channels instead of 12."
        
        input_shape_str = str(ecg_data.shape)

        if "1D" in model_choice:
            if ecg_data.ndim != 2:
                return None, f"Expected 1D waveform, got {ecg_data.shape}."
            tensor_data = preprocess_1d(ecg_data, norm_strategy).to(DEVICE)
        else:
            tensor_data = preprocess_2d_cwt(ecg_data).to(DEVICE)
            
        with torch.no_grad():
            logits = model(tensor_data)
            
            # SMART SIGMOID DETECTION: If model already outputs 0-1, don't apply sigmoid again!
            if (logits >= 0).all() and (logits <= 1).all():
                probs = logits.cpu().numpy()[0]
            else:
                probs = torch.sigmoid(logits).cpu().numpy()[0]
            
        if np.isnan(probs).any():
            return None, "Error: The model returned NaNs. Check normalization."
            
        fig = create_dashboard_plot(probs, thresholds, true_labels, model_choice)
        predicted_classes = [CLASSES[i] for i in range(len(CLASSES)) if probs[i] >= thresholds[i]]
        
        report = f"### 📋 Clinical Diagnostic Report\n\n"
        report += f"**Active Architecture:** `{model_choice}`\n"
        report += f"**Processed Signal Shape:** `{input_shape_str}`\n\n"
        
        if not true_labels:
            report += "*(No ground truth labels found in filename.)*\n\n"
        else:
            report += f"- **Ground Truth (Actual):** {', '.join(true_labels)}\n"
            
        report += f"- **AI Diagnosis:** {', '.join(predicted_classes) if predicted_classes else 'Clear (No Anomalies)'}\n\n"
        
        if true_labels:
            report += "#### 🎯 Performance Evaluation:\n"
            correct_preds = set(true_labels).intersection(set(predicted_classes))
            missed_preds = set(true_labels).difference(set(predicted_classes))
            false_alarms = set(predicted_classes).difference(set(true_labels))
            
            if correct_preds:
                report += f"✅ **True Positives:** {', '.join(correct_preds)}\n"
            if missed_preds:
                report += f"❌ **False Negatives:** {', '.join(missed_preds)}\n"
            if false_alarms:
                report += f"⚠️ **False Positives:** {', '.join(false_alarms)}\n"

        return fig, report
        
    except Exception as e:
        return None, f"An error occurred: {str(e)}"

# --- UI DESIGN ---
custom_theme = gr.themes.Default(primary_hue="blue", secondary_hue="slate").set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
)

with gr.Blocks(theme=custom_theme, title="YTU ECG Diagnostic UI") as app:
    gr.HTML("""
        <div style="text-align: center; max-width: 800px; margin: 0 auto; padding-bottom: 20px;">
            <h1 style="color: #2b6cb0; font-size: 2.5rem; margin-bottom: 5px;">⚕️ Intelligent ECG Diagnostics</h1>
            <p style="color: #4a5568; font-size: 1.1rem;">Multi-Label Cardiovascular Disease Detection Pipeline</p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### ⚙️ System Configuration")
            model_dropdown = gr.Dropdown(
                choices=list(MODELS_CONFIG.keys()), 
                value=list(MODELS_CONFIG.keys())[0], 
                label="Select Neural Architecture"
            )
            
            with gr.Accordion("⚙️ Advanced 1D Preprocessing (Debug)", open=True):
                gr.Markdown("*If 1D predictions are stuck at 90+%, change this to match your training code!*")
                norm_dropdown = gr.Dropdown(
                    choices=["Raw (No Normalization)", "Z-Score (Per Lead)", "Z-Score (Global)", "Min-Max (0 to 1)"],
                    value="Raw (No Normalization)",
                    label="1D Normalization Strategy"
                )
            
            gr.Markdown("### 📂 Patient Data")
            file_input = gr.File(label="Upload ECG (.npy)", file_types=[".npy"])
            analyze_btn = gr.Button("🚀 Run Analysis", variant="primary", size="lg")
            
        with gr.Column(scale=2):
            plot_output = gr.Plot(label="Confidence & Threshold Analysis")
            report_output = gr.Markdown()
            
    analyze_btn.click(
        fn=analyze_ecg, 
        inputs=[file_input, model_dropdown, norm_dropdown], 
        outputs=[plot_output, report_output]
    )

if __name__ == "__main__":
    print("🚀 Initializing Diagnostic Server...")
    app.launch(inbrowser=True)