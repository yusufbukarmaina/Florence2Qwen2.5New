"""
Beaker Volume Estimator — Gradio Demo (Updated for Diagnostic Fixed Training)
Tabs:
  1. Live Inference   — upload image → get volume from both models
  2. Florence-2 Metrics — MAE / RMSE / R² + plots
  3. Qwen2.5-VL Metrics — same for Qwen
  4. Model Comparison — side-by-side comparison
  5. About
"""

import os, json, re, torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr
from transformers import AutoProcessor, AutoModelForCausalLM, Qwen2VLForConditionalGeneration
from peft import PeftModel

# Updated paths to match diagnostic training
FLORENCE_MODEL_DIR = "./trained_models/florence2_final"
QWEN_MODEL_DIR     = "./trained_models/qwen2_5vl_final"
METRICS_JSON       = "./trained_models/complete_metrics.json"
TEST_SET_DIR       = "./trained_models/test_set"

# Model base paths (for loading with PEFT)
FLORENCE_BASE = "microsoft/Florence-2-base"
QWEN_BASE = "Qwen/Qwen2-VL-2B-Instruct"

MAX_IMAGE_SIZE = 512

# ── helpers ──────────────────────────────────────────────────────────────────

def resize_image(img, max_size=MAX_IMAGE_SIZE):
    """Resize image to reduce memory usage"""
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    img = img.convert("RGB")
    w, h = img.size
    if w > max_size or h > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
    return img

def extract_volume(text):
    """Enhanced volume extraction matching training code"""
    if not text:
        return None
    
    text = str(text).strip()
    
    # Multiple patterns matching ImprovedVolumeExtractor
    patterns = [
        r'(\d+\.?\d*)\s*(?:ml|mL|ML)',
        r'(?:volume|Volume)[\s:]+(\d+\.?\d*)',
        r'^(\d+\.?\d*)$',
        r'(?:is|=|:)\s*(\d+\.?\d*)',
        r'(\d+\.?\d*)\s*$',
        r'\b(\d+\.?\d*)\b',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            for match in matches:
                try:
                    val = float(match)
                    if 0 <= val <= 250:  # Valid range
                        return val
                except ValueError:
                    continue
    
    return None

# ── lazy model cache ──────────────────────────────────────────────────────────

_fm = _fp = _qm = _qp = None

def load_florence():
    """Load Florence-2 with PEFT adapter"""
    global _fm, _fp
    if _fm: 
        return _fm, _fp
    
    if not os.path.isdir(FLORENCE_MODEL_DIR):
        raise FileNotFoundError(f"Florence model not found: {FLORENCE_MODEL_DIR}")
    
    print("Loading Florence-2 (base + PEFT adapter)...")
    _fp = AutoProcessor.from_pretrained(FLORENCE_MODEL_DIR, trust_remote_code=True)
    
    # Load base model
    base = AutoModelForCausalLM.from_pretrained(
        FLORENCE_BASE,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    # Load PEFT adapter
    _fm = PeftModel.from_pretrained(base, FLORENCE_MODEL_DIR)
    _fm.eval()
    
    print("✅ Florence-2 loaded successfully")
    return _fm, _fp

def load_qwen():
    """Load Qwen2.5-VL with PEFT adapter"""
    global _qm, _qp
    if _qm: 
        return _qm, _qp
    
    if not os.path.isdir(QWEN_MODEL_DIR):
        raise FileNotFoundError(f"Qwen model not found: {QWEN_MODEL_DIR}")
    
    print("Loading Qwen2.5-VL (base + PEFT adapter)...")
    _qp = AutoProcessor.from_pretrained(QWEN_MODEL_DIR, trust_remote_code=True)
    
    # Load base model
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        QWEN_BASE,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    # Load PEFT adapter
    _qm = PeftModel.from_pretrained(base, QWEN_MODEL_DIR)
    _qm.eval()
    
    print("✅ Qwen2.5-VL loaded successfully")
    return _qm, _qp

def move_inputs_to_device_dtype(inputs, model):
    """Move tensors to model device and dtype"""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    out = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device)
            if v.dtype.is_floating_point:
                v = v.to(dtype)
        out[k] = v
    return out

# ── inference ─────────────────────────────────────────────────────────────────

def predict_florence(img):
    """Predict volume using Florence-2 (matches diagnostic training prompt)"""
    model, proc = load_florence()
    img = resize_image(img)
    
    # MATCH TRAINING PROMPT
    prompt = "<VQA>What is the volume of liquid in this beaker measured in milliliters (mL)?"
    
    inputs = proc(images=img, text=prompt, return_tensors="pt", padding=True)
    inputs = move_inputs_to_device_dtype(inputs, model)
    
    with torch.no_grad():
        ids = model.generate(
            **inputs, 
            max_new_tokens=10,  # Just need a number
            do_sample=False,
            num_beams=1,
        )
    
    text = proc.batch_decode(ids, skip_special_tokens=True)[0]
    vol = extract_volume(text)
    
    return text, vol if vol is not None else 0.0

def predict_qwen(img):
    """Predict volume using Qwen2.5-VL (matches diagnostic training prompt)"""
    model, proc = load_qwen()
    img = resize_image(img)
    
    # MATCH TRAINING PROMPT
    question = "What is the volume of liquid in this beaker measured in milliliters (mL)?"
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": question},
            ],
        }
    ]
    
    text_prompt = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text_prompt], images=[img], return_tensors="pt", padding=True)
    
    # Move to device
    device = next(model.parameters()).device
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
    
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=10, do_sample=False, num_beams=1)
    
    # Decode only new tokens
    gen_ids = ids[0, inputs["input_ids"].shape[1]:]
    text = proc.decode(gen_ids, skip_special_tokens=True)
    
    vol = extract_volume(text)
    return text, vol if vol is not None else 0.0

def run_inference(pil_image):
    """Run both models on the input image"""
    if pil_image is None:
        return "—", "—", "—", "—"
    
    try:
        f_raw, f_vol = predict_florence(pil_image)
        f_disp = f"{f_vol:.1f} mL" if f_vol else "⚠️ Could not extract"
    except Exception as e:
        f_raw, f_disp = f"❌ Error: {str(e)}", "—"
    
    try:
        q_raw, q_vol = predict_qwen(pil_image)
        q_disp = f"{q_vol:.1f} mL" if q_vol else "⚠️ Could not extract"
    except Exception as e:
        q_raw, q_disp = f"❌ Error: {str(e)}", "—"
    
    return f_raw, f_disp, q_raw, q_disp

# ── metrics helpers ───────────────────────────────────────────────────────────

def load_metrics():
    """Load complete metrics from diagnostic training"""
    if not os.path.isfile(METRICS_JSON):
        return None
    with open(METRICS_JSON) as f:
        return json.load(f)

def get_model_metrics(metrics_data, model_name):
    """Extract evaluation metrics for a specific model"""
    if not metrics_data:
        return {}
    return metrics_data.get("evaluation_metrics", {}).get(model_name, {})

def metric_cards_html(m):
    """Generate HTML cards for metrics display"""
    mae  = m.get("mae",  float("nan"))
    rmse = m.get("rmse", float("nan"))
    r2   = m.get("r2",   float("nan"))
    mape = m.get("mape", float("nan"))
    used = m.get("num_used_samples", 0)
    failed = m.get("failed_extractions", 0)
    success_rate = m.get("extraction_success_rate", 0) * 100
    
    def card(title, val, unit, color="#3b82f6"):
        return (
            f"<div style='background:#1e293b;padding:20px;border-radius:12px;"
            f"text-align:center;min-width:140px;box-shadow:0 2px 8px #0004;"
            f"border-left:4px solid {color}'>"
            f"<div style='color:#94a3b8;font-size:13px;margin-bottom:6px;font-weight:600'>{title}</div>"
            f"<div style='color:#f8fafc;font-size:32px;font-weight:700;margin:4px 0'>{val}</div>"
            f"<div style='color:#64748b;font-size:11px;margin-top:4px'>{unit}</div>"
            f"</div>"
        )
    
    cards = (
        "<div style='display:flex;gap:16px;flex-wrap:wrap;padding:16px 0'>"
        + card("MAE",  f"{mae:.2f}",  "mL (lower is better)", "#3b82f6")
        + card("RMSE", f"{rmse:.2f}", "mL (lower is better)", "#8b5cf6")
        + card("R²",   f"{r2:.4f}",   "(-∞ to 1, higher is better)", "#10b981" if r2 > 0 else "#ef4444")
        + card("MAPE", f"{mape:.1f}", "% (lower is better)", "#f59e0b")
        + "</div>"
    )
    
    info = (
        f"<div style='background:#0f172a;padding:12px 20px;border-radius:8px;margin-top:12px'>"
        f"<p style='margin:4px 0;color:#94a3b8;font-size:13px'>"
        f"📊 <b>{used}</b> samples used | "
        f"❌ <b>{failed}</b> failed extractions | "
        f"✅ <b>{success_rate:.1f}%</b> extraction success rate"
        f"</p></div>"
    )
    
    return cards + info

def make_plots(metrics, model_name):
    """Generate prediction vs ground truth and error distribution plots"""
    preds = np.array(metrics.get("predictions", []))
    gt    = np.array(metrics.get("ground_truth", []))
    
    if len(preds) == 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, "No evaluation data available\nRun training first",
                ha="center", va="center", fontsize=14, color="#64748b")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig, fig
    
    # Scatter plot: Predicted vs Ground Truth
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    ax1.scatter(gt, preds, alpha=0.5, s=30, color="#38bdf8", edgecolors='black', linewidth=0.5, label="Predictions")
    
    # Perfect prediction line
    lo, hi = min(gt.min(), preds.min()), max(gt.max(), preds.max())
    ax1.plot([lo, hi], [lo, hi], "r--", lw=2.5, label="Perfect Prediction", alpha=0.8)
    
    ax1.set_xlabel("Ground Truth (mL)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Prediction (mL)", fontsize=12, fontweight='bold')
    ax1.set_title(f"{model_name} — Predicted vs Ground Truth", fontsize=13, fontweight="bold", pad=15)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Error distribution histogram
    err = preds - gt
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    
    n, bins, patches = ax2.hist(err, bins=30, edgecolor="black", alpha=0.75, color="#818cf8")
    
    # Color bars based on error magnitude
    for i, patch in enumerate(patches):
        if abs(bins[i]) < 10:
            patch.set_facecolor('#10b981')  # Green for small errors
        elif abs(bins[i]) < 30:
            patch.set_facecolor('#818cf8')  # Blue for medium errors
        else:
            patch.set_facecolor('#ef4444')  # Red for large errors
    
    ax2.axvline(0, color="red", lw=2.5, linestyle="--", label="Zero error", alpha=0.8)
    ax2.axvline(err.mean(), color="lime", lw=2.5, linestyle="--",
                label=f"Mean error: {err.mean():.2f} mL", alpha=0.8)
    
    ax2.set_xlabel("Prediction Error (mL)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Frequency", fontsize=12, fontweight='bold')
    ax2.set_title(f"{model_name} — Error Distribution", fontsize=13, fontweight="bold", pad=15)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    
    return fig1, fig2

def refresh_metrics(model_key, model_name):
    """Load and display metrics for a specific model"""
    metrics_data = load_metrics()
    
    if metrics_data is None:
        msg = (
            "<div style='background:#7f1d1d;padding:20px;border-radius:8px;color:#fecaca'>"
            "<h3 style='margin:0 0 8px 0'>⚠️ No Metrics Found</h3>"
            "<p style='margin:0'>File <code>complete_metrics.json</code> not found.</p>"
            "<p style='margin:8px 0 0 0'>Please run the diagnostic training script first.</p>"
            "</div>"
        )
        return msg, None, None
    
    m = get_model_metrics(metrics_data, model_key)
    
    if not m:
        msg = (
            f"<div style='background:#7f1d1d;padding:20px;border-radius:8px;color:#fecaca'>"
            f"<h3 style='margin:0 0 8px 0'>⚠️ No Metrics for {model_name}</h3>"
            f"<p style='margin:0'>Model '{model_key}' not found in metrics file.</p>"
            f"</div>"
        )
        return msg, None, None
    
    return metric_cards_html(m), *make_plots(m, model_name)

def get_comparison_data():
    """Get comparison data for both models"""
    metrics_data = load_metrics()
    
    if not metrics_data:
        return None
    
    f_metrics = get_model_metrics(metrics_data, "Florence-2")
    q_metrics = get_model_metrics(metrics_data, "Qwen2.5-VL")
    
    training_metrics = metrics_data.get("training_metrics", {})
    
    return {
        "florence": {
            "eval": f_metrics,
            "train": training_metrics.get("Florence-2", {})
        },
        "qwen": {
            "eval": q_metrics,
            "train": training_metrics.get("Qwen2.5-VL", {})
        }
    }

def create_comparison_html():
    """Create side-by-side comparison table"""
    data = get_comparison_data()
    
    if not data:
        return "<p style='color:#f87171'>No comparison data available. Run training first.</p>"
    
    f_eval = data["florence"]["eval"]
    q_eval = data["qwen"]["eval"]
    f_train = data["florence"]["train"]
    q_train = data["qwen"]["train"]
    
    def metric_row(label, f_val, q_val, unit="", lower_better=True):
        # Determine winner
        if f_val != "—" and q_val != "—":
            try:
                f_num = float(f_val)
                q_num = float(q_val)
                if lower_better:
                    f_best = f_num < q_num
                else:
                    f_best = f_num > q_num
            except:
                f_best = False
        else:
            f_best = False
        
        q_best = not f_best if (f_val != "—" and q_val != "—") else False
        
        f_style = "color:#10b981;font-weight:bold" if f_best else "color:#f8fafc"
        q_style = "color:#10b981;font-weight:bold" if q_best else "color:#f8fafc"
        
        return f"""
        <tr>
            <td style='padding:12px;color:#94a3b8;border-bottom:1px solid #334155'>{label}</td>
            <td style='padding:12px;text-align:center;border-bottom:1px solid #334155;{f_style}'>{f_val}{unit}</td>
            <td style='padding:12px;text-align:center;border-bottom:1px solid #334155;{q_style}'>{q_val}{unit}</td>
        </tr>
        """
    
    table = f"""
    <div style='background:#1e293b;padding:24px;border-radius:12px;margin:20px 0'>
        <h2 style='color:#f1f5f9;margin:0 0 20px 0;text-align:center'>Model Comparison</h2>
        <table style='width:100%;border-collapse:collapse;color:#e2e8f0'>
            <thead>
                <tr style='background:#0f172a'>
                    <th style='padding:14px;text-align:left;color:#94a3b8;border-bottom:2px solid #3b82f6'>Metric</th>
                    <th style='padding:14px;text-align:center;color:#94a3b8;border-bottom:2px solid #3b82f6'>Florence-2</th>
                    <th style='padding:14px;text-align:center;color:#94a3b8;border-bottom:2px solid #3b82f6'>Qwen2.5-VL</th>
                </tr>
            </thead>
            <tbody>
                <tr><td colspan='3' style='padding:8px;background:#0f172a;color:#64748b;font-weight:bold'>Evaluation Metrics</td></tr>
                {metric_row("MAE (mL)", 
                    f"{f_eval.get('mae', 0):.2f}" if f_eval.get('mae') else "—",
                    f"{q_eval.get('mae', 0):.2f}" if q_eval.get('mae') else "—",
                    lower_better=True)}
                {metric_row("RMSE (mL)", 
                    f"{f_eval.get('rmse', 0):.2f}" if f_eval.get('rmse') else "—",
                    f"{q_eval.get('rmse', 0):.2f}" if q_eval.get('rmse') else "—",
                    lower_better=True)}
                {metric_row("R²", 
                    f"{f_eval.get('r2', 0):.4f}" if f_eval.get('r2') is not None else "—",
                    f"{q_eval.get('r2', 0):.4f}" if q_eval.get('r2') is not None else "—",
                    lower_better=False)}
                {metric_row("MAPE (%)", 
                    f"{f_eval.get('mape', 0):.1f}" if f_eval.get('mape') else "—",
                    f"{q_eval.get('mape', 0):.1f}" if q_eval.get('mape') else "—",
                    lower_better=True)}
                {metric_row("Samples Used", 
                    str(f_eval.get('num_used_samples', 0)),
                    str(q_eval.get('num_used_samples', 0)),
                    lower_better=False)}
                {metric_row("Extraction Success", 
                    f"{f_eval.get('extraction_success_rate', 0)*100:.1f}" if f_eval.get('extraction_success_rate') else "—",
                    f"{q_eval.get('extraction_success_rate', 0)*100:.1f}" if q_eval.get('extraction_success_rate') else "—",
                    "%",
                    lower_better=False)}
                <tr><td colspan='3' style='padding:8px;background:#0f172a;color:#64748b;font-weight:bold'>Training Info</td></tr>
                {metric_row("Training Time", 
                    f"{f_train.get('training_time_minutes', 0):.1f}" if f_train.get('training_time_minutes') else "—",
                    f"{q_train.get('training_time_minutes', 0):.1f}" if q_train.get('training_time_minutes') else "—",
                    " min",
                    lower_better=True)}
                {metric_row("Trainable Params", 
                    f"{f_train.get('trainable_parameters', 0):,}" if f_train.get('trainable_parameters') else "—",
                    f"{q_train.get('trainable_parameters', 0):,}" if q_train.get('trainable_parameters') else "—",
                    lower_better=True)}
            </tbody>
        </table>
        <p style='color:#64748b;font-size:12px;margin:16px 0 0;text-align:center'>
            ✅ Green indicates better performance
        </p>
    </div>
    """
    
    return table

# ── UI ────────────────────────────────────────────────────────────────────────

css = """
body        { background:#0f172a; color:#e2e8f0; font-family:'Inter',sans-serif; }
.gr-block   { background:#1e293b !important; border-color:#334155 !important; }
.gr-button  { background:#3b82f6 !important; color:white !important; border:none !important; 
              font-weight:600 !important; transition:all 0.2s !important; }
.gr-button:hover { background:#2563eb !important; transform:translateY(-1px) !important; }
.gr-box { border-radius:12px !important; }
"""

HEADER = """
<div style='text-align:center;padding:32px 0 16px;background:linear-gradient(135deg, #1e293b 0%, #0f172a 100%)'>
  <h1 style='color:#f1f5f9;font-size:2.5rem;font-weight:800;margin:0;
             background:linear-gradient(135deg, #3b82f6, #8b5cf6);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
    🧪 Beaker Volume Estimator
  </h1>
  <p style='color:#94a3b8;margin:12px 0 0;font-size:15px;font-weight:500'>
    Fine-tuned Florence-2 &amp; Qwen2.5-VL for Liquid Volume Measurement
  </p>
  <p style='color:#64748b;margin:6px 0 0;font-size:13px'>
    Diagnostic Training Version with Enhanced Prompts &amp; Integer Answers
  </p>
</div>
"""

with gr.Blocks(css=css, title="Beaker Volume Estimator", theme=gr.themes.Soft()) as demo:
    gr.HTML(HEADER)

    with gr.Tabs():

        # ── Tab 1: Live inference ──────────────────────────────────────────
        with gr.Tab("🔬 Live Inference"):
            gr.Markdown("""
            ### Upload a beaker image to get volume predictions from both models
            Both models use the improved prompts from diagnostic training.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    img_in  = gr.Image(type="pil", label="📷 Beaker Image", height=400)
                    pred_btn = gr.Button("🚀 Predict Volume", variant="primary", size="lg")
                    gr.Markdown("""
                    <div style='background:#0f172a;padding:12px;border-radius:8px;margin-top:12px'>
                    <p style='margin:0;color:#64748b;font-size:12px'>
                    💡 <b>Tip:</b> For best results, use clear images with visible liquid levels
                    </p>
                    </div>
                    """)
                
                with gr.Column(scale=2):
                    with gr.Group():
                        gr.Markdown("#### 🎯 Florence-2 Results")
                        f_raw = gr.Textbox(label="Raw model output", lines=2, interactive=False)
                        f_vol = gr.Textbox(label="📊 Estimated volume", interactive=False)
                    
                    with gr.Group():
                        gr.Markdown("#### 🎯 Qwen2.5-VL Results")
                        q_raw = gr.Textbox(label="Raw model output", lines=2, interactive=False)
                        q_vol = gr.Textbox(label="📊 Estimated volume", interactive=False)
            
            pred_btn.click(
                run_inference, 
                inputs=[img_in],
                outputs=[f_raw, f_vol, q_raw, q_vol]
            )

        # ── Tab 2: Florence-2 metrics ──────────────────────────────────────
        with gr.Tab("📊 Florence-2 Metrics"):
            gr.Markdown("## Florence-2 — Test Set Evaluation")
            gr.Markdown("Evaluation metrics from the diagnostic training with improved prompts and integer answers.")
            
            f2_btn  = gr.Button("🔄 Load / Refresh Metrics", variant="secondary")
            f2_html = gr.HTML("<p style='color:#64748b'>Click 'Load / Refresh' to display results.</p>")
            
            with gr.Row():
                f2_scatter = gr.Plot(label="Predicted vs Ground Truth")
                f2_error   = gr.Plot(label="Error Distribution")
            
            f2_btn.click(
                fn=lambda: refresh_metrics("Florence-2", "Florence-2"),
                outputs=[f2_html, f2_scatter, f2_error]
            )

        # ── Tab 3: Qwen metrics ────────────────────────────────────────────
        with gr.Tab("📊 Qwen2.5-VL Metrics"):
            gr.Markdown("## Qwen2.5-VL — Test Set Evaluation")
            gr.Markdown("Evaluation metrics from the diagnostic training with improved prompts and integer answers.")
            
            q2_btn  = gr.Button("🔄 Load / Refresh Metrics", variant="secondary")
            q2_html = gr.HTML("<p style='color:#64748b'>Click 'Load / Refresh' to display results.</p>")
            
            with gr.Row():
                q2_scatter = gr.Plot(label="Predicted vs Ground Truth")
                q2_error   = gr.Plot(label="Error Distribution")
            
            q2_btn.click(
                fn=lambda: refresh_metrics("Qwen2.5-VL", "Qwen2.5-VL"),
                outputs=[q2_html, q2_scatter, q2_error]
            )

        # ── Tab 4: Model Comparison ────────────────────────────────────────
        with gr.Tab("⚖️ Model Comparison"):
            gr.Markdown("## Side-by-Side Model Comparison")
            gr.Markdown("Compare performance metrics and training details between both models.")
            
            comp_btn = gr.Button("🔄 Load Comparison", variant="secondary")
            comp_html = gr.HTML("<p style='color:#64748b'>Click 'Load Comparison' to display results.</p>")
            
            comp_btn.click(
                fn=create_comparison_html,
                outputs=[comp_html]
            )

        # ── Tab 5: About ───────────────────────────────────────────────────
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
## About This Demo

This demo showcases two vision-language models fine-tuned to estimate liquid volume from beaker photographs.

### 🎯 Models

| Model | Base Model | Parameters | Trainable |
|-------|-----------|-----------|-----------|
| **Florence-2** | microsoft/Florence-2-base | 232M | ~0.5% via LoRA (r=32) |
| **Qwen2.5-VL** | Qwen/Qwen2-VL-2B-Instruct | 2B | ~0.3% via LoRA (r=32) |

### 📚 Training Details

- **Dataset**: [yusufbukarmaina/Beakers1](https://huggingface.co/datasets/yusufbukarmaina/Beakers1)
- **Split**: 700 train / 150 val / 150 test
- **LoRA Configuration**: 
  - Rank: 32, Alpha: 64
  - Target modules: q_proj, v_proj, k_proj, o_proj
  - Dropout: 0.15
- **Training**: 10 epochs with cosine LR schedule
- **Test Set**: Saved before training to `./trained_models/test_set/`

### 🔧 Key Improvements (Diagnostic Version)

#### Better Prompts
```
Florence: "<VQA>What is the volume of liquid in this beaker measured in milliliters (mL)?"
Qwen: "What is the volume of liquid in this beaker measured in milliliters (mL)?"
```

#### Consistent Answer Format
- Training: Integer answers (e.g., "150" not "150.0")
- Inference: Constrained generation (max_new_tokens=10, deterministic)

#### Enhanced Training Configuration
- **Florence LR**: 1e-4 (higher to overcome pre-training bias)
- **Qwen LR**: 3e-5 (stable)
- **Weight Decay**: 0.05 (stronger regularization)
- **Warmup**: 10% of training steps
- **Gradient Clipping**: max_norm=1.0

#### Improved Volume Extraction
- 6 different regex patterns
- Validates extracted values (0-250 mL range)
- Multiple extraction attempts with text cleaning

### 📈 Metrics Explained

- **MAE** (Mean Absolute Error): Average prediction error in mL. Lower is better.
- **RMSE** (Root Mean Squared Error): Penalizes large errors more heavily. Lower is better.
- **R²** (Coefficient of Determination): Measures prediction quality. 1.0 = perfect, 0 = as good as mean, negative = worse than mean.
- **MAPE** (Mean Absolute Percentage Error): Percentage error. Lower is better.

### 🏗️ Architecture

Both models use **LoRA (Low-Rank Adaptation)** for efficient fine-tuning:
- Only ~0.3-0.5% of parameters are trained
- Full model loaded with PEFT adapter
- Inference on FP16 for memory efficiency

### 📁 Output Structure

```
trained_models/
├── florence2_final/          # Florence adapter + processor
├── qwen2_5vl_final/          # Qwen adapter + processor
├── test_set/                 # Saved test data
│   ├── test_data.pkl
│   ├── metadata.json
│   ├── sample_info.json
│   └── distribution_stats.json
├── complete_metrics.json     # All training & eval metrics
├── gpu_usage_history.csv     # GPU tracking
├── summary_table.csv         # Results summary
└── *_raw_responses.json      # Debugging info
```

### 🚀 Running Locally

```bash
# Install dependencies
pip install torch transformers peft gradio pillow numpy matplotlib scikit-learn

# Launch demo
python gradio_demo.py --port 7860 --share
```

### 📝 Citation

If you use this work, please cite:
```bibtex
@misc{beaker_volume_estimator,
  title={Fine-tuning Vision-Language Models for Beaker Volume Estimation},
  author={Your Name},
  year={2026}
}
```

### 🔗 Links

- [Florence-2 Model](https://huggingface.co/microsoft/Florence-2-base)
- [Qwen2-VL Model](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- [Dataset](https://huggingface.co/datasets/yusufbukarmaina/Beakers1)
- [PEFT Library](https://github.com/huggingface/peft)

---

<div style='text-align:center;padding:20px;color:#64748b'>
Made with ❤️ using Gradio, Transformers, and PEFT
</div>
""")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Beaker Volume Estimator Demo")
    p.add_argument("--share", action="store_true", help="Create public share link")
    p.add_argument("--port", type=int, default=7860, help="Port number")
    a = p.parse_args()
    
    print("=" * 80)
    print("🧪 Beaker Volume Estimator - Gradio Demo")
    print("=" * 80)
    print(f"\nStarting server on port {a.port}...")
    if a.share:
        print("Creating public share link...")
    print("\n⚠️  Models will be loaded on first inference (may take ~30 seconds)")
    print("=" * 80)
    
    demo.launch(
        share=a.share, 
        server_port=a.port, 
        server_name="0.0.0.0",
        show_error=True
    )
