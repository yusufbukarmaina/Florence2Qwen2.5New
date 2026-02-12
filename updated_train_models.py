"""
Complete Training Pipeline with Comprehensive Metrics Tracking (FULL + FIXED)
For Thesis Documentation - Tracks training time, GPU usage, and all evaluation metrics

Fixes:
- Correct Florence evaluation: loads trained adapter checkpoint
- Correct Florence labels: prompt + answer causal LM alignment, mask prompt tokens
- Robust volume extraction: reject impossible values, avoid grabbing random numbers
- Qwen evaluation: decode ONLY newly generated tokens
- dtype-safe input casting for fp16 models
- safer LR for LoRA fine-tuning
"""

import os
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    Qwen2VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from PIL import Image
import re
import warnings
import gc
import time
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# =============================================================================
# CONFIG
# =============================================================================

class Config:
    HF_DATASET_NAME = "yusufbukarmaina/Beakers1"
    STREAMING = True

    TRAIN_SAMPLES = 700
    VAL_SAMPLES = 150
    TEST_SAMPLES = 150

    FLORENCE_MODEL = "microsoft/Florence-2-base"
    QWEN_MODEL = "Qwen/Qwen2-VL-2B-Instruct"

    MAX_IMAGE_SIZE = 512

    FLORENCE_LR = 2e-5  # Was 1e-4 (5x lower)
    QWEN_LR = 1e-5      # Was 3e-5 (3x lower)
    # LoRA
    LORA_R = 16         # Was 32 (half the size)
    LORA_ALPHA = 32     # Was 64
    LORA_DROPOUT = 0.1  # Was 0.15

    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION = 16

    # ✅ safer LR
    

    NUM_EPOCHS = 15
    WARMUP_STEPS = 50
    MAX_LENGTH = 256

    FP16 = True
    GRADIENT_CHECKPOINTING = True

    OUTPUT_DIR = "./trained_models"
    SAVE_STEPS = 200
    EVAL_STEPS = 200
    LOGGING_STEPS = 25

    # volume sanity range
    MAX_VOLUME_ML = 250


# =============================================================================
# HELPERS
# =============================================================================

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def resize_image(image, max_size=512):
    """Resize image to reduce memory"""
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB")

    w, h = image.size
    if w > max_size or h > max_size:
        if w >= h:
            new_w = max_size
            new_h = int(h * (max_size / w))
        else:
            new_h = max_size
            new_w = int(w * (max_size / h))
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return image


def move_inputs_to_device_dtype(inputs, model):
    """
    Move tensors to model device.
    Cast floating tensors to model dtype (important for fp16 models).
    """
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


# =============================================================================
# METRICS TRACKER
# =============================================================================

class MetricsTracker:
    def __init__(self):
        self.metrics = {
            "experiment_info": {},
            "dataset_info": {},
            "training_metrics": {},
            "gpu_metrics": {},
            "evaluation_metrics": {},
            "timing_metrics": {},
        }
        self.gpu_history = []
        self.start_time = None

    def start_experiment(self, config):
        self.start_time = time.time()
        self.metrics["experiment_info"] = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "gpu_total_memory_gb": (torch.cuda.get_device_properties(0).total_memory / 1e9) if torch.cuda.is_available() else 0,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        }

    def record_dataset_info(self, train_size, val_size, test_size):
        total = train_size + val_size + test_size
        self.metrics["dataset_info"] = {
            "total_samples": total,
            "train_samples": train_size,
            "val_samples": val_size,
            "test_samples": test_size,
            "train_ratio": train_size / total if total else 0,
            "val_ratio": val_size / total if total else 0,
            "test_ratio": test_size / total if total else 0,
        }

    def record_gpu_usage(self, phase="training"):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            max_alloc = torch.cuda.max_memory_allocated() / 1e9
            self.gpu_history.append(
                {
                    "timestamp": time.time() - self.start_time,
                    "phase": phase,
                    "allocated_gb": allocated,
                    "reserved_gb": reserved,
                    "max_allocated_gb": max_alloc,
                }
            )
            return allocated, reserved, max_alloc
        return 0, 0, 0

    def record_training_metrics(self, model_name, train_time, num_epochs, num_parameters, trainable_parameters):
        if model_name not in self.metrics["training_metrics"]:
            self.metrics["training_metrics"][model_name] = {}

        self.metrics["training_metrics"][model_name].update(
            {
                "training_time_seconds": train_time,
                "training_time_minutes": train_time / 60,
                "training_time_hours": train_time / 3600,
                "num_epochs": num_epochs,
                "total_parameters": int(num_parameters),
                "trainable_parameters": int(trainable_parameters),
                "trainable_percentage": (trainable_parameters / num_parameters) * 100 if num_parameters else 0,
            }
        )

    def record_evaluation_metrics(self, model_name, metrics_dict):
        self.metrics["evaluation_metrics"][model_name] = metrics_dict

    def generate_summary_table(self):
        summary = {
            "Model": [],
            "Train Time (min)": [],
            "Parameters": [],
            "Trainable %": [],
            "MAE (mL)": [],
            "RMSE (mL)": [],
            "R²": [],
            "MAPE (%)": [],
            "Peak GPU (GB)": [],
        }

        for model_name in self.metrics["training_metrics"].keys():
            tm = self.metrics["training_metrics"][model_name]
            summary["Model"].append(model_name)
            summary["Train Time (min)"].append(f"{tm['training_time_minutes']:.2f}")
            summary["Parameters"].append(f"{tm['total_parameters']:,}")
            summary["Trainable %"].append(f"{tm['trainable_percentage']:.2f}")

            if model_name in self.metrics["evaluation_metrics"]:
                em = self.metrics["evaluation_metrics"][model_name]
                summary["MAE (mL)"].append(f"{em['mae']:.2f}")
                summary["RMSE (mL)"].append(f"{em['rmse']:.2f}")
                summary["R²"].append(f"{em['r2']:.4f}")
                summary["MAPE (%)"].append(f"{em.get('mape', 0):.2f}")
            else:
                summary["MAE (mL)"].append("N/A")
                summary["RMSE (mL)"].append("N/A")
                summary["R²"].append("N/A")
                summary["MAPE (%)"].append("N/A")

            model_gpu = [g for g in self.gpu_history if model_name.lower() in g["phase"].lower()]
            peak_gpu = max([g["max_allocated_gb"] for g in model_gpu]) if model_gpu else 0
            summary["Peak GPU (GB)"].append(f"{peak_gpu:.2f}")

        return pd.DataFrame(summary)

    def save_all_metrics(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        json_path = os.path.join(output_dir, "complete_metrics.json")
        with open(json_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        print(f"✅ Complete metrics saved to: {json_path}")

        if self.gpu_history:
            gpu_df = pd.DataFrame(self.gpu_history)
            gpu_csv = os.path.join(output_dir, "gpu_usage_history.csv")
            gpu_df.to_csv(gpu_csv, index=False)
            print(f"✅ GPU usage history saved to: {gpu_csv}")

        summary_df = self.generate_summary_table()
        summary_csv = os.path.join(output_dir, "summary_table.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"✅ Summary table saved to: {summary_csv}")

        latex_path = os.path.join(output_dir, "summary_table.tex")
        with open(latex_path, "w") as f:
            f.write(summary_df.to_latex(index=False))
        print(f"✅ LaTeX table saved to: {latex_path}")

        return summary_df


# =============================================================================
# TRAINING CALLBACK
# =============================================================================

class MetricsCallback(TrainerCallback):
    def __init__(self, tracker, model_name):
        self.tracker = tracker
        self.model_name = model_name
        self.epoch_start_time = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        self.tracker.record_gpu_usage(phase=f"{self.model_name}_epoch_{state.epoch}")

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.epoch_start_time
        print(f"  Epoch {state.epoch} completed in {epoch_time/60:.2f} minutes")
        self.tracker.record_gpu_usage(phase=f"{self.model_name}_epoch_{state.epoch}_end")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0:
            self.tracker.record_gpu_usage(phase=f"{self.model_name}_step_{state.global_step}")


# =============================================================================
# DATA PROCESSOR
# =============================================================================

class DatasetProcessor:
    def __init__(self, config: Config):
        self.config = config

    def load_and_split_dataset(self):
        print("📥 Loading dataset...")
        ds = load_dataset(self.config.HF_DATASET_NAME, split="train", streaming=self.config.STREAMING)
        ds = ds.shuffle(seed=42, buffer_size=500)

        train_data, val_data, test_data = [], [], []

        for ex in ds:
            if "image" not in ex:
                continue
            if len(train_data) < self.config.TRAIN_SAMPLES:
                train_data.append(ex)
            elif len(val_data) < self.config.VAL_SAMPLES:
                val_data.append(ex)
            elif len(test_data) < self.config.TEST_SAMPLES:
                test_data.append(ex)
            else:
                break

            if (len(train_data) + len(val_data) + len(test_data)) % 100 == 0:
                print(f"✓ Loaded {len(train_data) + len(val_data) + len(test_data)}")

        print(f"\n✅ Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        return train_data, val_data, test_data

    def extract_volume(self, text) -> float | None:
        """
        Robust extraction:
        - extracts a numeric token
        - rejects impossible values
        """
        if text is None:
            return None
        s = str(text)

        m = re.search(r"(-?\d+(?:\.\d+)?)", s)
        if not m:
            return None

        v = float(m.group(1))
        if v < 0 or v > (self.config.MAX_VOLUME_ML + 20):
            return None
        return v

    def get_gt_volume(self, ex) -> float | None:
        """
        Prefer volume_ml if present, else parse volume_label.
        """
        if "volume_ml" in ex and ex["volume_ml"] is not None:
            try:
                v = float(ex["volume_ml"])
                return v
            except Exception:
                pass

        v = self.extract_volume(ex.get("volume_label", None))
        return v


# =============================================================================
# FLORENCE COLLATOR
# =============================================================================

def florence_collate_fn(features):
    pixel_values = torch.stack([f["pixel_values"] for f in features])

    input_ids = torch.nn.utils.rnn.pad_sequence(
        [f["input_ids"] for f in features], batch_first=True, padding_value=0
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [f["attention_mask"] for f in features], batch_first=True, padding_value=0
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [f["labels"] for f in features], batch_first=True, padding_value=-100
    )
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# =============================================================================
# FLORENCE TRAINER
# =============================================================================

class FlorenceTrainer:
    def __init__(self, config: Config, tracker: MetricsTracker):
        self.config = config
        self.tracker = tracker
        self.model = None
        self.processor = None

    def setup_model(self):
        print("\n🤖 Loading Florence-2...")
        clear_memory()

        self.processor = AutoProcessor.from_pretrained(self.config.FLORENCE_MODEL, trust_remote_code=True)

        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.FLORENCE_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        if self.config.GRADIENT_CHECKPOINTING:
            base_model.gradient_checkpointing_enable()

        base_model = prepare_model_for_kbit_training(base_model)

        lora_cfg = LoraConfig(
            r=self.config.LORA_R,
            lora_alpha=self.config.LORA_ALPHA,
            target_modules=self.config.LORA_TARGET_MODULES,
            lora_dropout=self.config.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base_model, lora_cfg)
        self.model.print_trainable_parameters()

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total_params, trainable_params

    def train(self, train_data, val_data):
        print("\n🚀 Training Florence-2...")
        train_start = time.time()

        total_params, trainable_params = self.setup_model()
        self.tracker.record_gpu_usage("florence2_model_loaded")

        processor = self.processor
        cfg = self.config
        dp = DatasetProcessor(cfg)

        class FlorenceDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                ex = self.data[idx]
                img = resize_image(ex["image"], cfg.MAX_IMAGE_SIZE)

                prompt = "<VQA>What is the liquid volume in mL? Answer ONLY with a number between 0 and 250."
                gt = dp.get_gt_volume(ex)
                if gt is None:
                    gt = 0.0

                answer = f"{gt:.1f}".rstrip("0").rstrip(".")  # nice clean number string
                full_text = prompt + " " + answer

                inputs = processor(images=img, text=full_text, return_tensors="pt", padding=True, truncation=True)

                # Labels aligned with input_ids
                labels = inputs["input_ids"].clone()

                # Mask prompt tokens so loss only on answer
                prompt_only = processor(images=img, text=prompt, return_tensors="pt", padding=True, truncation=True)
                prompt_len = prompt_only["input_ids"].shape[1]
                labels[:, :prompt_len] = -100

                return {
                    "pixel_values": inputs["pixel_values"].squeeze(0),
                    "input_ids": inputs["input_ids"].squeeze(0),
                    "attention_mask": inputs["attention_mask"].squeeze(0),
                    "labels": labels.squeeze(0),
                }

        train_dataset = FlorenceDataset(train_data)
        eval_dataset = FlorenceDataset(val_data)

        args = TrainingArguments(
            output_dir=f"{cfg.OUTPUT_DIR}/florence2",
            num_train_epochs=cfg.NUM_EPOCHS,
            per_device_train_batch_size=cfg.BATCH_SIZE,
            gradient_accumulation_steps=cfg.GRADIENT_ACCUMULATION,
            learning_rate=cfg.LEARNING_RATE,
            warmup_steps=cfg.WARMUP_STEPS,
            logging_steps=cfg.LOGGING_STEPS,
            save_steps=cfg.SAVE_STEPS,
            eval_steps=cfg.EVAL_STEPS,
            eval_strategy="steps",
            save_strategy="steps",
            save_total_limit=2,
            fp16=cfg.FP16,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=florence_collate_fn,
            callbacks=[MetricsCallback(self.tracker, "Florence-2")],
        )

        trainer.train()

        train_time = time.time() - train_start
        self.tracker.record_training_metrics("Florence-2", train_time, cfg.NUM_EPOCHS, total_params, trainable_params)

        final_dir = f"{cfg.OUTPUT_DIR}/florence2_final"
        os.makedirs(final_dir, exist_ok=True)

        # Save adapter + processor
        self.model.save_pretrained(final_dir)
        self.processor.save_pretrained(final_dir)

        print(f"✅ Florence-2 trained in {train_time/60:.2f} minutes")
        return final_dir

    @staticmethod
    def load_trained(config: Config, ckpt_dir: str):
        """
        Load base + trained PEFT adapter for evaluation.
        """
        processor = AutoProcessor.from_pretrained(ckpt_dir, trust_remote_code=True)

        base = AutoModelForCausalLM.from_pretrained(
            config.FLORENCE_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(base, ckpt_dir)
        model.eval()
        return model, processor


# =============================================================================
# QWEN DATA COLLATOR
# =============================================================================

class QwenDataCollator:
    def __init__(self, processor, max_image_size=512):
        self.processor = processor
        self.max_image_size = max_image_size

    def __call__(self, features):
        batch_text = []
        batch_images = []

        for f in features:
            if "text" in f and "image" in f:
                batch_text.append(f["text"])
                batch_images.append(resize_image(f["image"], self.max_image_size))

        if batch_text and batch_images:
            inputs = self.processor(text=batch_text, images=batch_images, return_tensors="pt", padding=True)
            labels = inputs["input_ids"].clone()
            return {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "pixel_values": inputs.get("pixel_values"),
                "image_grid_thw": inputs.get("image_grid_thw"),
                "labels": labels,
            }

        # fallback (should not happen)
        return {
            "input_ids": torch.zeros((len(features), 10), dtype=torch.long),
            "attention_mask": torch.zeros((len(features), 10), dtype=torch.long),
            "labels": torch.full((len(features), 10), -100, dtype=torch.long),
        }


# =============================================================================
# QWEN TRAINER
# =============================================================================

class QwenTrainer:
    def __init__(self, config: Config, tracker: MetricsTracker):
        self.config = config
        self.tracker = tracker
        self.model = None
        self.processor = None

    def setup_model(self):
        print("\n🤖 Loading Qwen2.5-VL...")
        clear_memory()

        self.processor = AutoProcessor.from_pretrained(self.config.QWEN_MODEL, trust_remote_code=True)

        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.config.QWEN_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        if self.config.GRADIENT_CHECKPOINTING:
            base_model.gradient_checkpointing_enable()

        base_model = prepare_model_for_kbit_training(base_model)

        lora_cfg = LoraConfig(
            r=self.config.LORA_R,
            lora_alpha=self.config.LORA_ALPHA,
            target_modules=self.config.LORA_TARGET_MODULES,
            lora_dropout=self.config.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(base_model, lora_cfg)
        self.model.print_trainable_parameters()

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total_params, trainable_params

    def train(self, train_data, val_data):
        print("\n🚀 Training Qwen2.5-VL...")
        train_start = time.time()

        total_params, trainable_params = self.setup_model()
        self.tracker.record_gpu_usage("qwen2.5vl_model_loaded")

        cfg = self.config
        processor = self.processor
        dp = DatasetProcessor(cfg)

        class QwenDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                ex = self.data[idx]
                image = resize_image(ex["image"], cfg.MAX_IMAGE_SIZE)

                question = "What is the liquid volume in mL? Answer ONLY with a number between 0 and 250."
                gt = dp.get_gt_volume(ex)
                if gt is None:
                    gt = 0.0
                answer = f"{gt:.1f}".rstrip("0").rstrip(".")

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": question},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": str(answer)}],
                    },
                ]

                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                return {"text": text, "image": image}

        train_dataset = QwenDataset(train_data)
        eval_dataset = QwenDataset(val_data)

        args = TrainingArguments(
            output_dir=f"{cfg.OUTPUT_DIR}/qwen2_5vl",
            num_train_epochs=cfg.NUM_EPOCHS,
            per_device_train_batch_size=cfg.BATCH_SIZE,
            gradient_accumulation_steps=cfg.GRADIENT_ACCUMULATION,
            learning_rate=cfg.LEARNING_RATE,
            warmup_steps=cfg.WARMUP_STEPS,
            logging_steps=cfg.LOGGING_STEPS,
            save_steps=cfg.SAVE_STEPS,
            eval_steps=cfg.EVAL_STEPS,
            eval_strategy="steps",
            save_strategy="steps",
            save_total_limit=2,
            fp16=cfg.FP16,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="none",
        )

        collator = QwenDataCollator(processor, cfg.MAX_IMAGE_SIZE)

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            callbacks=[MetricsCallback(self.tracker, "Qwen2.5-VL")],
        )

        trainer.train()

        train_time = time.time() - train_start
        self.tracker.record_training_metrics("Qwen2.5-VL", train_time, cfg.NUM_EPOCHS, total_params, trainable_params)

        final_dir = f"{cfg.OUTPUT_DIR}/qwen2_5vl_final"
        os.makedirs(final_dir, exist_ok=True)

        self.model.save_pretrained(final_dir)
        self.processor.save_pretrained(final_dir)

        print(f"✅ Qwen2.5-VL trained in {train_time/60:.2f} minutes")
        return final_dir

    @staticmethod
    def load_trained(config: Config, ckpt_dir: str):
        processor = AutoProcessor.from_pretrained(ckpt_dir, trust_remote_code=True)

        base = Qwen2VLForConditionalGeneration.from_pretrained(
            config.QWEN_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        model = PeftModel.from_pretrained(base, ckpt_dir)
        model.eval()
        return model, processor


# Add this FIXED evaluator to replace the one in your code

class ModelEvaluator:
    def __init__(self, config: Config, tracker: MetricsTracker):
        self.config = config
        self.tracker = tracker
        self.dp = DatasetProcessor(config)
        self.extractor = ImprovedVolumeExtractor(
            min_vol=config.MIN_VOLUME_ML,
            max_vol=config.MAX_VOLUME_ML
        )

    def evaluate(self, model, processor, test_data, model_name):
        print(f"\n📊 Evaluating {model_name}...")
        eval_start = time.time()
        model.eval()

        preds = []
        gts = []
        failed_extractions = 0
        raw_responses = []

        with torch.no_grad():
            for i, ex in enumerate(test_data):
                try:
                    img = resize_image(ex["image"], self.config.MAX_IMAGE_SIZE)
                    gt = self.dp.get_gt_volume(ex)
                    if gt is None:
                        continue

                    if "florence" in model_name.lower():
                        prompt = "<VQA>What is the volume of liquid in this beaker measured in milliliters (mL)?"
                        inputs = processor(images=img, text=prompt, return_tensors="pt", padding=True)
                        inputs = move_inputs_to_device_dtype(inputs, model)

                        # Get input token length to decode only NEW tokens
                        input_len = inputs["input_ids"].shape[1]
                        
                        out = model.generate(
                            **inputs, 
                            max_new_tokens=10,
                            do_sample=False,
                            num_beams=1,
                        )
                        
                        # Decode FULL output (for debugging)
                        full_text = processor.batch_decode(out, skip_special_tokens=True)[0]
                        
                        # Decode ONLY new tokens (the answer)
                        new_tokens = out[0, input_len:]
                        answer_only = processor.decode(new_tokens, skip_special_tokens=True)
                        
                        # Try multiple extraction strategies
                        pred = None
                        
                        # Strategy 1: Extract from answer-only
                        pred = self.extractor.extract_from_response(answer_only, multiple_attempts=True)
                        
                        # Strategy 2: If that failed, try cleaning the full text
                        if pred is None and full_text:
                            # Remove prompt if present
                            cleaned = full_text.replace(prompt, "").strip()
                            # Remove task tags
                            cleaned = cleaned.replace("<VQA>", "").replace("</VQA>", "").strip()
                            # Try splitting on '?'
                            if '?' in cleaned:
                                cleaned = cleaned.split('?')[-1].strip()
                            pred = self.extractor.extract_from_response(cleaned, multiple_attempts=True)
                        
                        # Strategy 3: Extract from full text as last resort
                        if pred is None:
                            pred = self.extractor.extract_from_response(full_text, multiple_attempts=True)
                        
                        # Save detailed debug info
                        raw_responses.append({
                            "index": i,
                            "full_response": full_text,
                            "answer_only": answer_only,
                            "extracted": pred,
                            "gt": gt
                        })

                    else:  # Qwen
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
                        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        inputs = processor(text=[text_prompt], images=[img], return_tensors="pt", padding=True)
                        
                        device = next(model.parameters()).device
                        for k, v in inputs.items():
                            if isinstance(v, torch.Tensor):
                                inputs[k] = v.to(device)

                        out = model.generate(**inputs, max_new_tokens=10, do_sample=False, num_beams=1)
                        gen_ids = out[0, inputs["input_ids"].shape[1]:]
                        text = processor.decode(gen_ids, skip_special_tokens=True)
                        
                        pred = self.extractor.extract_from_response(text, multiple_attempts=True)
                        
                        raw_responses.append({
                            "index": i, 
                            "response": text, 
                            "extracted": pred, 
                            "gt": gt
                        })

                    if pred is None:
                        failed_extractions += 1
                        continue

                    preds.append(pred)
                    gts.append(gt)

                    if (i + 1) % 50 == 0:
                        print(f"  {i+1}/{len(test_data)}")

                except Exception as e:
                    print(f"  Error at sample {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        eval_time = time.time() - eval_start

        # Save raw responses with detailed info
        responses_path = os.path.join(self.config.OUTPUT_DIR, f"{model_name}_raw_responses.json")
        with open(responses_path, 'w') as f:
            json.dump(raw_responses, f, indent=2)
        print(f"💾 Saved raw responses to: {responses_path}")

        # Print first few responses for debugging
        if "florence" in model_name.lower() and raw_responses:
            print(f"\n🔍 Sample Florence outputs:")
            for resp in raw_responses[:3]:
                print(f"  GT={resp['gt']}")
                print(f"  Full: '{resp['full_response'][:80]}...'")
                print(f"  Answer only: '{resp['answer_only']}'")
                print(f"  Extracted: {resp['extracted']}")
                print()

        if len(preds) < 5:
            metrics = {
                "mae": float("nan"),
                "rmse": float("nan"),
                "r2": float("nan"),
                "mape": float("nan"),
                "eval_time_seconds": eval_time,
                "num_used_samples": len(preds),
                "failed_extractions": failed_extractions,
                "predictions": preds,
                "ground_truth": gts,
            }
            self.tracker.record_evaluation_metrics(model_name, metrics)
            print(f"⚠️ {model_name}: Not enough valid predictions (used {len(preds)} samples).")
            return metrics

        p = np.array(preds, dtype=np.float32)
        g = np.array(gts, dtype=np.float32)

        mae = mean_absolute_error(g, p)
        rmse = np.sqrt(mean_squared_error(g, p))
        r2 = r2_score(g, p)

        mask = g >= 5
        mape = mean_absolute_percentage_error(g[mask], p[mask]) * 100 if mask.any() else 0.0

        metrics = {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "mape": float(mape),
            "eval_time_seconds": float(eval_time),
            "num_used_samples": int(len(p)),
            "failed_extractions": int(failed_extractions),
            "extraction_success_rate": float(len(p) / (len(p) + failed_extractions)) if (len(p) + failed_extractions) > 0 else 0,
            "predictions": p.tolist(),
            "ground_truth": g.tolist(),
        }

        self.tracker.record_evaluation_metrics(model_name, metrics)
        print(f"📈 {model_name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}, MAPE={mape:.2f}% (used={len(p)})")
        return metrics

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("🚀 Complete Training Pipeline with Comprehensive Metrics (FULL + FIXED)")
    print("=" * 80)

    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    tracker = MetricsTracker()
    tracker.start_experiment(config)
    tracker.record_gpu_usage("experiment_start")

    # Load and split
    dp = DatasetProcessor(config)
    train_data, val_data, test_data = dp.load_and_split_dataset()
    tracker.record_dataset_info(len(train_data), len(val_data), len(test_data))

    # Train Florence
    f_trainer = FlorenceTrainer(config, tracker)
    f_ckpt = f_trainer.train(train_data, val_data)
    del f_trainer
    clear_memory()

    # Train Qwen
    q_trainer = QwenTrainer(config, tracker)
    q_ckpt = q_trainer.train(train_data, val_data)
    del q_trainer
    clear_memory()

    # Evaluate (load trained checkpoints properly)
    evaluator = ModelEvaluator(config, tracker)

    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)

    f_model, f_proc = FlorenceTrainer.load_trained(config, f_ckpt)
    _ = evaluator.evaluate(f_model, f_proc, test_data, "Florence-2")
    del f_model, f_proc
    clear_memory()

    q_model, q_proc = QwenTrainer.load_trained(config, q_ckpt)
    _ = evaluator.evaluate(q_model, q_proc, test_data, "Qwen2.5-VL")
    del q_model, q_proc
    clear_memory()

    # Save metrics
    print("\n" + "=" * 80)
    print("SAVING COMPREHENSIVE METRICS")
    print("=" * 80)
    summary_df = tracker.save_all_metrics(config.OUTPUT_DIR)

    print("\n" + "=" * 80)
    print("SUMMARY TABLE FOR THESIS")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("🎉 COMPLETE!")
    print("=" * 80)
    print(f"\nAll metrics saved to: {config.OUTPUT_DIR}/")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


