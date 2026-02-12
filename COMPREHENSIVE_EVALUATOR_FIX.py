"""
COMPREHENSIVE FIX - Handles ALL Florence-2 output formats
Includes detailed logging to diagnose output format
"""

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
