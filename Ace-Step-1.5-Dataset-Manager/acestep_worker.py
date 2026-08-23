#!/usr/bin/env python3
import sys
import json
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_ID = "ACE-Step/acestep-captioner"

def generate_caption(audio_path, trigger_tag="Doorsish"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Load processor and 11B multimodal captioner
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=dtype, 
        device_map="auto" if device == "cuda" else None
    )

    # Process raw audio input
    inputs = processor(audios=audio_path, return_tensors="pt").to(device, dtype)
    
    # Generate high-accuracy music description
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256)
        caption_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Prepend custom LoRA trigger tag cleanly
    final_caption = f"{trigger_tag}, {caption_text.strip()}"
    return final_caption

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: acestep_worker.py <audio_path> [trigger_tag]"}))
        sys.exit(1)
        
    audio = sys.argv[1]
    trigger = sys.argv[2] if len(sys.argv) > 2 else ""
    caption = generate_caption(audio, trigger)
    print(json.dumps({"audio": audio, "caption": caption}))
