import re
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

model_id = "./trocr-large-printed"

device = torch.device("cpu")

print("Loading model...")

image_processor = AutoImageProcessor.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = VisionEncoderDecoderModel.from_pretrained(model_id, torch_dtype=torch.float32, low_cpu_mem_usage=False).to(device)
print("Model loaded.")

model.to(device)
model.eval()

if device.type == "cuda":
    model.half()

image_path = "01001.jpg"
image = Image.open(image_path).convert("RGB")

pixel_values = image_processor(images=image, return_tensors="pt").pixel_values.to(device)

if device.type == "cuda" and next(model.parameters()).dtype == torch.float16:
    pixel_values = pixel_values.half()


with torch.inference_mode():
    generated_ids = model.generate(pixel_values, num_beams=4, max_length=16, early_stopping=True)

text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
text = text.strip()

print("Raw OCR output:", text)

digits = "".join(re.findall(r"\d", text))
print("Digits only:", digits)
