from transformers import AutoProcessor, AutoModelForVision2Seq, Idefics3ForConditionalGeneration
from PIL import Image, ImageOps
import torch
from peft import PeftModel
from huggingface_hub import snapshot_download

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_model_name = "HuggingFaceTB/SmolVLM-256M-Instruct"

model = Idefics3ForConditionalGeneration.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
).to(device)

processor = AutoProcessor.from_pretrained(base_model_name)

repo_local_path = snapshot_download(
    repo_id="Irina1402/smolvlm-painting-description"
)

model = PeftModel.from_pretrained(model, model_id=repo_local_path)
model.eval()



def process_chat(text: str = None, image: Image.Image = None):
    """Process the input and generate a response using SmolVLM."""

    image_data = None  # Initialize the image_data variable

    inputs = []
    if image:
        image_data = Image.open(image.file).convert("RGB")
        image_data = ImageOps.exif_transpose(image_data) 
        inputs.append({"type": "image"})

    if text:
        inputs.append({"type": "text", "text": text})

    message = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]

    prompt = processor.apply_chat_template(message, add_generation_prompt=True)

    print(f"Prepared prompt:\n{prompt}") 

    processed_inputs = processor(
        text=prompt,
        images=[image_data] if image_data else None,  
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(**processed_inputs, max_new_tokens=350,repetition_penalty=1.1)

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    assistant_text = generated_text.split("Assistant:", 1)[-1].strip()

    if "." in assistant_text:
        last_period_idx = assistant_text.rfind(".")
        assistant_text = assistant_text[:last_period_idx + 1].strip()

    return assistant_text
