from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image, ImageOps
from classification import classify_image
from smolVLMchat import process_chat
from io import BytesIO

app = FastAPI()

@app.post("/classify/")
async def classify(
    file: UploadFile = File(...),
    model_name: str = Form(...) 
):
    """Endpoint to classify an uploaded image with a chosen model."""
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    image = ImageOps.exif_transpose(image)

    result = classify_image(image, model_name=model_name)

    return result


@app.post("/chat/")
async def chat(text: str = Form(None), image: UploadFile = File(None)):
    """Endpoint to chat with SmolVLM using text and image."""
    response = process_chat(text=text, image=image)
    return {"response": response}
