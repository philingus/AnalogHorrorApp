import gradio as gr
import cv2
import numpy as np
from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image, ImageDraw, ImageFont

# Load Stable Diffusion Inpainting Model
try:
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
except Exception as e:
    print(f"Error loading Stable Diffusion pipeline: {e}")

# Function to apply analog effects
def apply_analog_effects(image):
    print("Applying analog effects...")
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    image_color = cv2.merge([noisy_image, noisy_image, noisy_image])
    shift = 10
    image_color[:, :, 0] = np.roll(image_color[:, :, 0], shift, axis=1)
    return Image.fromarray(image_color)

# Function to detect and distort faces
def distort_faces(image):
    print("Detecting and distorting faces...")
    image_np = np.array(image)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = image_np[y:y+h, x:x+w]
        distorted_face = cv2.resize(face, (w // 2, h // 2))
        distorted_face = cv2.resize(distorted_face, (w, h))
        image_np[y:y+h, x:x+w] = distorted_face

    return Image.fromarray(image_np)

# Function to overlay cryptic text
def overlay_text(image, text="DO NOT TRUST IT"):
    print("Overlaying text...")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except OSError:
        font = ImageFont.load_default()
    draw.text((10, 10), text, fill="red", font=font)
    return image

# Main pipeline function
def generate_analog_horror(image):
    try:
        analog_image = apply_analog_effects(image)
        distorted_image = distort_faces(analog_image)
        final_image = overlay_text(distorted_image)
        return final_image
    except Exception as e:
        print(f"Error in processing pipeline: {e}")
        return None

# Gradio Interface
demo = gr.Interface(
    fn=generate_analog_horror,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Image(type="pil", label="Analog Horror Image"),
    title="Analog Horror Image Generator",
    description="Upload an image and transform it into an eerie analog horror-style output using AI."
)

demo.launch()

