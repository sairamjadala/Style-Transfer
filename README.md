# Style-Transfer
Style Transfer Using Stable Diffusion &amp; ComfyUI
from PIL import Image
import matplotlib.pyplot as plt

# Load the lion image
image_path = "lion.jpg "  # Path to the uploaded image
content_image = Image.open(image_path).convert("RGB")

# Display the content image
plt.imshow(content_image)
plt.axis("off")
plt.title("Loaded Lion Image")
plt.show()
from diffusers import StableDiffusionPipeline
import torch

# Load Stable Diffusion pipeline
def load_pipeline(model_id="CompVis/stable-diffusion-v1-4"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipeline.to(device)
    return pipeline

pipeline = load_pipeline()
print("Stable Diffusion Pipeline Loaded Successfully.")

# Generate stylized image
def generate_stylized_image(prompt, pipeline):
    image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5)["images"][0]
    return image

# Define the style prompt (Van Gogh style as example)
prompt = "A majestic lion painted in the style of Van Gogh"

# Apply style transfer
stylized_image = generate_stylized_image(prompt, pipeline)

# Display the stylized image
plt.imshow(stylized_image)
plt.axis("off")
plt.title("Stylized Lion Image")
plt.show()

# Save the stylized image
stylized_image.save("stylized_lion_image.png")
print("Stylized lion image saved as stylized_lion_image.png")

from google.colab import files

# Download the stylized lion image
files.download("stylized_lion_image.png")
