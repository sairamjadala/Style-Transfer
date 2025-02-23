# Style-Transfer
Style Transfer Using Stable Diffusion &amp; ComfyUI
# Upload image from local system
from google.colab import files
from PIL import Image
import matplotlib.pyplot as plt

# Upload lion image
uploaded = files.upload()

# Load and display uploaded lion image
for file_name in uploaded.keys():
    content_image = Image.open(file_name).convert("RGB")
    plt.imshow(content_image)
    plt.axis("off")
    plt.title("Uploaded Lion Image")
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

# Initialize pipeline
pipeline = load_pipeline()
print("✅ Stable Diffusion Pipeline Loaded Successfully.")

# Generate stylized image using Stable Diffusion
def generate_stylized_image(prompt, pipeline):
    image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5)["images"][0]
    return image

# Define the artistic style prompt (Van Gogh style)
prompt = "A majestic lion painted in the style of Van Gogh"

# Apply style transfer
stylized_image = generate_stylized_image(prompt, pipeline)

# Display the stylized lion image
plt.imshow(lion)
plt.axis("off")
plt.title("Lion(Van Gogh Style)")
plt.show()

# Save the stylized image
stylized_image.save("lion.png")
print("✅ Stylized lion image saved as lino.png")

