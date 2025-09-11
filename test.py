from diffusers import FluxPipeline, FluxInpaintPipeline
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Sam2Processor, Sam2Model
import numpy as np

# pipe = FluxPipeline.from_pretrained(
#     "black-forest-labs/FLUX.1-dev",  # or "black-forest-labs/FLUX.1-schnell" for faster generation
#     torch_dtype=torch.bfloat16
# ).to("cuda")

# prompt = "a knight in shining armor gallops on a brown horse, wearing a blue hat, holding a sword in his right hand. A princess sits beside him on the same horse, wearing a flowing gown and a jeweled crown"
# prompt = "a high quality photo of " + prompt
# image = pipe(prompt).images[0]

# image.save("knight_horse.png")

# Load an image from path
image_path = "knight_horse.png"
image = Image.open(image_path).convert("RGB")

model_id = "IDEA-Research/grounding-dino-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# VERY important: text queries need to be lowercased + end with a dot
text = "a knight riding a horse. a princess riding a horse."

inputs = processor(images=image, text=text, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

result = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    threshold=0.4,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)[0]

print(result)
# draw boxes and labels on the image

draw = ImageDraw.Draw(image)
font = ImageFont.load_default()
for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
    box = box.tolist()
    draw.rectangle(box, outline="red", width=3)
    draw.text((box[0], box[1]), f"{label}: {score:.2f}", fill="red", font=font)
image.save("knight_dino.png")

sam_model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to(device)
sam_processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")

sam_inputs = sam_processor(images=image, input_boxes=result["boxes"][None, :].cpu(), return_tensors="pt").to(device)

with torch.no_grad():
    sam_outputs = sam_model(**sam_inputs)

masks = sam_processor.post_process_masks(sam_outputs.pred_masks.cpu(), sam_inputs["original_sizes"])[0]

print(f"Generated {masks.shape[0]} masks with shape: {masks.shape}")

original_image = Image.open(image_path).convert("RGB")
original_np = np.array(original_image)

prompts = {
    "a knight": "a sitting knight in a shining armor with his arms in high gaurd stance",
    "a horse": "a galloping brown horse",
    "a blue hat": "a blue hat",
    "a sword": "a medieval sword",
    "a crown": "the Imperial State Crown of England.",
    "a princess": "a sitting princess, wearing a flowing gown"
}

inpaint_pipe = FluxInpaintPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
).to('cuda')

for i, (mask, score, label, box) in enumerate(zip(masks, result["scores"], result["labels"], result["boxes"])):
    print(f"\nProcessing {label} (detection score: {score:.3f})...")
    mask_np = mask.squeeze().numpy().astype(int)
    mask_np = mask_np.mean(axis=0) > 0.5  # Combine masks if multiple exist for the same object

    # print(f"Mask shape: {mask_np.shape}, True pixels: {np.sum(mask_np)}, False pixels: {np.sum(~mask_np)}")

    segmented_image = original_np.copy()
    segmented_image[~mask_np] = 0

    segmented_pil = Image.fromarray(segmented_image)
    segmented_pil.save(f"segmented_{label}_{i}_black_bg.png")
    # init_image = Image.fromarray(segmented_image)
    # mask_image = Image.fromarray(~mask_np.astype(np.uint8))
    # mask_image = mask_image.convert("L")

    # result_image = inpaint_pipe(
    #     prompt=prompts[label],
    #     image=init_image,
    #     mask_image=mask_image,
    #     num_inference_steps=40,
    #     # guidance_scale=7.5,
    # ).images[0]
    # result_image.save(f"inpainted_{label}_{i}.png")


