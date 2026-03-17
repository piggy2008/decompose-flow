import os.path

import model_loader
import pipeline
import pipeline_no_ed
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch
from util import load_part_of_model2, load_part_of_model
DEVICE = "cuda:0"

# ALLOW_CUDA = True
# ALLOW_MPS = False
#
# if torch.cuda.is_available() and ALLOW_CUDA:
#     DEVICE = "cuda:3"
# elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
#     DEVICE = "mps"
print(f"Using device: {DEVICE}")

save_root = '../experiments/checkpoints_260316_164131' # best model

# save_root = '../experiments/checkpoints_250722_171443'
# save_gates_root = '../experiments/checkpoints_250212_162152'
# gates_pretrained = os.path.join(save_root, 'control_gates_140.pth')
tokenizer = CLIPTokenizer("../data/vocab.json", merges_file="../data/merges.txt")
model_file = "../data/v1-5-pruned-emaonly.ckpt"

models = model_loader.preload_models_from_standard_weights(model_file, DEVICE, in_channels=3, out_channels=3, image_size=256)

diffusion = models['diffusion']

checkpoint = torch.load(os.path.join(save_root, 'sd_diffusion_160.pth'), map_location=DEVICE)
# checkpoint = torch.load(os.path.join(save_root, 'sd_diffusion_160.pth'), map_location=DEVICE) ## best model for now
# load_part_of_model(diffusion, checkpoint['model'], True)
#
diffusion.load_state_dict(checkpoint['model'], strict=True)
models['diffusion'] = diffusion
## TEXT TO IMAGE

# prompt = "A dog with sunglasses, wearing comfy hat, looking at camera, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
prompt = "remove the underwater color noise of the object in the underwater"
uncond_prompt = "the image exist haze, please remove haze and make it clearer"  # Also known as negative prompt
do_cfg = False
cfg_scale = 8  # min: 1, max: 14

## IMAGE TO IMAGE

# input_image = None
# Comment to disable image to image
# image_name = '851.jpg'
# image_path = '../../../data/UIE-dataset/UIEBD2/test/image/' + image_name
# image_root = '../../../data/LSUI/test_input'
image_root = '../../../data/UIE-dataset/UIEBD/test/input'
# image_root = '/media/ty/My Passport/iccv_data/U45/input'
# image_root = '/media/ty/My Passport/kyudai_data/EUVP2/test_samples/Inp'
# image_root = '/media/ty/My Passport/kyudai_data/UIEB/input_test_uw'
paths = [path for path in os.listdir(image_root)]

save_image_root = save_root + '/results/UIEBD'
# Higher values means more noise will be added to the input image, so the result will further from the input image.
# Lower values means less noise is added to the input image, so output will be closer to the input image.
strength = 0.9

## SAMPLER

sampler = "rf"
num_inference_steps = 10
seed = 66666

# output_image = pipeline.generate(
#     prompt=prompt,
#     uncond_prompt=uncond_prompt,
#     input_image=input_image,
#     strength=strength,
#     do_cfg=do_cfg,
#     cfg_scale=cfg_scale,
#     sampler_name=sampler,
#     n_inference_steps=num_inference_steps,
#     seed=seed,
#     models=models,
#     device=DEVICE,
#     idle_device="cpu",
#     tokenizer=tokenizer,
# )

# Combine the input image and the output image into a single image.
# save_path = os.path.join(save_root, 'results')
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
# Image.fromarray(output_image).save(os.path.join(save_path, image_name))

output_image = pipeline_no_ed.generate_all(
    prompt=prompt,
    uncond_prompt=uncond_prompt,
    input_image_root=image_root,
    image_path=paths,
    save_root=save_image_root,
    strength=strength,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    seed=seed,
    models=models,
    device=DEVICE,
    idle_device="cpu",
    tokenizer=tokenizer,
)