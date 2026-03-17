from clip import CLIP
from clip2 import CLIPTextEncoder
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion
from DiT import DiT
from new_clip import LMHead, LanguageModel
import torch

import model_converter
from sd.USUIR.net import net

def preload_models_from_standard_weights(ckpt_path, device, in_channels=6, out_channels=3, number_decompose=3, image_size=512):
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    # encoder = VAE_Encoder().to(device)
    # encoder.load_state_dict(state_dict['encoder'], strict=True)
    #
    # decoder = VAE_Decoder().to(device)
    # decoder.load_state_dict(state_dict['decoder'], strict=True)

    # diffusion = Diffusion().to(device)
    diffusion = DiT(depth=8, in_channels=in_channels, out_channels=out_channels,
                    hidden_size=384, patch_size=4, num_heads=6, input_size=image_size, number_decompose=number_decompose)
    # diffusion.load_state_dict(state_dict['diffusion'], strict=True)
    usuir = net()
    usuir.load_state_dict(torch.load('../data/UIEBD_final.pth', map_location=lambda storage, loc: storage))
    # ECCV24 clip encoding
    # lh_head_pretrained = 'config/lm_instructir-7d.pt'
    # language_model = LanguageModel(model='TaylorAI/bge-micro-v2')
    #
    # lm_head = LMHead(embedding_dim=384, hidden_dim=256, num_classes=7)
    # lm_head.load_state_dict(torch.load(lh_head_pretrained), strict=True)
    # clip = language_model
    # clip_head = lm_head

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)
    # clip = CLIPTextEncoder()

    return {
        'clip': clip,
        # 'clip_head': clip_head,
        # 'encoder': encoder,
        # 'decoder': decoder,
        'diffusion': diffusion,
        'usuir': usuir,
    }

if __name__ == '__main__':
    preload_models_from_standard_weights('../data/v1-5-pruned-emaonly.ckpt', 'cpu')