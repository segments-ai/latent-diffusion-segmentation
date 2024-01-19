"""
Author: Wouter Van Gansbeke

File with descriptor models for latent diffusion training
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch
import torch.nn as nn
from typing import Optional
from transformers import CLIPVisionModel, CLIPVisionModelWithProjection, CLIPTokenizer, CLIPTextModel
from functools import partial


class MyCLIPVisionModel(CLIPVisionModel):
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        out = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return {'last_feat': out.last_hidden_state.permute(0, 2, 1)}


class MyCLIPVisionModelWithProjection(CLIPVisionModelWithProjection):
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]  # pooled_output

        image_embeds = self.visual_projection(pooled_output)
        # last_hidden_state = vision_outputs.last_hidden_state
        # hidden_states = vision_outputs.hidden_states
        # attentions = vision_outputs.attentions

        return {'last_feat': image_embeds.unsqueeze(-1)}


def get_dino_image_descriptor_model():
    raise NotImplementedError('Not yet supported')


def get_mae_image_descriptor_model():
    raise NotImplementedError('Not yet supported')


def get_image_descriptor_model(descriptor_name, pretrained_model_path, unet):
    text_encoder = tokenizer = image_descriptor_model = None
    if descriptor_name == 'clip_image':
        # image_descriptor_model = MyCLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        image_descriptor_model = MyCLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        unet.modify_encoder_hidden_state_proj(1024, 768)

    elif descriptor_name == 'clip_image_proj':
        # image_descriptor_model = MyCLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        image_descriptor_model = MyCLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")

    elif descriptor_name == 'dino_image':
        raise NotImplementedError('DINO is not yet supported')
        get_dino_image_descriptor_model()
        unet.modify_encoder_hidden_state_proj(768, 768)
        print('adding linear projection to unet for image descriptors')

    elif descriptor_name == 'mae':
        raise NotImplementedError('MAE is not yet supported')
        get_mae_image_descriptor_model()
        unet.modify_encoder_hidden_state_proj(768, 768)
        print('adding linear projection to unet for image descriptors')

    elif descriptor_name == 'learnable':
        unet.define_learnable_embeddings(128, 768)
        print(f'Successfully added learnable object queries to unet as {unet.object_queries}')

    elif descriptor_name == 'remove':
        unet.remove_cross_attention()
        print('Successfully removed cross attention layers from unet')

    else:
        assert descriptor_name == 'none'
        # load the pretrained CLIP model
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
        print('Succesfully loaded pretrained CLIP text encoder')

    return image_descriptor_model, text_encoder, tokenizer
