from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, StableDiffusionPipeline,AutoencoderTiny
from diffusers.utils.import_utils import is_xformers_available
from os.path import isfile
from pathlib import Path

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from torch.cuda.amp import custom_bwd, custom_fwd
try:
    from .perpneg_utils import weighted_perpendicular_aggregator
except:
    from perpneg_utils import weighted_perpendicular_aggregator
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True



from dataclasses import dataclass
from diffusers.utils.outputs import BaseOutput
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.vae import DecoderOutput,DecoderTiny, EncoderTiny
from typing import Tuple, Union

@dataclass
class TaesdOutput(BaseOutput):
    """
    Output of AutoencoderTiny encoding method.

    Args:
        latents (`torch.Tensor`): Encoded outputs of the `Encoder`.

    """

    latents: torch.Tensor


class Taesd(nn.Module):
    def __init__(self,device, fp16
                 ):
        super().__init__()
        self.device = device
        self.precision_t = torch.float16 if fp16 else torch.float32
        #if init_ckpt

        in_channels = 3
        out_channels = 3
        encoder_block_out_channels: Tuple[int] = (64, 64, 64, 64)
        decoder_block_out_channels: Tuple[int] = (64, 64, 64, 64)
        act_fn: str = "relu"
        latent_channels: int = 4
        upsampling_scaling_factor: int = 2
        num_encoder_blocks: Tuple[int] = (1, 3, 3, 3)
        num_decoder_blocks: Tuple[int] = (3, 3, 3, 1)
        latent_magnitude: int = 3
        latent_shift: float = 0.5
        force_upcast: float = False
        scaling_factor: float = 1.0


        self.decoder =  DecoderTiny(
            in_channels=latent_channels,
            out_channels=out_channels,
            num_blocks=num_decoder_blocks,
            block_out_channels=decoder_block_out_channels,
            upsampling_scaling_factor=upsampling_scaling_factor,
            act_fn=act_fn,
        ).to(device).to(self.precision_t)
        self.encoder =  EncoderTiny(
            in_channels=in_channels,
            out_channels=latent_channels,
            num_blocks=num_encoder_blocks,
            block_out_channels=encoder_block_out_channels,
            act_fn=act_fn,
        ).to(device).to(self.precision_t)

        print("Loading pretrained model")
        vae = AutoencoderTiny.from_pretrained("pretrained/taesd", torch_dtype=self.precision_t).to(device)
        self.decoder.load_state_dict(vae.decoder.state_dict())
        self.encoder.load_state_dict(vae.encoder.state_dict())
        del vae


        self.latent_magnitude = latent_magnitude
        self.latent_shift = latent_shift
        self.scaling_factor = scaling_factor


        self.use_slicing = False
        self.use_tiling = False

        # only relevant if vae tiling is enabled
        self.spatial_scale_factor = 2 ** out_channels
        self.tile_overlap_factor = 0.125
        self.tile_sample_min_size = 512
        self.tile_latent_min_size = self.tile_sample_min_size // self.spatial_scale_factor



    def _tiled_encode(self, x: torch.FloatTensor) -> torch.FloatTensor:
        r"""Encode a batch of images using a tiled encoder.

        When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
        steps. This is useful to keep memory use constant regardless of image size. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output.

        Args:
            x (`torch.FloatTensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.autoencoder_tiny.AutoencoderTinyOutput`] instead of a plain tuple.

        Returns:
            [`~models.autoencoder_tiny.AutoencoderTinyOutput`] or `tuple`:
                If return_dict is True, a [`~models.autoencoder_tiny.AutoencoderTinyOutput`] is returned, otherwise a
                plain `tuple` is returned.
        """
        # scale of encoder output relative to input
        sf = self.spatial_scale_factor
        tile_size = self.tile_sample_min_size

        # number of pixels to blend and to traverse between tile
        blend_size = int(tile_size * self.tile_overlap_factor)
        traverse_size = tile_size - blend_size

        # tiles index (up/left)
        ti = range(0, x.shape[-2], traverse_size)
        tj = range(0, x.shape[-1], traverse_size)

        # mask for blending
        blend_masks = torch.stack(
            torch.meshgrid([torch.arange(tile_size / sf) / (blend_size / sf - 1)] * 2, indexing="ij")
        )
        blend_masks = blend_masks.clamp(0, 1).to(x.device)

        # output array
        out = torch.zeros(x.shape[0], 4, x.shape[-2] // sf, x.shape[-1] // sf, device=x.device)
        for i in ti:
            for j in tj:
                tile_in = x[..., i : i + tile_size, j : j + tile_size]
                # tile result
                tile_out = out[..., i // sf : (i + tile_size) // sf, j // sf : (j + tile_size) // sf]
                tile = self.encoder(tile_in)
                h, w = tile.shape[-2], tile.shape[-1]
                # blend tile result into output
                blend_mask_i = torch.ones_like(blend_masks[0]) if i == 0 else blend_masks[0]
                blend_mask_j = torch.ones_like(blend_masks[1]) if j == 0 else blend_masks[1]
                blend_mask = blend_mask_i * blend_mask_j
                tile, blend_mask = tile[..., :h, :w], blend_mask[..., :h, :w]
                tile_out.copy_(blend_mask * tile + (1 - blend_mask) * tile_out)
        return out

    def _tiled_decode(self, x: torch.FloatTensor) -> torch.FloatTensor:
        r"""Encode a batch of images using a tiled encoder.

        When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
        steps. This is useful to keep memory use constant regardless of image size. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output.

        Args:
            x (`torch.FloatTensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.autoencoder_tiny.AutoencoderTinyOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        # scale of decoder output relative to input
        sf = self.spatial_scale_factor
        tile_size = self.tile_latent_min_size

        # number of pixels to blend and to traverse between tiles
        blend_size = int(tile_size * self.tile_overlap_factor)
        traverse_size = tile_size - blend_size

        # tiles index (up/left)
        ti = range(0, x.shape[-2], traverse_size)
        tj = range(0, x.shape[-1], traverse_size)

        # mask for blending
        blend_masks = torch.stack(
            torch.meshgrid([torch.arange(tile_size * sf) / (blend_size * sf - 1)] * 2, indexing="ij")
        )
        blend_masks = blend_masks.clamp(0, 1).to(x.device)

        # output array
        out = torch.zeros(x.shape[0], 3, x.shape[-2] * sf, x.shape[-1] * sf, device=x.device)
        for i in ti:
            for j in tj:
                tile_in = x[..., i : i + tile_size, j : j + tile_size]
                # tile result
                tile_out = out[..., i * sf : (i + tile_size) * sf, j * sf : (j + tile_size) * sf]
                tile = self.decoder(tile_in)
                h, w = tile.shape[-2], tile.shape[-1]
                # blend tile result into output
                blend_mask_i = torch.ones_like(blend_masks[0]) if i == 0 else blend_masks[0]
                blend_mask_j = torch.ones_like(blend_masks[1]) if j == 0 else blend_masks[1]
                blend_mask = (blend_mask_i * blend_mask_j)[..., :h, :w]
                tile_out.copy_(blend_mask * tile + (1 - blend_mask) * tile_out)
        return out


    @apply_forward_hook
    def encode(
            self, x: torch.FloatTensor, return_dict: bool = True
    ) -> Union[TaesdOutput, Tuple[torch.FloatTensor]]:
        if self.use_slicing and x.shape[0] > 1:
            output = [self._tiled_encode(x_slice) if self.use_tiling else self.encoder(x) for x_slice in x.split(1)]
            output = torch.cat(output)
        else:
            output = self._tiled_encode(x) if self.use_tiling else self.encoder(x)

        if not return_dict:
            return (output,)

        return TaesdOutput(latents=output)

    @apply_forward_hook
    def decode(self, x: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, Tuple[torch.FloatTensor]]:
        print('decode in Taesd')
        if self.use_slicing and x.shape[0] > 1:
            output = [self._tiled_decode(x_slice) if self.use_tiling else self.decoder(x) for x_slice in x.split(1)]
            output = torch.cat(output)
        else:
            output = self._tiled_decode(x) if self.use_tiling else self.decoder(x)

        if not return_dict:
            return (output,)

        return DecoderOutput(sample=output)


    def decode_latents(self, latents):
        '''

        :param latents: [B, 4, H, W],
        :return: imgs [B, 3, H, W], in [0, 1]
        '''
        latents = 1 / self.scaling_factor * latents

        imgs = self.decode(latents).sample




        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        '''

        :param imgs:  [B, 3, H, W], in [0, 1]
        :return: latents: [B, 4, H, W],
        '''


        imgs = 2 * imgs - 1 # to [-1, 1]

        posterior = self.encode(imgs).latent_dist

        latents = posterior.sample() * self.scaling_factor

        return latents


class StableDiffusion(nn.Module):
    def __init__(self, device, fp16, vram_O, sd_version='2.1', hf_key=None, t_range=[0.02, 0.98]):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        print(f'[INFO] loading stable diffusion...')

        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        self.precision_t = torch.float16 if fp16 else torch.float32

        # Create model
        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.precision_t)

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        self.taesd = Taesd(device, fp16)
        self.vae = AutoencoderKL.from_pretrained('./pretrained/vae-ft-mse-840000-ema-pruned', torch_dtype=self.precision_t).to(self.device)



        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=self.precision_t)

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loaded stable diffusion!')

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        # prompt: [str]

        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings


    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100, as_latent=False, grad_scale=1,
                   save_guidance_path:Path=None):


        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1

            # feature_image + (1 - weights_samples) * bcg_image
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, (latents.shape[0],), dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)
            noise_pred = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)


        # import kiui
        # latents_tmp = torch.randn((1, 4, 64, 64), device=self.device)
        # latents_tmp = latents_tmp.detach()
        # kiui.lo(latents_tmp)
        # self.scheduler.set_timesteps(30)
        # for i, t in enumerate(self.scheduler.timesteps):
        #     latent_model_input = torch.cat([latents_tmp] * 3)
        #     noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']
        #     noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        #     noise_pred = noise_pred_uncond + 10 * (noise_pred_pos - noise_pred_uncond)
        #     latents_tmp = self.scheduler.step(noise_pred, t, latents_tmp)['prev_sample']
        # imgs = self.decode_latents(latents_tmp)
        # kiui.vis.plot_image(imgs)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        if save_guidance_path:
            with torch.no_grad():
                if as_latent:
                    pred_rgb_512 = self.decode_latents(latents)

                # visualize predicted denoised image
                # The following block of code is equivalent to `predict_start_from_noise`...
                # see zero123_utils.py's version for a simpler implementation.
                alphas = self.scheduler.alphas.to(latents)
                total_timesteps = self.max_step - self.min_step + 1
                index = total_timesteps - t.to(latents.device) - 1 
                b = len(noise_pred)
                a_t = alphas[index].reshape(b,1,1,1).to(self.device)
                sqrt_one_minus_alphas = torch.sqrt(1 - alphas)
                sqrt_one_minus_at = sqrt_one_minus_alphas[index].reshape((b,1,1,1)).to(self.device)                
                pred_x0 = (latents_noisy - sqrt_one_minus_at * noise_pred) / a_t.sqrt() # current prediction for x_0
                result_hopefully_less_noisy_image = self.decode_latents(pred_x0.to(latents.type(self.precision_t)))

                # visualize noisier image
                result_noisier_image = self.decode_latents(latents_noisy.to(pred_x0).type(self.precision_t))

                # TODO: also denoise all-the-way

                # all 3 input images are [1, 3, H, W], e.g. [1, 3, 512, 512]
                #print(F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False).shape, pred_rgb_512.shape)
                viz_images = torch.cat([pred_rgb_512, result_noisier_image, result_hopefully_less_noisy_image],dim=0)
                save_image(viz_images, save_guidance_path)

        targets = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), targets, reduction='sum') / latents.shape[0]

        return loss
    

    def train_step_perpneg(self, text_embeddings, weights, pred_rgb, guidance_scale=100, as_latent=False, grad_scale=1,
                   save_guidance_path:Path=None):


        B = pred_rgb.shape[0]
        K = (text_embeddings.shape[0] // B) - 1 # maximum number of prompts       

        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, (latents.shape[0],), dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * (1 + K))
            tt = torch.cat([t] * (1 + K))
            unet_output = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_text = unet_output[:B], unet_output[B:]
            delta_noise_preds = noise_pred_text - noise_pred_uncond.repeat(K, 1, 1, 1)
            noise_pred = noise_pred_uncond + guidance_scale * weighted_perpendicular_aggregator(delta_noise_preds, weights, B)            

        # import kiui
        # latents_tmp = torch.randn((1, 4, 64, 64), device=self.device)
        # latents_tmp = latents_tmp.detach()
        # kiui.lo(latents_tmp)
        # self.scheduler.set_timesteps(30)
        # for i, t in enumerate(self.scheduler.timesteps):
        #     latent_model_input = torch.cat([latents_tmp] * 3)
        #     noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']
        #     noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        #     noise_pred = noise_pred_uncond + 10 * (noise_pred_pos - noise_pred_uncond)
        #     latents_tmp = self.scheduler.step(noise_pred, t, latents_tmp)['prev_sample']
        # imgs = self.decode_latents(latents_tmp)
        # kiui.vis.plot_image(imgs)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        if save_guidance_path:
            with torch.no_grad():
                if as_latent:
                    pred_rgb_512 = self.decode_latents(latents)

                # visualize predicted denoised image
                # The following block of code is equivalent to `predict_start_from_noise`...
                # see zero123_utils.py's version for a simpler implementation.
                alphas = self.scheduler.alphas.to(latents)
                total_timesteps = self.max_step - self.min_step + 1
                index = total_timesteps - t.to(latents.device) - 1 
                b = len(noise_pred)
                a_t = alphas[index].reshape(b,1,1,1).to(self.device)
                sqrt_one_minus_alphas = torch.sqrt(1 - alphas)
                sqrt_one_minus_at = sqrt_one_minus_alphas[index].reshape((b,1,1,1)).to(self.device)                
                pred_x0 = (latents_noisy - sqrt_one_minus_at * noise_pred) / a_t.sqrt() # current prediction for x_0
                result_hopefully_less_noisy_image = self.decode_latents(pred_x0.to(latents.type(self.precision_t)))

                # visualize noisier image
                result_noisier_image = self.decode_latents(latents_noisy.to(pred_x0).type(self.precision_t))



                # all 3 input images are [1, 3, H, W], e.g. [1, 3, 512, 512]
                viz_images = torch.cat([pred_rgb_512, result_noisier_image, result_hopefully_less_noisy_image],dim=0)
                save_image(viz_images, save_guidance_path)

        targets = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), targets, reduction='sum') / latents.shape[0]

        return loss


    @torch.no_grad()
    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents

    def decode_latents(self, latents):

        latents = 1 / self.vae.scaling_factor * latents

        imgs = self.vae.decode(latents).sample




        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist

        latents = posterior.sample() * self.vae.scaling_factor

        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        pos_embeds = self.get_text_embeds(prompts) # [1, 77, 768]
        neg_embeds = self.get_text_embeds(negative_prompts)
        text_embeds = torch.cat([neg_embeds, pos_embeds], dim=0) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]


        imgs2 = self.taesd.decode_latents(latents)

        imgs = torch.cat([imgs,imgs2],dim=2)
        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')





        return imgs


if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    parser.add_argument('--use_tiny_vae', action='store_true', help="use tiny vae")
    parser.add_argument('--fp16', action='store_true', help="use float16 for training")
    parser.add_argument('--vram_O', action='store_true', help="optimization for low VRAM usage")
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = StableDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()


# python guidance/test_taesd.py  "upper body photo of caucasian man in black clothes, night city street, bokeh"  --hf_key pretrained/SG161222Realistic_Vision_V5.1_noVAE -H 512 -W 512 --seed 42 --use_tiny_vae

