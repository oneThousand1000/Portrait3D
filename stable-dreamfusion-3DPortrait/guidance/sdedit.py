from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, StableDiffusionPipeline,AutoencoderTiny
import numpy as np
from pathlib import Path
import glob
import os
# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

import tqdm
try:
    from .perpneg_utils import weighted_perpendicular_aggregator
except:
    from perpneg_utils import weighted_perpendicular_aggregator


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

class StableDiffusion(nn.Module):
    def __init__(self, device, fp16, vram_O, sd_version='2.1', hf_key=None, t_range=[0.02, 0.98],):
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

        self.vae = pipe.vae
        #self.vae = AutoencoderKL.from_pretrained('F:/high_quality_3DPortraitGAN/exp/stable-dreamfusion/pretrained/vae-ft-mse-840000-ema-pruned', torch_dtype=self.precision_t).to(self.device)
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
                #
                # # visualize predicted denoised image
                # # The following block of code is equivalent to `predict_start_from_noise`...
                # # see zero123_utils.py's version for a simpler implementation.
                # alphas = self.scheduler.alphas.to(latents)
                # total_timesteps = self.max_step - self.min_step + 1
                # index = total_timesteps - t.to(latents.device) - 1
                # b = len(noise_pred)
                # a_t = alphas[index].reshape(b, 1, 1, 1).to(self.device)
                # sqrt_one_minus_alphas = torch.sqrt(1 - alphas)
                # sqrt_one_minus_at = sqrt_one_minus_alphas[index].reshape((b, 1, 1, 1)).to(self.device)
                # pred_x0 = (latents_noisy - sqrt_one_minus_at * noise_pred) / a_t.sqrt()  # current prediction for x_0
                # result_hopefully_less_noisy_image = self.decode_latents(pred_x0.to(latents.type(self.precision_t)))
                #
                # # visualize noisier image
                # result_noisier_image = self.decode_latents(latents_noisy.to(pred_x0).type(self.precision_t))
                #
                # # TODO: also denoise all-the-way
                # # all 3 input images are [1, 3, H, W], e.g. [1, 3, 512, 512]
                # # print(F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False).shape, pred_rgb_512.shape)
                # viz_images = torch.cat([pred_rgb_512, result_noisier_image, result_hopefully_less_noisy_image], dim=0)
                # save_image(viz_images, save_guidance_path)

                guidance_eval_utils = {
                    "use_perp_neg": False,
                    "neg_guidance_weights": None,
                    "text_embeddings": text_embeddings,
                    "t_orig": t,
                    "latents_noisy": latents_noisy,
                    "noise_pred": noise_pred,
                    "guidance_scale": guidance_scale,
                    "return_imgs_final": False,
                }

                guidance_eval_out = self.guidance_eval(**guidance_eval_utils)
                # decode_latents(latents_1step).permute(0, 2, 3, 1)
                # "imgs_noisy": imgs_noisy,
                # "imgs_1step": imgs_1step,
                # "imgs_1orig": imgs_1orig,
                # "imgs_final": imgs_final,
                viz_images = [pred_rgb_512]
                for k in guidance_eval_out:
                    if k.startswith("imgs_"):
                        viz_images.append(guidance_eval_out[k])
                viz_images = torch.cat(viz_images, dim=0)

                save_image(viz_images, save_guidance_path)






        targets = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), targets, reduction='sum') / latents.shape[0]

        return loss

    @torch.no_grad()
    def get_noise_pred(
            self,
            latents_noisy,
            t,
            text_embeddings,
            use_perp_neg=False,
            neg_guidance_weights=None,
            guidance_scale=100.0,
    ):
        batch_size = latents_noisy.shape[0]

        if use_perp_neg:
            raise NotImplementedError
        else:
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred =  self.unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 2).to(self.device),
                encoder_hidden_states=text_embeddings,
            ).sample

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_text + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
            )

        return noise_pred


    @torch.no_grad()
    def guidance_eval(
            self,
            t_orig,
            text_embeddings,
            latents_noisy,
            noise_pred,
            use_perp_neg=False,
            neg_guidance_weights=None,
            guidance_scale=100.0,
            return_imgs_final=False,
    ):
        # use only 50 timesteps, and find nearest of those to t
        self.scheduler.set_timesteps(50)
        self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device)
        max_items_eval = 4
        bs = (
            min(max_items_eval, latents_noisy.shape[0])
            if max_items_eval > 0
            else latents_noisy.shape[0]
        )  # batch size
        large_enough_idxs = self.scheduler.timesteps_gpu.expand([bs, -1]) > t_orig[:bs].unsqueeze(
            -1)  # sized [bs,50] > [bs,1]
        idxs = torch.min(large_enough_idxs, dim=1)[1]
        t = self.scheduler.timesteps_gpu[idxs]

        fracs = list((t / self.scheduler.config.num_train_timesteps).cpu().numpy())
        imgs_noisy = self.decode_latents(latents_noisy[:bs])

        # get prev latent
        latents_1step = []
        pred_1orig = []
        for b in range(bs):
            step_output = self.scheduler.step(
                noise_pred[b: b + 1], t[b], latents_noisy[b: b + 1], eta=1
            )
            latents_1step.append(step_output["prev_sample"])
            pred_1orig.append(step_output["pred_original_sample"])
        latents_1step = torch.cat(latents_1step)
        pred_1orig = torch.cat(pred_1orig)
        imgs_1step = self.decode_latents(latents_1step)
        imgs_1orig = self.decode_latents(pred_1orig)

        res = {
            "bs": bs,
            "noise_levels": fracs,
            "imgs_noisy": imgs_noisy,
            "imgs_1step": imgs_1step,
            "imgs_1orig": imgs_1orig,

        }
        if return_imgs_final:
            latents_final = []
            for b, i in enumerate(idxs):
                latents = latents_1step[b: b + 1]
                text_emb = (
                    text_embeddings[
                        [b, b + len(idxs), b + 2 * len(idxs), b + 3 * len(idxs)], ...
                    ]
                    if use_perp_neg
                    else text_embeddings[[b, b + len(idxs)], ...]
                )
                neg_guid = neg_guidance_weights[b: b + 1] if use_perp_neg else None
                for t in self.scheduler.timesteps[i + 1:]:
                    # pred noise
                    # noise_pred = self.get_noise_pred(
                    #     latents, t, text_emb, use_perp_neg, neg_guid,guidance_scale = guidance_scale
                    # )

                    latent_model_input = torch.cat([latents] * 2, dim=0)
                    noise_pred = self.unet(
                        latent_model_input,
                        torch.cat([t.reshape(1)] * 2).to(self.device),
                        encoder_hidden_states=text_emb,
                    ).sample

                    # perform guidance (high scale from paper!)
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_text + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                    )


                    # get prev latent
                    latents = self.scheduler.step(noise_pred, t, latents, eta=1)[
                        "prev_sample"
                    ]
                latents_final.append(latents)

            latents_final = torch.cat(latents_final)
            imgs_final = self.decode_latents(latents_final)

            res["imgs_final"] = imgs_final

        return res
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
                alphas = self.alphas.to(latents)
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

        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

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

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs


    def denoise_latents(self, text_embeddings, start_t,num_inference_steps=50, guidance_scale=7.5, latents=None):


        self.scheduler.set_timesteps(num_inference_steps)
        for   t in tqdm.tqdm(self.scheduler.timesteps):
            if t>start_t:
                continue
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


    def sdedit(self, data_dir, height=512, width=512, num_inference_steps=50,test_data_dir = None, guidance_scale=7.5):


        noise_level = 200
        res_dir = data_dir
        origin_data_dir = os.path.join(res_dir, 'data')
        if not os.path.exists(origin_data_dir):
            print('no data dir')
            return

        update_data_dir = os.path.join(res_dir, 'update_data')
        os.makedirs(update_data_dir, exist_ok=True)

        if len(glob.glob(origin_data_dir + '/*.png')) == len(glob.glob(update_data_dir + '/*.png')):
            print('already done')
            return
        print('gen data for ', res_dir)

        name = os.path.basename(res_dir)

        prompt_path = os.path.join(test_data_dir, f'{name}/prompt.txt')
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r') as f:
                prompts = f.read().strip()
        else:
            raise ValueError('prompt.txt not exists')

        if isinstance(prompts, str):
            prompts = [prompts]
        # Prompts -> text embeds
        pos_embeds = self.get_text_embeds(prompts)  # [1, 77, 768]
        neg_embeds = self.get_text_embeds('worst quality, low quality, jpeg artifacts, blurry')
        text_embeds = torch.cat([neg_embeds, pos_embeds], dim=0)  # [2, 77, 768]

        for image_path in glob.glob(origin_data_dir + '/*.png'):
            image = PIL.Image.open(image_path).convert('RGB')
            image = np.array(image)

            origin_img = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)  # --> 0,1
            origin_img = origin_img / 255.0

            latents = self.encode_imgs(origin_img)

            t = torch.tensor([noise_level], dtype=torch.long,
                             device=self.device)

            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                noise = torch.randn_like(latents)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)

                latents = self.denoise_latents(text_embeds, noise_level, num_inference_steps=num_inference_steps,
                                               guidance_scale=guidance_scale, latents=latents_noisy)

            # Img latents -> imgs
            img = self.decode_latents(latents)  # [1, 3, 512, 512]
            # Img to Numpy
            img = img.detach().cpu().permute(0, 2, 3, 1).numpy()
            img = (img * 255).round().astype('uint8')[0]

            PIL.Image.fromarray(img).save(os.path.join(update_data_dir, os.path.basename(image_path)))



if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt
    import PIL

    parser = argparse.ArgumentParser()
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")


    parser.add_argument('--data_dir', type=str,help='Network pickle filename', required=True)
    parser.add_argument('--test_data_dir', type=str,help='test_data_dir', required=True)


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

    imgs = sd.sdedit(opt.data_dir,opt.H, opt.W, opt.steps,opt.test_data_dir)



#