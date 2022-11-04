import json
from IPython import display

import os, sys, time
import cv2
import numpy as np
import random
import requests
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from contextlib import nullcontext
from einops import rearrange, repeat
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from skimage.exposure import match_histograms
from torchvision.utils import make_grid
from types import SimpleNamespace
from torch import autocast
import re
from scipy.ndimage import gaussian_filter
from super_res import upscale_image
from typing import Tuple

sys.path.extend([
    'src/taming-transformers',
    'src/clip',
    'stable-diffusion/',
    'k-diffusion',
    'pytorch3d-lite',
    'AdaBins',
    'MiDaS',
])

from helpers import sampler_fn
from k_diffusion.external import CompVisDenoiser
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


def sanitize(prompt):
    whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    tmp = ''.join(filter(whitelist.__contains__, prompt))
    return tmp.replace(' ', '_')


def construct_RotationMatrixHomogenous(rotation_angles):
    assert (type(rotation_angles) == list and len(rotation_angles) == 3)
    RH = np.eye(4, 4)
    cv2.Rodrigues(np.array(rotation_angles), RH[0:3, 0:3])
    return RH


def add_noise(sample: torch.Tensor, noise_amt: float) -> torch.Tensor:
    return sample + torch.randn(sample.shape, device=sample.device) * noise_amt


def load_img(path, shape, use_alpha_as_mask=False):
    # use_alpha_as_mask: Read the alpha channel of the image as the mask image
    if path.startswith('http://') or path.startswith('https://'):
        image = Image.open(requests.get(path, stream=True).raw)
    else:
        image = Image.open(path)

    if use_alpha_as_mask:
        image = image.convert('RGBA')
    else:
        image = image.convert('RGB')

    image = image.resize(shape, resample=Image.Resampling.LANCZOS)

    mask_image = None
    if use_alpha_as_mask:
        # Split alpha channel into a mask_image
        red, green, blue, alpha = Image.Image.split(image)
        mask_image = alpha.convert('L')
        image = image.convert('RGB')

    image = np.array(image).astype(np.float16) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = 2. * image - 1.

    return image, mask_image


# used by prepare_mask
def load_mask_latent(mask_input, shape): 
    if isinstance(mask_input, str):
        if mask_input.startswith('http://') or mask_input.startswith(
                'https://'):
            mask_image = Image.open(requests.get(
                mask_input, stream=True).raw).convert('RGBA')
        else:
            mask_image = Image.open(mask_input).convert('RGBA')
    elif isinstance(mask_input, Image.Image):
        mask_image = mask_input
    else:
        raise Exception("mask_input must be a PIL image or a file name")

    mask_w_h = (shape[-1], shape[-2])
    mask = mask_image.resize(mask_w_h, resample=Image.Resampling.LANCZOS)
    mask = mask.convert("L")
    return mask


def prepare_mask(mask_input: Tuple[str, Image.Image],  # path to the mask image or a PIL Image object
                 mask_shape: list,                     # shape of the image to match, usually latent_image.shape
                 args: SimpleNamespace,                # args object from DeforumArgs
                 mask_brightness_adjust=1.0,           # brightness of the mask. 0 is black, 1 is no adjustment, >1 is brighter
                 mask_contrast_adjust=1.0):            # contrast of the mask. 0 is black, 1 is no adjustment, >1 is more contrast

    mask = load_mask_latent(mask_input, mask_shape)

    # Mask brightness/contrast adjustments
    if mask_brightness_adjust != 1:
        mask = TF.adjust_brightness(mask, mask_brightness_adjust)
    if mask_contrast_adjust != 1:
        mask = TF.adjust_contrast(mask, mask_contrast_adjust)

    # Mask image to array
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask, (4, 1, 1))
    mask = np.expand_dims(mask, axis=0)
    mask = torch.from_numpy(mask)

    if args.invert_mask:
        mask = ((mask - 0.5) * -1) + 0.5

    mask = np.clip(mask, 0, 1)
    return mask


def maintain_colors(prev_img, color_match_sample, mode):
    if mode == 'Match Frame 0 RGB':
        return match_histograms(prev_img,
                                color_match_sample,
                                multichannel=True)
    elif mode == 'Match Frame 0 HSV':
        prev_img_hsv = cv2.cvtColor(prev_img, cv2.COLOR_RGB2HSV)
        color_match_hsv = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2HSV)
        matched_hsv = match_histograms(prev_img_hsv,
                                       color_match_hsv,
                                       multichannel=True)
        return cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2RGB)
    else:  # Match Frame 0 LAB
        prev_img_lab = cv2.cvtColor(prev_img, cv2.COLOR_RGB2LAB)
        color_match_lab = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2LAB)
        matched_lab = match_histograms(prev_img_lab,
                                       color_match_lab,
                                       multichannel=True)
        return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)


class SamplerCallback(object):
    # Creates the callback function to be passed into the samplers for each step
    def __init__(self,
                 args,
                 device="cuda",
                 mask=None,
                 init_latent=None,
                 sigmas=None,
                 sampler=None,
                 verbose=False):
        self.sampler_name = args.sampler
        self.dynamic_threshold = args.dynamic_threshold
        self.static_threshold = args.static_threshold
        self.mask = mask
        self.init_latent = init_latent
        self.sigmas = sigmas
        self.sampler = sampler
        self.verbose = verbose

        self.batch_size = args.n_samples
        self.save_sample_per_step = args.save_sample_per_step
        self.show_sample_per_step = args.show_sample_per_step
        self.paths_to_image_steps = [
            os.path.join(args.outdir, f"{args.time}_{index:02}_{args.seed}")
            for index in range(args.n_samples)
        ]

        if self.save_sample_per_step:
            for path in self.paths_to_image_steps:
                os.makedirs(path, exist_ok=True)

        self.step_index = 0

        self.noise = None
        if init_latent is not None:
            self.noise = torch.randn_like(init_latent, device=device)

        self.mask_schedule = None
        if sigmas is not None and len(sigmas) > 0:
            self.mask_schedule, _ = torch.sort(sigmas / torch.max(sigmas))
        elif len(sigmas) == 0:
            self.mask = None  # no mask needed if no steps (usually happens because strength==1.0)

        if self.sampler_name in ["plms", "ddim"]:
            if mask is not None:
                assert sampler is not None, "Callback function for stable-diffusion samplers requires sampler variable"

        if self.sampler_name in ["plms", "ddim"]:
            # Callback function formated for compvis latent diffusion samplers
            self.callback = self.img_callback_
        else:
            # Default callback function uses k-diffusion sampler variables
            self.callback = self.k_callback_

        self.verbose_print = print if verbose else lambda *args, **kwargs: None

    def view_sample_step(self, model, latents, path_name_modifier=''):
        if self.save_sample_per_step or self.show_sample_per_step:
            samples = model.decode_first_stage(latents)
            if self.save_sample_per_step:
                fname = f'{path_name_modifier}_{self.step_index:05}.png'
                for i, sample in enumerate(samples):
                    sample = sample.double().cpu().add(1).div(2).clamp(0, 1)
                    sample = torch.tensor(np.array(sample))
                    grid = make_grid(sample, 4).cpu()
                    TF.to_pil_image(grid).save(
                        os.path.join(self.paths_to_image_steps[i], fname))
            if self.show_sample_per_step:
                print(path_name_modifier)
                self.display_images(samples)
        return

    def display_images(self, images):
        images = images.double().cpu().add(1).div(2).clamp(0, 1)
        images = torch.tensor(np.array(images))
        grid = make_grid(images, 4).cpu()
        display.display(TF.to_pil_image(grid))
        return

    # The callback function is applied to the image at each step
    def dynamic_thresholding_(self, img, threshold):
        # Dynamic thresholding from Imagen paper (May 2022)
        s = np.percentile(np.abs(img.cpu()),
                          threshold,
                          axis=tuple(range(1, img.ndim)))
        s = np.max(np.append(s, 1.0))
        torch.clamp_(img, -1 * s, s)
        torch.FloatTensor.div_(img, s)

    # Callback for samplers in the k-diffusion repo, called thus:
    #   callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
    def k_callback_(self, args_dict):
        self.step_index = args_dict['i']
        if self.dynamic_threshold is not None:
            self.dynamic_thresholding_(args_dict['x'], self.dynamic_threshold)
        if self.static_threshold is not None:
            torch.clamp_(args_dict['x'], -1 * self.static_threshold,
                         self.static_threshold)
        if self.mask is not None:
            init_noise = self.init_latent + self.noise * args_dict['sigma']
            is_masked = torch.logical_and(
                self.mask >= self.mask_schedule[args_dict['i']],
                self.mask != 0)
            new_img = init_noise * torch.where(
                is_masked, 1, 0) + args_dict['x'] * torch.where(
                    is_masked, 0, 1)
            args_dict['x'].copy_(new_img)

        self.view_sample_step(args_dict['denoised'], "x0_pred")

    # Callback for Compvis samplers
    # Function that is called on the image (img) and step (i) at each step
    def img_callback_(self, img, i, device="cuda"):
        self.step_index = i
        # Thresholding functions
        if self.dynamic_threshold is not None:
            self.dynamic_thresholding_(img, self.dynamic_threshold)
        if self.static_threshold is not None:
            torch.clamp_(img, -1 * self.static_threshold,
                         self.static_threshold)
        if self.mask is not None:
            i_inv = len(self.sigmas) - i - 1
            init_noise = self.sampler.stochastic_encode(
                self.init_latent,
                torch.tensor([i_inv] * self.batch_size).to(device),
                noise=self.noise)
            is_masked = torch.logical_and(self.mask >= self.mask_schedule[i],
                                          self.mask != 0)
            new_img = init_noise * torch.where(
                is_masked, 1, 0) + img * torch.where(is_masked, 0, 1)
            img.copy_(new_img)

        self.view_sample_step(img, "x")


def sample_from_cv2(sample: np.ndarray) -> torch.Tensor:
    sample = ((sample.astype(float) / 255.0) * 2) - 1
    sample = sample[None].transpose(0, 3, 1, 2).astype(np.float16)
    sample = torch.from_numpy(sample)
    return sample


def sample_to_cv2(sample: torch.Tensor, type=np.uint8) -> np.ndarray:
    sample_f32 = rearrange(sample.squeeze().cpu().numpy(),
                           "c h w -> h w c").astype(np.float32)
    sample_f32 = ((sample_f32 * 0.5) + 0.5).clip(0, 1)
    sample_int8 = (sample_f32 * 255)
    return sample_int8.astype(type)


def check_is_number(value):
    float_pattern = r'^(?=.)([+-]?([0-9]*)(\.([0-9]+))?)$'
    return re.match(float_pattern, value)


# prompt weighting with colons and number coefficients (like 'bacon:0.75 eggs:0.25')
# borrowed from https://github.com/kylewlacy/stable-diffusion/blob/0a4397094eb6e875f98f9d71193e350d859c4220/ldm/dream/conditioning.py
# and https://github.com/raefu/stable-diffusion-automatic/blob/unstablediffusion/modules/processing.py
def get_uc_and_c(prompts, model, args, frame=0):
    prompt = prompts[0]  # they are the same in a batch anyway

    # get weighted sub-prompts
    negative_subprompts, positive_subprompts = split_weighted_subprompts(
        prompt, frame, not args.normalize_prompt_weights)

    uc = get_learned_conditioning(model, negative_subprompts, "", args, -1)
    c = get_learned_conditioning(model, positive_subprompts, prompt, args, 1)

    return (uc, c)


def get_learned_conditioning(model, weighted_subprompts, text, args, sign=1):
    if len(weighted_subprompts) < 1:
        log_tokenization(text, model, args.log_weighted_subprompts, sign)
        c = model.get_learned_conditioning(args.n_samples * [text])
    else:
        c = None
        for subtext, subweight in weighted_subprompts:
            log_tokenization(subtext, model, args.log_weighted_subprompts,
                             sign * subweight)
            if c is None:
                c = model.get_learned_conditioning(args.n_samples * [subtext])
                c *= subweight
            else:
                c.add_(model.get_learned_conditioning(args.n_samples *
                                                      [subtext]),
                       alpha=subweight)

    return c


def parse_weight(match, frame=0) -> float:
    import numexpr
    w_raw = match.group("weight")
    if w_raw == None:
        return 1
    if check_is_number(w_raw):
        return float(w_raw)
    else:
        t = frame
        if len(w_raw) < 3:
            print(
                'the value inside `-characters cannot represent a math function'
            )
            return 1
        return float(numexpr.evaluate(w_raw[1:-1]))


def normalize_prompt_weights(parsed_prompts):
    if len(parsed_prompts) == 0:
        return parsed_prompts
    weight_sum = sum(map(lambda x: x[1], parsed_prompts))
    if weight_sum == 0:
        print(
            "Warning: Subprompt weights add up to zero. Discarding and using even weights instead."
        )
        equal_weight = 1 / max(len(parsed_prompts), 1)
        return [(x[0], equal_weight) for x in parsed_prompts]
    return [(x[0], x[1] / weight_sum) for x in parsed_prompts]


def split_weighted_subprompts(text, frame=0, skip_normalize=False):
    """
    grabs all text up to the first occurrence of ':'
    uses the grabbed text as a sub-prompt, and takes the value following ':' as weight
    if ':' has no value defined, defaults to 1.0
    repeats until no text remaining
    """
    prompt_parser = re.compile(
        """
            (?P<prompt>         # capture group for 'prompt'
            (?:\\\:|[^:])+      # match one or more non ':' characters or escaped colons '\:'
            )                   # end 'prompt'
            (?:                 # non-capture group
            :+                  # match one or more ':' characters
            (?P<weight>((        # capture group for 'weight'
            -?\d+(?:\.\d+)?     # match positive or negative integer or decimal number
            )|(                 # or
            `[\S\s]*?`# a math function
            )))?                 # end weight capture group, make optional
            \s*                 # strip spaces after weight
            |                   # OR
            $                   # else, if no ':' then match end of line
            )                   # end non-capture group
            """, re.VERBOSE)
    negative_prompts = []
    positive_prompts = []
    for match in re.finditer(prompt_parser, text):
        w = parse_weight(match, frame)
        if w < 0:
            # negating the sign as we'll feed this to uc
            negative_prompts.append((match.group("prompt").replace("\\:",
                                                                   ":"), -w))
        elif w > 0:
            positive_prompts.append((match.group("prompt").replace("\\:",
                                                                   ":"), w))

    if skip_normalize:
        return (negative_prompts, positive_prompts)
    return (normalize_prompt_weights(negative_prompts),
            normalize_prompt_weights(positive_prompts))


# shows how the prompt is tokenized
# usually tokens have '</w>' to indicate end-of-word,
# but for readability it has been replaced with ' '
def log_tokenization(text, model, log=False, weight=1):
    if not log:
        return
    tokens = model.cond_stage_model.tokenizer._tokenize(text)
    tokenized = ""
    discarded = ""
    usedTokens = 0
    totalTokens = len(tokens)
    for i in range(0, totalTokens):
        token = tokens[i].replace('</w>', ' ')
        # alternate color
        s = (usedTokens % 6) + 1
        if i < model.cond_stage_model.max_length:
            tokenized = tokenized + f"\x1b[0;3{s};40m{token}"
            usedTokens += 1
        else:  # over max token length
            discarded = discarded + f"\x1b[0;3{s};40m{token}"
    print(
        f"\n>> Tokens ({usedTokens}), Weight ({weight:.2f}):\n{tokenized}\x1b[0m"
    )
    if discarded != "":
        print(
            f">> Tokens Discarded ({totalTokens-usedTokens}):\n{discarded}\x1b[0m"
        )


def generate(args,
             model,
             frame=0,
             device="cuda",
             return_latent=False,
             return_sample=False,
             return_c=False):
    seed_everything(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    sampler = PLMSSampler(model) if args.sampler == 'plms' else DDIMSampler(
        model)
    model_wrap = CompVisDenoiser(model)
    batch_size = args.n_samples
    prompt = args.prompt
    assert prompt is not None
    data = [batch_size * [prompt]]
    precision_scope = autocast if args.precision == "autocast" else nullcontext

    init_latent = None
    mask_image = None
    init_image = None
    if args.init_latent is not None:
        init_latent = args.init_latent
    elif args.init_sample is not None:
        with precision_scope("cuda"):
            init_latent = model.get_first_stage_encoding(
                model.encode_first_stage(args.init_sample))
    elif args.use_init and args.init_image != None and args.init_image != '':
        init_image, mask_image = load_img(
            args.init_image,
            shape=(args.W, args.H),
            use_alpha_as_mask=args.use_alpha_as_mask)
        init_image = init_image.to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        with precision_scope("cuda"):
            init_latent = model.get_first_stage_encoding(
                model.encode_first_stage(init_image))  # move to latent space

    if not args.use_init and args.strength > 0 and args.strength_0_no_init:
        print(
            "\nNo init image, but strength > 0. Strength has been auto set to 0, since use_init is False."
        )
        print(
            "If you want to force strength > 0 with no init, please set strength_0_no_init to False.\n"
        )
        args.strength = 0

    # Mask functions
    if args.use_mask:
        assert args.mask_file is not None or mask_image is not None, "use_mask==True: An mask image is required for a mask. Please enter a mask_file or use an init image with an alpha channel"
        assert args.use_init, "use_mask==True: use_init is required for a mask"
        assert init_latent is not None, "use_mask==True: An latent init image is required for a mask"

        mask = prepare_mask(
            args.mask_file if mask_image is None else mask_image,
            init_latent.shape, args.mask_contrast_adjust,
            args.mask_brightness_adjust)

        if (torch.all(mask == 0)
                or torch.all(mask == 1)) and args.use_alpha_as_mask:
            raise Warning(
                "use_alpha_as_mask==True: Using the alpha channel from the init image as a mask, but the alpha channel is blank."
            )

        mask = mask.to(device)
        mask = repeat(mask, '1 ... -> b ...', b=batch_size)
    else:
        mask = None

    assert not (
        (args.use_mask and args.overlay_mask) and
        (args.init_sample is None and init_image is None)
    ), "Need an init image when use_mask == True and overlay_mask == True"

    t_enc = int((1.0 - args.strength) * args.steps)

    # Noise schedule for the k-diffusion samplers (used for masking)
    k_sigmas = model_wrap.get_sigmas(args.steps)
    k_sigmas = k_sigmas[len(k_sigmas) - t_enc - 1:]

    if args.sampler in ['plms', 'ddim']:
        sampler.make_schedule(ddim_num_steps=args.steps,
                              ddim_eta=args.ddim_eta,
                              ddim_discretize='fill',
                              verbose=False)

    callback = SamplerCallback(args=args,
                               mask=mask,
                               init_latent=init_latent,
                               sigmas=k_sigmas,
                               sampler=sampler,
                               verbose=False).callback

    results = []
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for prompts in data:
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    if args.prompt_weighting:
                        uc, c = get_uc_and_c(prompts, model, args, frame)
                    else:
                        uc = model.get_learned_conditioning(batch_size * [""])
                        c = model.get_learned_conditioning(prompts)

                    if args.scale == 1.0:
                        uc = None
                    if args.init_c != None:
                        c = args.init_c

                    if args.sampler in [
                            "klms", "dpm2", "dpm2_ancestral", "heun", "euler",
                            "euler_ancestral"
                    ]:
                        samples = sampler_fn(c=c,
                                             uc=uc,
                                             args=args,
                                             model_wrap=model_wrap,
                                             init_latent=init_latent,
                                             t_enc=t_enc,
                                             device=device,
                                             cb=callback)
                    else:
                        # args.sampler == 'plms' or args.sampler == 'ddim':
                        if init_latent is not None and args.strength > 0:
                            z_enc = sampler.stochastic_encode(
                                init_latent,
                                torch.tensor([t_enc] * batch_size).to(device))
                        else:
                            z_enc = torch.randn([
                                args.n_samples, args.C, args.H // args.f,
                                args.W // args.f
                            ],
                                                device=device)
                        if args.sampler == 'ddim':
                            samples = sampler.decode(
                                z_enc,
                                c,
                                t_enc,
                                unconditional_guidance_scale=args.scale,
                                unconditional_conditioning=uc,
                                img_callback=callback)
                        elif args.sampler == 'plms':  # no "decode" function in plms, so use "sample"
                            shape = [
                                args.C, args.H // args.f, args.W // args.f
                            ]
                            samples, _ = sampler.sample(
                                S=args.steps,
                                conditioning=c,
                                batch_size=args.n_samples,
                                shape=shape,
                                verbose=False,
                                unconditional_guidance_scale=args.scale,
                                unconditional_conditioning=uc,
                                eta=args.ddim_eta,
                                x_T=z_enc,
                                img_callback=callback)
                        else:
                            raise Exception(
                                f"Sampler {args.sampler} not recognised.")

                    if return_latent:
                        results.append(samples.clone())

                    x_samples = model.decode_first_stage(samples)

                    if args.use_mask and args.overlay_mask:
                        # Overlay the masked image after the image is generated
                        if args.init_sample is not None:
                            img_original = args.init_sample
                        elif init_image is not None:
                            img_original = init_image
                        else:
                            raise Exception(
                                "Cannot overlay the masked image without an init image to overlay"
                            )

                        mask_fullres = prepare_mask(
                            args.mask_file
                            if mask_image is None else mask_image,
                            img_original.shape, args.mask_contrast_adjust,
                            args.mask_brightness_adjust)
                        mask_fullres = mask_fullres[:, :3, :, :]
                        mask_fullres = repeat(mask_fullres,
                                              '1 ... -> b ...',
                                              b=batch_size)

                        mask_fullres[mask_fullres < mask_fullres.max()] = 0
                        mask_fullres = gaussian_filter(mask_fullres,
                                                       args.mask_overlay_blur)
                        mask_fullres = torch.Tensor(mask_fullres).to(device)

                        x_samples = img_original * mask_fullres + x_samples * (
                            (mask_fullres * -1.0) + 1)

                    if return_sample:
                        results.append(x_samples.clone())

                    x_samples = torch.clamp((x_samples + 1.0) / 2.0,
                                            min=0.0,
                                            max=1.0)

                    if return_c:
                        results.append(c.clone())

                    for x_sample in x_samples:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(),
                                                    'c h w -> h w c')
                        image = Image.fromarray(x_sample.astype(np.uint8))
                        results.append(image)
    return results


# Select and Load Model
def load_model(args,                         # args from astro.py
              load_on_run_all=True,          # whether to load the model when running all cells
              half_precision=True):          # whether to use half precision
    model_map = {
        "sd-v1-5-full-ema.ckpt": {
            'url': 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt',
            'requires_login': True
        },
        "sd-v1-5.ckpt": {
            'url': 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt',
            'requires_login': True
        },
        "sd-v1-4-full-ema.ckpt": {
            'sha256': '14749efc0ae8ef0329391ad4436feb781b402f4fece4883c7ad8d10556d8a36a',
            'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt',
            'requires_login': True,
        },
        "sd-v1-4.ckpt": {
            'sha256': 'fe4efff1e174c627256e44ec2991ba279b3816e364b49f9be2abc0b3ff3f8556',
            'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt',
            'requires_login': True,
        },
        "sd-v1-3-full-ema.ckpt": {
            'sha256': '54632c6e8a36eecae65e36cb0595fab314e1a1545a65209f24fde221a8d4b2ca',
            'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-3-original/resolve/main/sd-v1-3-full-ema.ckpt',
            'requires_login': True,
        },
        "sd-v1-3.ckpt": {
            'sha256': '2cff93af4dcc07c3e03110205988ff98481e86539c51a8098d4f2236e41f7f2f',
            'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-3-original/resolve/main/sd-v1-3.ckpt',
            'requires_login': True,
        },
        "sd-v1-2-full-ema.ckpt": {
            'sha256': 'bc5086a904d7b9d13d2a7bccf38f089824755be7261c7399d92e555e1e9ac69a',
            'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-2-original/resolve/main/sd-v1-2-full-ema.ckpt',
            'requires_login': True,
        },
        "sd-v1-2.ckpt": {
            'sha256': '3b87d30facd5bafca1cbed71cfb86648aad75d1c264663c0cc78c7aea8daec0d',
            'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-2-original/resolve/main/sd-v1-2.ckpt',
            'requires_login': True,
        },
        "sd-v1-1-full-ema.ckpt": {
            'sha256': 'efdeb5dc418a025d9a8cc0a8617e106c69044bc2925abecc8a254b2910d69829',
            'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-1-original/resolve/main/sd-v1-1-full-ema.ckpt',
            'requires_login': True,
        },
        "sd-v1-1.ckpt": {
            'sha256': '86cd1d3ccb044d7ba8db743d717c9bac603c4043508ad2571383f954390f3cea',
            'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-1-original/resolve/main/sd-v1-1.ckpt',
            'requires_login': True,
        },
        "robo-diffusion-v1.ckpt": {
            'sha256': '244dbe0dcb55c761bde9c2ac0e9b46cc9705ebfe5f1f3a7cc46251573ea14e16',
            'url': 'https://huggingface.co/nousr/robo-diffusion/resolve/main/models/robo-diffusion-v1.ckpt',
            'requires_login': False,
        },
        "waifu-diffusion-v1-3.ckpt": {
            'sha256': '26cf2a2e30095926bb9fd9de0c83f47adc0b442dbfdc3d667d43778e8b70bece',
            'url': 'https://huggingface.co/hakurei/waifu-diffusion-v1-3/resolve/main/model-epoch05-float16.ckpt',
            'requires_login': False,
        },
        "disney-diffusion-v1.ckpt": {
            'url': 'https://huggingface.co/nitrosocke/mo-di-diffusion/blob/main/moDi-v1-pruned.ckpt',
            'requires_login': False,
        },
        '256x256-diffusion-uncond.pt': {
		    'sha256': 'a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1',
		    'url': 'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt',
            'requires_login': False
	    },
        '512x512-diffusion-uncond.pt': {
            'sha256': '9c111ab89e214862b76e1fa6a1b3f1d329b1a88281885943d2cdbe357ad57648',
            'url': 'https://huggingface.co/lowlevelware/512x512_diffusion_unconditional_ImageNet/resolve/main/512x512_diffusion_uncond_finetune_008100.pt',
            'requires_login': False
        },
        'portrait-diffusion-v1-0.pt': {
            'sha256': 'b7e8c747af880d4480b6707006f1ace000b058dd0eac5bb13558ba3752d9b5b9',
            'url': 'https://huggingface.co/felipe3dartist/portrait_generator_v001/resolve/main/portrait_generator_v001_ema_0.9999_1MM.pt',
            'requires_login': False
        },
        'pixelart-diffusion-v1-3.pt': {
            'sha256': 'a73b40556634034bf43b5a716b531b46fb1ab890634d854f5bcbbef56838739a',
            'url': 'https://huggingface.co/KaliYuga/PADexpanded/resolve/main/PADexpanded.pt',
            'requires_login': False
        },
        'pixel-art-diffusion-hard-256.pt': {
            'sha256': 'be4a9de943ec06eef32c65a1008c60ad017723a4d35dc13169c66bb322234161',
            'url': 'https://huggingface.co/KaliYuga/pixel_art_diffusion_hard_256/resolve/main/pixel_art_diffusion_hard_256.pt',
            'requires_login': False
        },
        'pixelart-diffusion-soft-256.pt': {
            'sha256': 'd321590e46b679bf6def1f1914b47c89e762c76f19ab3e3392c8ca07c791039c',
            'url': 'https://huggingface.co/KaliYuga/pixel_art_diffusion_soft_256/resolve/main/pixel_art_diffusion_soft_256.pt',
            'requires_login': False
        },
        'pixelart-diffusion-4k.pt': {
            'sha256': 'a1ba4f13f6dabb72b1064f15d8ae504d98d6192ad343572cc416deda7cccac30',
            'uri_list': 'https://huggingface.co/KaliYuga/pixelartdiffusion4k/resolve/main/pixelartdiffusion4k.pt',
            'requires_login': False
        },
        'watercolor-diffusion-v2.pt': {
            'sha256': '49c281b6092c61c49b0f1f8da93af9b94be7e0c20c71e662e2aa26fee0e4b1a9',
            'url': 'https://huggingface.co/KaliYuga/watercolordiffusion_2/resolve/main/watercolordiffusion_2.pt',
            'requires_login': False
        },
        'scifipulp-diffusion.pt': {
            'sha256': 'b79e62613b9f50b8a3173e5f61f0320c7dbb16efad42a92ec94d014f6e17337f',
            'url': 'https://huggingface.co/KaliYuga/PulpSciFiDiffusion/resolve/main/PulpSciFiDiffusion.pt',
            'requires_login': False
        }
    }

    # config path
    models_path = './'
    ckpt_config_path = args.custom_config_path if args.custom_config_path != None and args.custom_config_path != "" else args.custom_config_path
    if os.path.exists(ckpt_config_path):
        print(f"{ckpt_config_path} exists")
    else:
        ckpt_config_path = "./stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
    print(f"Using config: {ckpt_config_path}")

    # checkpoint path or download
    ckpt_path = args.custom_checkpoint_path if args.model_checkpoint == "custom" else os.path.join(
        models_path, args.model_checkpoint)
    ckpt_valid = True
    if os.path.exists(ckpt_path):
        print(f"{ckpt_path} exists")
    elif 'url' in model_map[args.model_checkpoint]:
        url = model_map[args.model_checkpoint]['url']

        # CLI dialogue to authenticate download
        if model_map[args.model_checkpoint]['requires_login']:
            print("This model requires an authentication token")
            print(
                "Please ensure you have accepted its terms of service before continuing."
            )

            username = input("What is your huggingface username?:")
            token = input("What is your huggingface token?:")

            _, path = url.split("https://")

            url = f"https://{username}:{token}@{path}"

        # contact server for model
        print(
            f"Attempting to download {args.model_checkpoint}...this may take a while"
        )
        ckpt_request = requests.get(url)
        request_status = ckpt_request.status_code

        # inform user of errors
        if request_status == 403:
            raise ConnectionRefusedError(
                "You have not accepted the license for this model.")
        elif request_status == 404:
            raise ConnectionError("Could not make contact with server")
        elif request_status != 200:
            raise ConnectionError(
                f"Some other error has ocurred - response code: {request_status}"
            )

        # write to model path
        with open(os.path.join(models_path, args.model_checkpoint),
                  'wb') as model_file:
            model_file.write(ckpt_request.content)
    else:
        print(
            f"Please download model checkpoint and place in {os.path.join(models_path, args.model_checkpoint)}"
        )
        ckpt_valid = False

    if args.check_sha256 and args.model_checkpoint != "custom" and ckpt_valid:
        import hashlib
        print("\n...checking sha256")
        with open(ckpt_path, "rb") as f:
            bytes = f.read()
            hash = hashlib.sha256(bytes).hexdigest()
            del bytes
        if not hasattr(model_map, args.model_checkpoint):
            print("No sha256 found for model")
        elif model_map[args.model_checkpoint]["sha256"] == hash:
            print("hash is correct\n")
        else:
            print("hash in not correct\n")
            ckpt_valid = False

    if ckpt_valid:
        print(f"Using ckpt: {ckpt_path}")

    if load_on_run_all and ckpt_valid:
        local_config = OmegaConf.load(f"{ckpt_config_path}")
        device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = load_model_from_config(local_config,
                                       f"{ckpt_path}",
                                       device=device,
                                       half_precision=half_precision)
        model = model.to(device)
        return model
    
    raise Exception("Failed to load model at checkpoint " + ckpt_path)


def load_model_from_config(config,
                           ckpt,
                           device,
                           verbose=False,
                           half_precision=True):
    map_location = "cuda"
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location=map_location)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if half_precision:
        model = model.half().to(device)
    else:
        model = model.to(device)
    model.eval()
    return model


def next_seed(args):
    if args.seed_behavior == 'iter':
        args.seed += 1
    elif args.seed_behavior == 'fixed':
        pass
    else:
        args.seed = random.randint(0, 2**8 - 1)
    return args.seed


def render_image_batch(args: SimpleNamespace, prompts: list[str] = [], upscale_ratio: int = 1) -> None:
    args.prompts = {k: f"{v:05d}" for v, k in enumerate(prompts)}

    # create output folder for the batch
    os.makedirs(args.outdir, exist_ok=True)
    if args.save_settings or args.save_samples:
        print(f"Saving to {os.path.join(args.outdir, args.time)}_*")

    # save settings for the batch
    if args.save_settings:
        filename = os.path.join(args.outdir, f"{args.time}_settings.txt")
        with open(filename, "w+", encoding="utf-8") as f:
            json.dump(dict(args.__dict__), f, ensure_ascii=False, indent=4)

    index = 0

    # function for init image batching
    init_array = []
    if args.use_init:
        if args.init_image == "":
            raise FileNotFoundError("No path was given for init_image")
        if args.init_image.startswith('http://') or args.init_image.startswith(
                'https://'):
            init_array.append(args.init_image)
        elif not os.path.isfile(args.init_image):
            if args.init_image[-1] != "/":  # avoids path error by adding / to end if not there
                args.init_image += "/"
            
            # iterates dir and appends images to init_array
            for image in sorted(os.listdir(args.init_image)):
                if image.split(".")[-1] in ("png", "jpg", "jpeg"):
                    init_array.append(args.init_image + image)
        else:
            init_array.append(args.init_image)
    else:
        init_array = [""]

    # when doing large batches don't flood browser with images
    clear_between_batches = args.n_batch >= 32
    model = load_model(args)

    for iprompt, prompt in enumerate(prompts):
        args.prompt = prompt
        print(f"Prompt {iprompt+1} of {len(prompts)}")
        print(f"{args.prompt}")

        all_images = []

        # init image batching
        for batch_index in range(args.n_batch):
            if clear_between_batches and batch_index % 32 == 0:
                display.clear_output(wait=True)
            print(f"Batch {batch_index+1} of {args.n_batch}")

            for image in init_array:
                args.init_image = image
                args.time = str(round(time.time()))
                results = generate(args, model)
                for image in results:
                    if args.make_grid:
                        all_images.append(T.functional.pil_to_tensor(image))
                    if args.save_samples:
                        if args.filename_format == "{timestring}_{index}_{prompt}.png":
                            filename = f"{args.time}_{sanitize(prompt)[:160]}.png"
                        else:
                            filename = f"{args.time}_{args.seed}.png"
                        if upscale_ratio > 1:
                            print(f"Upscaling {filename} by {upscale_ratio}")
                            image = upscale_image(image, upscale_ratio)
                        image.save(os.path.join(args.outdir, filename))
                    if args.display_samples:
                        display.display(image)
                    index += 1
                args.seed = next_seed(args)

        # make grid of all images, if applicable
        if args.make_grid:
            grid = make_grid(all_images,
                             nrow=int(len(all_images) / args.grid_rows))
            grid = rearrange(grid, 'c h w -> h w c').cpu().numpy()
            filename = f"{args.time}_{iprompt:05d}_grid_{args.seed}.png"
            grid_image = Image.fromarray(grid.astype(np.uint8))
            grid_image.save(os.path.join(args.outdir, filename))
            display.clear_output(wait=True)
            display.display(grid_image)

