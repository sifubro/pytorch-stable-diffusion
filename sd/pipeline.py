import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512 # SD can only produce images of 512x512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8  # size of latent tensor for the VAE i.e. 64x64
LATENTS_HEIGHT = HEIGHT // 8

# Do text-to-image and image-to-image
def generate(
    prompt,
    uncond_prompt=None, # Most of the time empty string. It's also called negative prompt (e.g. we want picture of cat but we DON'T want the cat to be on the sofa. Just place "sofa" to go away from this concept)
    input_image=None,  # for image-to-image
    strength=0.8, # related to input image. How much attention we want to pay to the initial starting image (or how much noise we want to add to it. Higher strength means higher noise hence less resemblance to input image)
    do_cfg=True,  # do classifier-free-guidance (2 outputs with and without the prompt)
    cfg_scale=7.5,  # how much attention to pay attention to our prompt. It is between [1,14] 
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},  # pretrained models
    seed=None, #to initialize RNG
    device=None,
    idle_device=None, # if we load model on cuda and we don't longer need we move it to the CPU
    tokenizer=None,
):
    with torch.no_grad(): # we are inferencing the model
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")

        if idle_device: 
            to_idle = lambda x: x.to(idle_device)  # to move to CPU
        else:
            to_idle = lambda x: x

        # Initialize random number generator according to the seed specified
        generator = torch.Generator(device=device) # to generate noise
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        # CLIP
        clip = models["clip"]
        clip.to(device)
        
        # with classifier-free-guidance we inference the model twice. Once with the prompt and the next time without it.
        if do_cfg:
            # Convert prompt into tokens. Convert into a list of length Seq_Len=77
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim). Converted to embeddings from CLIP
            cond_context = clip(cond_tokens)
            # Convert into a list of length Seq_Len=77
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            uncond_context = clip(uncond_tokens)
            # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (2 * Batch_Size, Seq_Len, Dim)
            # e.g. (2, 77, 768)
            context = torch.cat([cond_context, uncond_context])
        else:
            # Convert into a list of length Seq_Len=77
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            # e.g. (1, 77, 768)
            context = clip(tokens)


        to_idle(clip) # move the models to CPU after using them to offload them from GPU

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps) # how many steps to do for inference
            # For training it was 1000 steps but during inference we can do less (e.g. 50)
            # DDIM useusllay 20 steps and others with diff equations even less.
        else:
            raise ValueError("Unknown sampler value %s. ")

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH) # (1, 4, 64, 64)

        # If user specifies input image for image-to-image
        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT)) #(512, 512)
            # (Height, Width, Channel)
            input_image_tensor = np.array(input_image_tensor)
            # (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
            # (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1)) # Rescale between [-1, +1] from [0, 255] because the UNET wants the input to be between [-1, +1]
            # (Height, Width, Channel) -> (Batch_Size, Height, Width, Channel)
            input_image_tensor = input_image_tensor.unsqueeze(0) # add batch dimension
            # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2) # the encoder of VAE wants (B, Channel, Height, Width)

            # We need some noise for the VAE encoder (to be added at x= mean + stdev * noise)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = encoder(input_image_tensor, encoder_noise) # This latents would be (B, 4, 64, 64)
             

            # Add noise to the latents (the encoded input image)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            sampler.set_strength(strength=strength) # How much attention we want to pay to the initial starting image when generating the output image#
            # Higher strength means higher noise, hence the model will be more creative
            # by setting the strength the sampler will define a timestep schedule. If noise level stnregth 1.0 then max noise level. If 0.5 half noise level etc..
            latents = sampler.add_noise(latents, sampler.timesteps[0])  # add noise to the latents

            to_idle(encoder)
        else:
            # For text-to-image just sample random noise (we don't five input an image)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = torch.randn(latents_shape, generator=generator, device=device)

        # this is our UNET
        diffusion = models["diffusion"]
        diffusion.to(device)

        # Sampler defines timesteps. Each timestemp indicates a noise level
        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps): # for each of the timestemps we denoise the image
            # (1, 320)
            # It equal to the positional encoding for transformers model
            time_embedding = get_time_embedding(timestep).to(device)

            # (Batch_Size, 4, Latents_Height, Latents_Width) = (B, 4, 64, 64)
            model_input = latents

            # If doing classifier-free-guidance, send the conditional and unconditional input as prompt
            # Hence send the same latent twice (one to be used with conditional prompt and the other with the unconditional prompt)
            if do_cfg:
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
                model_input = model_input.repeat(2, 1, 1, 1)

            # model_output is the predicted noise by the UNET
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            model_output = diffusion(model_input, context, time_embedding)

            
            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                # combine the conditional and unconditional output
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # After UNET predicted the noise at the current timestemp, we need to remove that noise
            # Remove noise predicted by the UNET
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        # Load decoder to upscale the denoised latent image
        decoder = models["decoder"]
        decoder.to(device)
        # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
        images = decoder(latents)
        to_idle(decoder)

        # rescale from (-1,1) -> (0, 255)
        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
        # We want the channel dimension to be the last one to save to CPU
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]
    
def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    # torch.Size([1, 1]) * torch.Size([1, 160]) --> x= Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None] # [:, None] -> Adds additional dimension like unsqueeze
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
