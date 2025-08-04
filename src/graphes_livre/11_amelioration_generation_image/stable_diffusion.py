from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-2")

prompt = "An astronaut riding a horse in a tunnel, with the horse's hooves splashing in the water. The astronaut is waving a French flag."
image = pipe(prompt).images[0]
image.save("11A_astronaut_horse-stable-diffusion-v1-2.png")
