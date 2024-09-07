import torch
import uuid
from diffusers import FluxPipeline
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from PIL import Image
import io

class FluxImageToImage:
    def __init__(self):
        self.pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        self.pipe.vae.enable_tiling()
        self.pipe.enable_sequential_cpu_offload()
        self.setup_widgets()

    def setup_widgets(self):
        self.html_widget = widgets.HTML(
            value="<i>Made by <a href='https://www.youtube.com/@marat_ai' target='_blank'>marat_ai</a>, "
                  "<a href='https://www.patreon.com/marat_ai' target='_blank'>more_notebooks</a></i>",
            placeholder='Some HTML'
        )

        self.image_upload = widgets.FileUpload(accept='image/*', multiple=False, description='Upload Image')
        self.prompt = widgets.Textarea(value='a cat', description='Prompt', layout=widgets.Layout(width='40%', height='100px'))
        self.strength = widgets.FloatSlider(value=0.8, min=0, max=1, step=0.01, description='Strength')
        self.num_inference_steps = widgets.IntText(value=4, min=1, max=50, description='Steps')
        self.guidance_scale = widgets.FloatText(value=0.0, min=0, max=10, step=0.5, description='SFG scale')
        self.seed = widgets.IntText(value=1, min=0, max=9999999999999999999999999, description='Seed')
        self.random_seed = widgets.Checkbox(value=False, description='Random Seed')
        self.generate_button = widgets.Button(description='Generate', tooltip='Generate image', icon='check')
        self.output = widgets.Output()

        self.generate_button.on_click(self.generate_image)

    def set_generator(self):
        seed_value = 0 if self.random_seed.value else self.seed.value
        return torch.Generator("cpu").manual_seed(seed_value)

    def generate_image(self, button):
        with self.output:
            clear_output(wait=True)
            if not self.image_upload.value:
                print("Please upload an image first.")
                return
            
            try:
                image_file = next(iter(self.image_upload.value.values()))
                init_image = Image.open(io.BytesIO(image_file['content']))
                
                generator = self.set_generator()
                image = self.pipe(
                    prompt=self.prompt.value,
                    image=init_image,
                    strength=self.strength.value,
                    num_inference_steps=self.num_inference_steps.value,
                    guidance_scale=self.guidance_scale.value,
                    generator=generator
                ).images[0]
                
                uid = uuid.uuid4()
                image.save(f"{uid}.png")

                plt.figure(figsize=(10, 10))
                plt.imshow(image)
                plt.axis('off')
                plt.show()
            except Exception as e:
                print(f"An error occurred: {str(e)}")

    def display(self):
        display(self.html_widget, self.image_upload, self.prompt, self.strength, 
                self.num_inference_steps, self.guidance_scale, self.seed, 
                self.random_seed, self.generate_button, self.output)

def setup_flux_image_to_image():
    flux = FluxImageToImage()
    flux.display()
