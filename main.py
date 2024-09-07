import torch
from diffusers import FluxPipeline
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from PIL import Image
import io

def setup_pipeline_and_widgets():
    # Initialize the pipeline for image-to-image
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16)
    pipe.to("cuda")
    pipe.vae.enable_tiling()
    
    clear_output()
    
    def set_generator(random, seed_value):
        return torch.Generator("cuda").manual_seed(0) if random else torch.Generator("cuda").manual_seed(seed_value)

    # Image upload widget
    image_upload = widgets.FileUpload(
        accept='image/*',
        multiple=False,
        description='Upload Image'
    )

    # Prompt input
    prompt = widgets.Textarea(
        value='Transform the image into...',
        description='Prompt',
        disabled=False,
        layout=widgets.Layout(width='50%', height='100px')
    )

    # Number of inference steps
    num_inference_steps = widgets.IntSlider(
        value=4,
        min=1,
        max=50,
        step=1,
        description='Steps',
        disabled=False
    )

    # Seed input
    seed = widgets.IntText(
        value=1,
        min=0,
        max=9999999999,
        step=1,
        description='Seed',
        disabled=False
    )

    # Random seed checkbox
    random_seed = widgets.Checkbox(
        value=False,
        description='Random Seed',
        disabled=False,
        indent=False
    )

    # Generate button
    generate_button = widgets.Button(
        description='Generate',
        disabled=False,
        button_style='',
        tooltip='Generate image',
        icon='check'
    )

    # Output widget for the image
    output = widgets.Output()

    def generate_image(button):
        with output:
            clear_output(wait=True)
            if not image_upload.value:
                print("Please upload an image first.")
                return

            # Debug logging to check upload value
            print("image_upload.value:", image_upload.value)
            
            try:
                uploaded_file = list(image_upload.value.values())[0]
                content = uploaded_file['content']
                input_image = Image.open(io.BytesIO(content))
                generator = set_generator(random_seed.value, seed.value)
                
                result = pipe(
                    prompt=prompt.value,
                    image=input_image,
                    num_inference_steps=num_inference_steps.value,
                    generator=generator
                ).images[0]

                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(input_image)
                plt.title("Input Image")
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(result)
                plt.title("Generated Image")
                plt.axis('off')
                
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"An error occurred: {str(e)}")

    generate_button.on_click(generate_image)

    # Display the widgets and output
    display(widgets.VBox([
        image_upload,
        prompt,
        widgets.HBox([num_inference_steps, seed, random_seed]),
        generate_button,
        output
    ]))

if __name__ == "__main__":
    setup_pipeline_and_widgets()
