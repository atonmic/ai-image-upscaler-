import gradio as gr
from upscaler import upscale_image

def process_image(image, model_name, scale):
    try:
        output_image = upscale_image(image, model_choice=model_name, scale=scale)
        return [image, output_image]
    except Exception as e:
        return gr.update(value=f"❌ Error: {str(e)}"), None

with gr.Blocks(title="AI Image Upscaler") as demo:
    gr.Markdown(
        """
        # 🖼️ AI Image Upscaler  
        Enhance your images using RealESRGAN, GFPGAN, or Anime upscaling.  
        Just upload, choose model & scale, and see the magic ✨
        """
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Image", type="pil")
            model_select = gr.Dropdown(
                label="Select Model",
                choices=["RealESRGAN", "GFPGAN+RealESRGAN", "Anime"],
                value="RealESRGAN"
            )
            scale_select = gr.Radio(
                label="Upscale Factor",
                choices=[2, 4],
                value=4
            )
            submit_btn = gr.Button("🔄 Upscale Image")

        with gr.Column():
            output_gallery = gr.Gallery(label="Original vs Upscaled", columns=2, height=400)

    submit_btn.click(
        fn=process_image,
        inputs=[image_input, model_select, scale_select],
        outputs=output_gallery
    )

demo.launch()