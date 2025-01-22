import os
import time
from threading import Thread

import toml
import torch
import gradio as gr
import webview
from PIL import Image

import pixeloe.torch.env as pixeloe_env
from pixeloe.torch.minmax import dilate_cont, erode_cont, KERNELS
from pixeloe.torch.outline import outline_expansion
from pixeloe.torch.pixelize import pixelize
from pixeloe.torch.utils import pre_resize, to_numpy
from pixeloe.logger import logger

pixeloe_env.TORCH_COMPILE = False


base_dir = os.path.dirname(os.path.abspath(__file__))
client_config: dict = toml.load(os.path.join(base_dir, "config.toml"))["client"]
app_mode = client_config.get("use_standalone_window", False)
downsample_mode = [
    "contrast",
    "k_centroid",
    "lanczos",
    "nearest",
    "bilinear",
    "bicubic",
    "area",
]
avaliabe_devices = {"cpu": lambda: None}
if torch.cuda.is_available():
    avaliabe_devices["cuda"] = torch.cuda.empty_cache
if torch._C._mps_is_available():
    avaliabe_devices["mps"] = torch.xpu.empty_cache
if torch.xpu.is_available():
    avaliabe_devices["xpu"] = torch.xpu.empty_cache
device = torch.device(list(avaliabe_devices)[-1])
clean = list(avaliabe_devices.values())[-1]
logger.info(f"Using device: {device}")


def runner_wrapper(func):
    def runner(*args, **kwargs):
        clean()
        result = func(*args, **kwargs)
        clean()
        return result

    return runner


def h_bind(target_h, target_w, img, bind):
    if not bind:
        return target_h, target_w
    W, H = img.size
    ratio = W / H
    target_w = int(round(target_h * ratio))
    return target_w


def w_bind(target_h, target_w, img, bind):
    if not bind:
        return target_h, target_w
    W, H = img.size
    ratio = W / H
    target_h = int(round(target_w / ratio))
    return target_h


@runner_wrapper
def pixelize_image(
    img: Image,
    target_h: int,
    target_w: int,
    patch_size: int,
    thickness: int,
    downsample_mode: str,
    colors: int,
    dither: str,
) -> Image:
    img_t = (
        pre_resize(img, target_size=(target_w, target_h), patch_size=patch_size)
        .to(device)
        .half()
    )
    result_t = pixelize(
        img_t,
        pixel_size=patch_size,
        thickness=thickness,
        mode=downsample_mode,
        do_quant=colors > 0,
        num_colors=colors,
        dither_mode=dither,
    )
    result = Image.fromarray(to_numpy(result_t)[0])
    return result


def pixelization_ui():
    with gr.Row():
        with gr.Column():
            inp = gr.Image(label="Input Image", type="pil")
            submit = gr.Button("Submit")
        with gr.Column():
            result = gr.Image(label="Result Image")
            with gr.Accordion("Settings", open=False), gr.Row():
                with gr.Column(min_width=50, scale=2):
                    with gr.Row():
                        with gr.Column(min_width=25, scale=3):
                            target_w = gr.Number(label="Width", value=256)
                        with gr.Column(min_width=25, scale=3):
                            target_h = gr.Number(label="Height", value=256)
                        with gr.Column(min_width=5, scale=1):
                            bind = gr.Checkbox(label="Bind", value=True)
                    patch_size = gr.Slider(
                        label="Patch Size", minimum=2, maximum=8, step=1, value=4
                    )
                    thickness = gr.Slider(
                        label="Thickness", minimum=0, maximum=6, step=1, value=3
                    )
                with gr.Column(min_width=50, scale=1):
                    down = gr.Dropdown(
                        label="Downsample Mode",
                        choices=downsample_mode,
                        value="contrast",
                    )
                    colors = gr.Number(
                        label="Colors Quantization", value=-1, minimum=-1, maximum=256
                    )
                    dither = gr.Dropdown(
                        label="Dithering Method",
                        choices=["None", "ordered", "error_diffusion"],
                        value="ordered",
                    )
    target_w.input(
        w_bind,
        inputs=[target_h, target_w, inp, bind],
        outputs=target_h,
        trigger_mode="always_last",
    )
    target_h.input(
        h_bind,
        inputs=[target_h, target_w, inp, bind],
        outputs=target_w,
        trigger_mode="always_last",
    )
    submit.click(
        pixelize_image,
        inputs=[inp, target_h, target_w, patch_size, thickness, down, colors, dither],
        outputs=result,
    )


@runner_wrapper
def outline_expansion_image(
    img: Image,
    target_h: int,
    target_w: int,
    patch_size: int,
    thickness: int,
) -> Image:
    img_t = (
        pre_resize(img, target_size=(target_w, target_h), patch_size=patch_size)
        .to(device)
        .half()
    )
    dilate_t = dilate_cont(img_t, KERNELS[thickness].to(img_t))
    dilate = Image.fromarray(to_numpy(dilate_t)[0])
    erode_t = erode_cont(img_t, KERNELS[thickness].to(img_t))
    erode = Image.fromarray(to_numpy(erode_t)[0])
    oe_t, w = outline_expansion(img_t, thickness, thickness, patch_size, 10, 3)
    oe = Image.fromarray(to_numpy(oe_t)[0])
    w = Image.fromarray(to_numpy(w.repeat(1, 3, 1, 1))[0])
    return oe, dilate, erode, w


def outline_expansion_ui():
    with gr.Row():
        with gr.Column():
            inp = gr.Image(label="Input Image", type="pil")
            submit = gr.Button("Submit")
        with gr.Column():
            result = gr.Image(label="Result Image")
            with gr.Accordion("Settings", open=False):
                with gr.Row():
                    with gr.Column(min_width=25, scale=4):
                        target_w = gr.Number(label="Width", value=256)
                    with gr.Column(min_width=25, scale=4):
                        target_h = gr.Number(label="Height", value=256)
                    with gr.Column(min_width=5, scale=1):
                        bind = gr.Checkbox(label="Bind", value=True)
                with gr.Row():
                    with gr.Column(min_width=50):
                        patch_size = gr.Slider(
                            label="Patch Size", minimum=2, maximum=8, step=1, value=4
                        )
                    with gr.Column(min_width=50):
                        thickness = gr.Slider(
                            label="Thickness", minimum=0, maximum=6, step=1, value=3
                        )
    with gr.Row():
        with gr.Column():
            dilate_img = gr.Image(label="Dilated Image")
        with gr.Column():
            erode_img = gr.Image(label="Eroded Image")
        with gr.Column():
            weight_img = gr.Image(label="Weight Map")
    target_w.input(
        w_bind,
        inputs=[target_h, target_w, inp, bind],
        outputs=target_h,
        trigger_mode="always_last",
    )
    target_h.input(
        h_bind,
        inputs=[target_h, target_w, inp, bind],
        outputs=target_w,
        trigger_mode="always_last",
    )
    submit.click(
        outline_expansion_image,
        inputs=[inp, target_h, target_w, patch_size, thickness],
        outputs=[result, dilate_img, erode_img, weight_img],
    )


def change_device(device_name):
    global device, clean
    clean()
    device = device_name
    clean = avaliabe_devices[device]
    logger.info(f"Using device: {device}")


def settings_ui():
    with gr.Row():
        with gr.Column():
            device = gr.Dropdown(
                label="Device",
                choices=avaliabe_devices,
                value=list(avaliabe_devices)[-1],
            )
    device.change(change_device, inputs=[device])


def introduction_ui():
    with gr.Row():
        gr.Markdown(
            """
# PixelOE: Detail-Oriented ***Pixel***ization based on Contrast-Aware ***O***utline ***E***xpansion.
**Create stunning pixel art from high-resolution images without AI or complex networks.**
"""
        )


def ui():
    with gr.Blocks(
        title="PixelOE",
        theme=gr.themes.Soft(),
        css=open(os.path.join(base_dir, "client.css"), "r", encoding="utf-8").read(),
    ) as website:
        with gr.Tabs(elem_id="main-tabs"):
            with gr.Tab("Pixelization", elem_classes="page-tab"):
                pixelization_ui()
            with gr.Tab("Outline Expansion", elem_classes="page-tab"):
                outline_expansion_ui()
            with gr.Tab("Glogal Settings", elem_classes="page-tab"):
                settings_ui()
            with gr.Tab("Introduction", elem_classes="page-tab"):
                introduction_ui()
    return website


if __name__ == "__main__":
    website = ui()
    gr_thread = Thread(
        target=website.launch,
        daemon=True,
        kwargs={"inbrowser": False, "pwa": app_mode},
    )
    gr_thread.start()
    while not website.local_url:
        time.sleep(0.01)
    if app_mode:
        webview.create_window(
            "PixelOE",
            website.local_url + "?__theme=dark",
            resizable=True,
            zoomable=True,
        )
        webview.start()
    else:
        input("Press Enter to close gradio")
