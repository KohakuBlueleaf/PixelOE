# PixelOE: Transform Images into Stunning Pixel Art

Welcome to PixelOE, a powerful tool that generates high-quality pixel art from standard images using an innovative contrast-aware approach. Unlike other pixelization methods, PixelOE preserves crucial visual details through intelligent outline expansion and adaptive downscaling - all without requiring AI or complex neural networks.

<br>

## Key Features

Transform your images with these powerful capabilities:

<br>

### Detail-Oriented Processing
PixelOE uses a novel contrast-aware approach that identifies and preserves important visual elements. By expanding outlines before downscaling, the algorithm ensures that fine details remain crisp and recognizable in the final pixel art.

<br>

### Smart Downsampling Options
Choose from multiple downsampling methods to achieve your desired artistic style:
- Contrast-Aware: Intelligently preserves important luminance details
- K-Centroid: Creates clean, representative colors for each pixel region
- Classic Options: Including bicubic, bilinear, and nearest-neighbor methods

<br>

### Color Enhancement
Fine-tune your pixel art with advanced color processing:
- Palette Optimization: Reduce colors while maintaining visual quality
- Dithering Options: Apply ordered or error diffusion dithering
- Color Matching: Preserve the original image's color characteristics

<br>

### High Performance
- Fast Processing: Utilizes PyTorch for efficient computation
- GPU Support: Leverage hardware acceleration when available
- Batch Processing: Handle multiple images efficiently

<br>
## Getting Started

1. Choose the "Pixelization" tab to start transforming your images
2. Upload an image using the input panel
3. Adjust settings to achieve your desired style:
   - Resolution controls
   - Patch size for detail level
   - Outline thickness
   - Downsampling method
   - Color quantization options
4. Click "Submit" to generate your pixel art

<br>

## Need More Control?

Visit the "Outline Expansion" tab to fine-tune the detail preservation process and visualize the intermediate steps of the algorithm, including dilation, erosion, and weight map generation.

---

<br>

Created by Shih-Ying Yeh (KohakuBlueleaf)  
Licensed under Apache License 2.0