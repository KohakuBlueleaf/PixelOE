from setuptools import setup, find_packages
from pixeloe import cli


setup(
    name="pixeloe",
    packages=find_packages(),
    version="0.0.8",
    url="https://github.com/KohakuBlueleaf/PixelOE",
    description="Detail-Oriented Pixelization based on Contrast-Aware Outline Expansion.",
    license="Apache License 2.0",
    author="Shih-Ying Yeh(KohakuBlueLeaf)",
    author_email="apolloyeh0123@gmail.com",
    zip_safe=False,
    install_requires=["opencv-python", "numpy", "pillow"],
    entry_points={
        "console_scripts": [
            f"pixeloe.{command}=pixeloe.cli:{func}"
            for command, func in cli.command_map.items()
        ]
    },
    python_requires=">=3.10",
)
