from setuptools import setup, find_packages


setup(
    name="pixeloe",
    packages=find_packages(),
    version="0.0.2",
    url="https://github.com/KohakuBlueleaf/PixelOE",
    description="Detail-Oriented Pixelization based on Contrast-Aware Outline Expansion.",
    author="Shih-Ying Yeh(KohakuBlueLeaf)",
    author_email="apolloyeh0123@gmail.com",
    zip_safe=False,
    install_requires=[
        "opencv-python",
        "numpy",
        "pillow"
    ],
    entry_points={"console_scripts": ["pixeloe.pixelize=pixeloe.cli:main"]},
    python_requires=">=3.10",
)
