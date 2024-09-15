import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="yolo-detector",
    version="1.0.0",
    author="Justmore5mins",
    author_email="austinyu0607@gmail.com",
    description="a simple yolo detector",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Justmore5mins/YOLO-Detector",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License : MIT License",
        "Operating System :: OS Independent",
    ],
)
