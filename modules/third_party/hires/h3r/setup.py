import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="torchalign-BAOSHENG-YU", # Replace with your own username
    version="0.0.1",
    author="Baosheng Yu",
    author_email="baosheng.yu.usyd@gmail.com",
    description="Facial landmark detection.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/baoshengyu/HRRR/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)