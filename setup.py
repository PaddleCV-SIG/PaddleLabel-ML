import pathlib
from setuptools import setup, find_packages, Extension

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text(encoding="utf-8")


with open("requirements.txt", "r") as fin:
    REQUIRED_PACKAGES = fin.read()

setup(
    name="paddlelabel_ml",
    version=open((HERE / "paddlelabel_ml" / "version"), "r").read().strip(),
    description="Web Based Multi Purpose Annotation - ML backend",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/PaddleCV-SIG/PP-Label-ML",
    author="PaddleCV-SIG",
    author_email="me@linhan.email",
    license="Apache Software License",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    packages=find_packages(exclude=("test", "tool")),
    include_package_data=True,
    install_requires=REQUIRED_PACKAGES,
    entry_points={
        "console_scripts": [
            "paddlelabel_ml=paddlelabel_ml.__main__:run",
        ]
    },
)
