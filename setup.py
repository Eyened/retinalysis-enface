from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="rtnls_enface",
    version="0.1.0",
    author="Jose Vargas",
    author_email="j.vargasquiros@erasmusmc.nl",
    description="Utilities for analyzing retinal enface images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    entry_points={"console_scripts": []},
    install_requires=[
        "numpy == 1.*",
        "pandas == 2.*",
        "scikit-learn==1.*",
        "scipy == 1.*",
        "opencv-python == 4.*",
        "matplotlib==3.*",
        "joblib==1.*",
        "Pillow == 9.*",
        "click==8.*",
        "pydicom==2.3.1",
    ],
    python_requires=">=3.9",
)
