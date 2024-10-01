from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


install_requires = [
    'altair',
    'bokeh',
    'dask-jobqueue',
    'folium',
    'geopandas',
    'matplotlib',
    'numpy',
    'odc-stac',
    'pandas',
    'planetary-computer',
    'pystac-client',
    'rioxarray',
    'setuptools',
    'tqdm',
    'xarray',
    'zarr',
    ]


setup(name='stacathome', 
        version='0.0.1',
        description="stacathome",
        author="Markus Zehner",
        author_email="mzehner@bgc-jena.mpg.de",
        url="",
        long_description=long_description,
        long_description_content_type="text/markdown",
        classifiers=[
                "Intended Audience :: Science/Research",
                "License :: OSI Approved :: MIT License",
                "Programming Language :: Python :: 3"
                 ],
        packages=['stacathome'],
        install_requires=install_requires,
        )