from setuptools import setup, find_packages

print(find_packages())

setup(
    name='spp',
    version='0.1.2',
    packages=find_packages(),
    install_requires=[
        'pydub',
        'librosa',
    ],
)