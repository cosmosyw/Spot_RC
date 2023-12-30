from setuptools import setup, find_packages

setup(
    name='DNA_spot_finder',
    version='0.1',
    description="DNA MERFISH spot finder",
    author="Cosmos Wang",
    author_email="cosmosyw93@gmail.com",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'DNA_spot_finder = DNA_spot_finder.DNA_spot_finder:DNA_spot_finder'
        ]
    },
)