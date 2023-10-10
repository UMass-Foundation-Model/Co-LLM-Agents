import re
from pathlib import Path
from setuptools import setup, find_packages

readme = Path('README.md').read_text(encoding='utf-8')
# Replace relative markdown links with absolute https links.
readme = re.sub(r'\[(.*?)\]\(doc/(.*?)\)', r'[\1][https://github.com/alters-mit/transport_challenge_multi_agent/blob/main/doc/\2]', readme)

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='transport_challenge_multi_agent',
    version="0.2.2",
    description='High-level multi-agent Transport Challenge API for TDW.',
    long_description=readme,
    long_description_content_type='text/markdown',
    url='https://github.com/alters-mit/magnebot',
    author='Esther Alter',
    author_email="alters@mit.edu",
    keywords='unity simulation tdw robotics agents',
    packages=find_packages(),
    include_package_data=True,
    install_requires=required,
)
