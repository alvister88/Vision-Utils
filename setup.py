from setuptools import setup, find_packages
import codecs
import os.path


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name='visionml_utils',
    version='0.0.0',
    description='Util scripts and files for computer vision deep learning model training.',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    author='Alvin Zhu',
    license='MIT',
    project_urls={'GitHub':'https://github.com/alvister88/Vision-Utils'},
    packages=find_packages(include=['visionml_utils', 'visionml_utils.*']),
    install_requires=requirements,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Framework :: Robot Framework',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
    ],
)

