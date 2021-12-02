from setuptools import setup, find_packages

from pathlib import Path

__version__ = ''
line = open(Path(__file__).parent/f's3awfs/__version__.py').readline()
exec(line)

def _mltGlob(curDir, *globs):
  curDir = Path(curDir)
  outFiles = []
  for curGlob in globs:
    outFiles.extend(curDir.glob(curGlob))
  outFiles = [str(f) for f in outFiles]
  return outFiles

here = Path(__file__).parent
reqs = here/'requirements.txt'
required = []
if reqs.exists():
  lines = (here/'requirements.txt').open('r').readlines()
  for line in lines:
    if not line.startswith('#'):
      required.append(line.strip('\n'))

# Get the long description from the README file
if (here/'README.md').exists():
  long_description = (here / 'README.md').read_text(encoding='utf-8')
else:
  long_description = ''

setup(
  name='s3a-workflows',
  version=__version__,
  package_dir={'s3awfs':'s3awfs'},
  packages=find_packages(),
  extras_require={
      'full': ['tensorflow>=2.0', 'tensorboard']
  },
  install_requires=required,
  include_package_data=True,
  url='https://gitlab.com/ficsresearch/s3a-workflows',
  download_url='https://gitlab.com/ficsresearch/s3a-workflows',
  license='MIT',
  description='Modular, hierarchical S3A data workflows',
  long_description=long_description,
  long_description_content_type='text/markdown',
  author='Nathan Jessurun',
  author_email='ntjessu@gmail.com',
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which python versions that you want to support
  ],
  python_requires='>=3.8.*',
)
