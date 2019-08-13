from setuptools import setup, find_packages

setup(name='pixel_world',
      version='1.3.0',
      install_requires=['gym'],  # And any other dependencies foo needs,
      include_package_data=True,
      packages=find_packages()
)  
