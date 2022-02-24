from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="chronio",
    version="0.0.1",
    author="Aaron Limoges",
    author_email="aaronlimoges42@gmail.com",
    description="ChronIO: A toolbox for analysis of physiological time series data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alimoges42/chronio",
    project_urls={
        "Bug Tracker": "https://github.com/alimoges42/chronio/issues",
    },
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Researchers',
        'Topic :: Life Sciences :: Analysis Tools',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GPL-3 License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
)
