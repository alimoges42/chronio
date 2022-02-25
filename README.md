# ChronIO: A toolbox for analysis of physiological time series data

## Overview
ChronIO is a scientific toolbox developed in Python. It enables the analysis of datasets that are sampled 
at a consistent rate. Currently, chronIO works well for simple analyses of behavioral data. In the coming weeks,
additional features will allow more complex analyses to be performed and for more types of data to be analyzed. 
New releases and updates will be available here.

## Features
 ###**Existing**
 - Identify event onsets, durations, and terminations
 - Get the durations of intervals between events
 - Retrieve windows of a defined shape around points of interest in a behavioral dataset

 ###**Coming Soon (~March 2022)**
 - Save your files in a consistent format
 - Support for photometry and calcium imaging data
 - Simple data visualization

 ###**Long-term (Spring 2022 and beyond)**
 - Build templates that model features of your experimental paradigm for easy analyses 
 - Advanced data visualization

## Installation
Right now, Anaconda is needed for installation of this software. To install, first download this software 
(either using `git clone` or by downloading and unzipping the folder). Next, using Anaconda prompt, navigate to 
the location you saved the folder. Type each of the following commands and make sure that each command is executed 
successfully before you move on to the next.

```angular2html
conda env create -f environment.yml -n chronio
conda activate chronio
pip install .
```

_Note: If you wish to develop code for this project, you should replace the final line with `pip install -e .`_

### Developer Information
Aaron Limoges (Github: [@alimoges42](https://github.com/alimoges42))
