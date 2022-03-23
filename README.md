# ChronIO: A toolbox for analysis of physiological time series data

## Overview
ChronIO is a scientific toolbox developed in Python. It enables the analysis of physiological datasets that are sampled 
at a consistent rate. For example, it may be used for analysis of behavioral and neural data. Currently, chronIO works 
well for simple analyses of behavioral data. In the coming weeks, additional features will allow more complex 
analyses to be performed and for more types of data to be analyzed. 
New releases and updates will be available here.

## Features
**Existing**
 - [x] Identify event onsets, durations, and terminations
 - [x] Get the durations of intervals between events
 - [x] Retrieve windows of a defined shape around points of interest in a behavioral dataset
 - [x] Save your files in a consistent format
 - [x] Support for photometry and calcium imaging data
 - [x] Build templates that model features of your experimental paradigm for easy analyses

**Coming Soon (Spring 2022)**
 - [ ] Advanced data visualization for exploratory data analysis

## Installation
Right now, Anaconda is needed for installation of this software. To install, first download this software 
(either using `git clone` or by downloading and unzipping the folder). Next, using Anaconda Prompt, navigate to 
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
