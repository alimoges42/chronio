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

**Coming Soon**
 - [ ] Advanced data visualization for exploratory data analysis

## Installing ChronIO
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

## Updating ChronIO
Because ChronIO installation is currently a manual process, updating ChronIO is also manual. There are currently two 
methods through which you may update. The proper (albeit more involved way) is to completely reinstall the ChronIO 
environment through Anaconda. A shortcut also exists, provided the updated version of ChronIO does not contain any 
new packages in the environment.yml file used to create the ChronIO environment. Both approaches are described in 
further detail below.

### Option 1: Reinstall via Anaconda
**Anaconda Navigator:** Simply click on the "Environments" tab, then click on the ChronIO environment, 
and click "Remove". Now skip to the paragraph underneath the **Anaconda Prompt** section below for further
reinstallation instructions.

**Anaconda Prompt:** Type the commands below into an Anaconda Prompt (be sure to run as administrator). 
Once you have done this, be sure to read the paragraph underneath for further reinstallation instructions.
```angular2html
conda activate chronio
pip uninstall chronio
conda deactivate            
conda env remove -n chronio 
```

At this time, regardless of whether you opted for the **Anaconda Prompt** or **Anaconda Navigator** steps, you will 
have deleted the ChronIO Anaconda environment. 
If you initially downloaded ChronIO using `git clone`, you can run `git pull` from inside the folder that you cloned.
Alternatively, if you initially downloaded the zip folder, you should download the new zip folder and unzip, overwriting 
the previously downloaded/unzipped folders. Either of these options will result in downloading the latest 
ChronIO code available on GitHub.
Then repeat the installation process previously described to update your version of ChronIO.


### Option 2: The copy/paste shortcut
An alternative updating option is to copy/paste the updated version of ChronIO into the chronio environment located 
in your Anaconda folder. If you initially downloaded ChronIO using `git clone`, you can run `git pull` from inside the 
folder that you cloned.
Alternatively, if you initially downloaded the zip folder, you should download the new zip folder and unzip, overwriting 
the previously downloaded/unzipped folders. You can identify the location of your installed version of 
ChronIO by running the following commands in a Python interpreter (for example, in a terminal or Jupyter notebook):

```angular2html
import chronio
print(chronio.__file__)
```
This will output something similar to the following:
```angular2html
'c:\\users\\username\\anaconda3\\envs\\chronio\\lib\\site-packages\\chronio\\__init__.py'
```
With this code, open up your desktop file browser and navigate to the `...\\site-packages\\chronio\\` folder above,
making sure not to go deeper than that.
In another window, open up the updated ChronIO folder that you either unzipped or pulled. Simply copy and paste the 
`src\\chronio` folder (and subfolders) to overwrite the `site-packages\\chronio` folder. 
This will copy the lasest version of ChronIO to your Anaconda directory.

### Developer Information
Aaron Limoges (Github: [@alimoges42](https://github.com/alimoges42))
