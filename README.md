[![Build Status](https://travis-ci.org/jd-13/TFHelpers.svg?branch=master)](https://travis-ci.org/jd-13/TFHelpers)
[![codecov](https://codecov.io/gh/jd-13/TFHelpers/branch/master/graph/badge.svg)](https://codecov.io/gh/jd-13/TFHelpers)


# TFHelpers
Small collection of classes which implement common tasks in Tensorflow.  

Documentation is available [here](https://jd-13.github.io/TFHelpers/)

## Setup
### Requirements
* tensorflow
* scikit-learn
* numpy

### Installation
Built wheel files are uploaded to the releases page, where the latest stable release is marked
"Latest Release", and development builds are marked "Pre-release".

Download the .whl file for the release marked "Latest Release", and then install the wheel using
pip, providing the path to the downloaded wheel:  
`pip install path/to/TFHelpers-<version>-py3-none-any.whl`

### Colab notebooks
The following will install the v0.0.1 version on a Colab notebook:  
`!wget https://github.com/jd-13/TFHelpers/releases/download/v0.0.1/TFHelpers-0.0.1-py3-none-any.whl`  
`!pip install TFHelpers-0.0.1-py3-none-any.whl`

#### Pre-release builds (not recommended)
If you wish to use features which are included in a pre-release build but not yet a stable release,
you can install a pre-release wheel by downloading the appropriate pre-release .whl and install
using pip:  
`pip install --upgrade --force-reinstall path/to/TFHelpers-<version>-py3-none-any.whl`

The additional flags `--upgrade --force-reinstall` are required as the wheel's version numbers are not
incremented for pre-release builds, and therefore pip will not attempt to upgrade an existing
TFHelpers installation.
