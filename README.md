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

TFHelpers is tested and developed on a macOS and Linux environment, so some features may not work as
expected on Windows. If you find something that doesn't work properly on Windows, please raise an
issue.

### Installation
Built wheel files are uploaded to the releases page, where the latest stable release is marked
"Latest Release", and development builds are marked "Pre-release".

Download the .whl file for the release marked "Latest Release", and then install the wheel using
pip, providing the path to the downloaded wheel:  
`pip install path/to/TFHelpers-<version>-py3-none-any.whl`

#### Colab notebooks
The following will install the v0.1.0 version on a Colab notebook:  
`!wget https://github.com/jd-13/TFHelpers/releases/download/v0.1.0/TFHelpers-0.1.0-py3-none-any.whl`  
`!pip install TFHelpers-0.1.0-py3-none-any.whl`

#### Pre-release builds (not recommended)
If you wish to use features which are included in a pre-release build but not yet a stable release,
you can install a pre-release wheel by downloading the appropriate pre-release .whl and install
using pip:  
`pip install --upgrade --force-reinstall path/to/TFHelpers-<version>-py3-none-any.whl`

The additional flags `--upgrade --force-reinstall` are required as the wheel's version numbers are not
incremented for pre-release builds, and therefore pip will not attempt to upgrade an existing
TFHelpers installation.

## Contributing
Feedback and contributions are very welcome, if you find a bug or would like to see a feature added
please raise an issue and we can discuss how to resolve it.
