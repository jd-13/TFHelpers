from setuptools import setup

setup(
    name="TFHelpers",
    version='0.0.1',
    description="Small collection of classes which implement common tasks in Tensorflow",
    #long_description=long_description,
    url="https://github.com/jd-13/TFHelpers",
    author="White Elephant Audio",
    author_email="jack@whiteelephantaudio.com",
    classifiers=[
        "Development Status :: 4 - Beta",

        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords="tensorflow machine-learning",
    packages=["TFHelpers"],
    install_requires=["tensorflow", "sklearn", "numpy"],
)
