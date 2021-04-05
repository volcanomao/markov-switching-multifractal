# 00 - Get Started



Different views pertain re editors, settings, etc. If you have not used Python before, the following describes a procedure that has worked well for me.

### I. Setting up a virtual environment to run Python and the MSM

1. Download the Python install package from [the official python website](https://www.python.org/downloads/windows/). Be sure that the version is Python 3.0.0 or above, and check the requirement of the [tensorflow library.](https://www.tensorflow.org/install) -- I prefer not to use a third party provider like [Anaconda](https://www.anaconda.com/). Although it is faster to get the popular IDEs via Anaconda, I have found it to be more difficult to debug once things get moderately comoplex.

   > **Important: check box at bottom "Add Python 3.7 to PATH'**

2. Install [gitbash](https://gitforwindows.org/)

3. (Optional) A lot of Python libraries require that you install Microsoft Visual Studio C++. These Python libraries may fail, sometimes with unhelpful error messages, in the abscence of it. IMHO, save some pain later and install it at the outset [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

4. (Optional) Gitbash's console is great. But another console that I like is  [cmder](https://cmder.net/). It supports more customization, and split screen features.

5. (Choose your own) You will need a Python script editor on your computer. I like [PyCharm](https://www.jetbrains.com/pycharm/). Other popular choices are [Sublime Text](https://www.sublimetext.com/), and [Atom](https://atom.io/).

6. Open the console (using gitbash/cmder/etc.), **with administration privilege**. 

7. Check your Python version and upgrade `pip` to the latest version by typing the following commands in the console.

   ```bash
   python -V
   pip install --upgrade pip
   ```

8. Install virtual environment.

   ```bash
   pip install virtualenv
   ```



### II. Setting up a new project.



1. Tip: Create a virtual environment. This will ensure that the packages you need to install are properly documented. It will also prevent packages from contaminating your original python installation.

   ```
   virtualenv .venv
   ```

2. Activate the virtual environment.

   ```
   .venv\Scripts\activate
   ```

3. (Optional) Install cookiecutter, this will help you to set up a standard repository structure.

   ```
   pip install cookiecutter
   ```

4. Navigate to your project directory, and create the folder structure using cookiecutter. (See [details](https://drivendata.github.io/cookiecutter-data-science/) for guidance re the default file structure.)

   ```
   cookiecutter https://github.com/drivendata/cookiecutter-data-science
   ```

   Directory structure:

   ```
   ├── LICENSE
   ├── Makefile           <- Makefile with commands like `make data` or `make train`
   ├── README.md          <- The top-level README for developers using this project.
   ├── data
   │   ├── external       <- Data from third party sources.
   │   ├── interim        <- Intermediate data that has been transformed.
   │   ├── processed      <- The final, canonical data sets for modeling.
   │   └── raw            <- The original, immutable data dump.
   │
   ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
   │
   ├── models             <- Trained and serialized models, model predictions, or model summaries
   │
   ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
   │                         the creator's initials, and a short `-` delimited description, e.g.
   │                         `1.0-jqp-initial-data-exploration`.
   │
   ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
   │
   ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
   │   └── figures        <- Generated graphics and figures to be used in reporting
   │
   ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
   │                         generated with `pip freeze > requirements.txt`
   │
   ├── setup.py           <- Make this project pip installable with `pip install -e`
   ├── src                <- Source code for use in this project.
   │   ├── __init__.py    <- Makes src a Python module
   │   │
   │   ├── data           <- Scripts to download or generate data
   │   │   └── make_dataset.py
   │   │
   │   ├── features       <- Scripts to turn raw data into features for modeling
   │   │   └── build_features.py
   │   │
   │   ├── models         <- Scripts to train models and then use trained models to make
   │   │   │                 predictions
   │   │   ├── predict_model.py
   │   │   └── train_model.py
   │   │
   │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
   │       └── visualize.py
   │
   └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
   ```

   

5. Install jupyter notebook and jupyter lab. I tend to use lab.

   ```
   pip install jupyter
   pip install jupyterlab
   ```

6. Install the following machine learning libraries.

   ```
   pip install numpy
   pip install pandas
   pip install python-dotenv
   pip install scikit-learn
   pip install scipy
   pip install matplotlib
   pip install seaborn
   pip install tensorflow
   pip install cupy
   pip install numba
   ```

