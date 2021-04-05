# 00 - Get Started



There are a bunch of recommendations on the internet. Different people can have different preferences for editors, settings etc. The following describes the procedure of what worked well for me. Feel free to take this as a reference and add your personal touch to it.

### I. Setting up a new machine for data science

1. Download python install package from [the official python website](https://www.python.org/downloads/windows/). Make sure that the version is Python 3.0.0 or above, and check the requirement of the [tensorflow library.](https://www.tensorflow.org/install) I prefer not to install python from third party providers like [Anaconda](https://www.anaconda.com/). Although it is faster to get all popular IDEs from anaconda, it is often more difficult to debug through third party app when something go wrong.

   > **Important: check box at bottom "Add Python 3.7 to PATH'**

2. Install [gitbash](https://gitforwindows.org/)

3. (Optional) A lot of data analytics python libraries require you to install the Microsoft Visual Studio C++ command line tools to be installed. Unfortunately, when you try to install these python libraries, they often fail with unhelpful error messages. So better to install it right from the beginning! [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

4. (Optional) Gitbash's console is is definitely good, another console that I like is  [cmder](https://cmder.net/). It supports a lot more customization, and split screen features.

5. (Choose your own) You will need a python script editor on your computer. By far I think [PyCharm](https://www.jetbrains.com/pycharm/) works best. Other popular choices are [Sublime Text](https://www.sublimetext.com/), [Atom](https://atom.io/).

6. Open the console (using gitbash/cmder/anaconda prompt etc.), **with administration privilege**. If it is not opened with administration privilege, it might fail to install some python packages.

7. Check python version and upgrade `pip` to the latest version by typing the following commands in the console.

   ```bash
   python -V
   pip install --upgrade pip
   ```

8. Install virtual environment.

   ```bash
   pip install virtualenv
   ```



### II. Setting up a new data science project.



1. Create a virtual environment for each project. This ensure that every packages to be installed are properly documented. It also prevents the project packages from contaminating your original python installation. Sometimes if you are working on several projects, and they require different versions of the same package, virtual environment ensures that they are separately and properly installed. Common names for the virtual environment are `venv`, `.venv`, `.venv_project_name`. If you create the virtual environment inside your project folder, make sure that they are excluded from git using the `.gitignore` file.

   ```
   virtualenv .venv
   ```

2. Activate the virtual environment.

   ```
   .venv\Scripts\activate
   ```

3. Install cookiecutter, this helps you to set up a standard repository structure for a data science project.

   ```
   pip install cookiecutter
   ```

4. Navigate to your project directory, and create the folder structure using cookiecutter. (Please take a look at the [details](https://drivendata.github.io/cookiecutter-data-science/) on how to use the default file structure.)

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

   

5. Install jupyter notebook and jupyter lab. Whether to use jupyter notebook or jupyter lab, it is completely a personal preference.

   ```
   pip install jupyter
   pip install jupyterlab
   ```

6. Install some common machine learning libraries.

   ```
   pip install numpy
   pip install pandas
   pip install python-dotenv
   pip install scikit-learn
   pip install matplotlib
   pip install seaborn
   pip install tensorflow
   pip install pandas-profiling[notebook]
   pip install progressbar2
   ```

7. (Optional) Install the following if the project involves text processing.

   ```
   pip install nltk
   pip install Unidecode
   pip install pycontractions
   pip install gensim
   ```

8. (Optional) Install [jupyter notebook extensions](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/install.html) to advance your editor.

   ```
   pip install jupyter_contrib_nbextensions
   jupyter contrib nbextension install --user
   ```

   

9. (Under development) Make sure that you have access to [FTI's repository](https://gitlab.com/fti-ai-team/99-code-library) Install the FTI library by using the following command:

   ```
   pip install git+ssh://git@gitlab.com/fti-ai-team/99-code-library.git#egg=ftids
   ```

10. When you finish you work, deactivate the virtual environment.

    ```
    deactivate
    ```

    



### III. Jupyter Lab / Jupyter Notebook

We provide you with a handy notebook template to begin with:

![notebook_screenshot](https://gitlab.com/fti-ai-team/00-get-started/-/raw/master/notebook_screeshot.PNG)



### Explanation of different files in the repository



FTI_Logo_cmyk.eps: Official logo from marketing

FTI_Logo_cmyk.png: png version of the official logo

FTI_Logo_RGB.eps: Official logo from marketing

FTI_Logo_RGB.png: png version of the official logo



