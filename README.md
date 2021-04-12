# Background

The Markov-switching multifractal stochastic volatility model (MSM) of Calvet & Fisher (2004, 2008)) permits the parsimonious specification of a high-dimensional state space. In Collins (2020) I showed that MSM's out-of-sample performance improved when the state space was expanded, high-frequency data used, and microstructure noise taken into account. I enabled maximum likelihood estimation and analytical forecasting with up to 13 volatility frequencies and over 8,000 states, some eight times higher than previous literature, using a Python implementation of MSM, the code for which is presented in this repo (see MSM_03.py). 

This Python implementation of MSM introduced a stochastic algorithm that combined heuristic procedures with local searches to perform an enhanced exploration of the state space in conjunction with local optimization. In my work, rigorous preparation and cleansing of data, sparse sampling, and return innovations weighted by the respective depth of the best bid and ask, mitigated microstructure noise. These developments resulted in a well-specified model, better able to use the increased information provided to it by large high frequency (HF) datasets. In-sample model selection tests showed statistically significant monotonic improvement in the model as more volatility components were introduced. MSM(13) was compared to the relative accuracy of out-of-sample forecasts produced by a realized volatility measure using the heterogeneous autoregressive (HAR) model of Corsi (2009). MSM(13) provided comparatively better, statistically significant, forecasts than HAR most of the time at 1-hour, 1-day, and 2-day horizons for equity HF (Apple and J.P.Morgan) and foreign exchange HF (EURUSD) returns series. MZ regressions showed little sign of bias in the MSM(13) forecasts. These results suggest MSM may provide a viable alternative to established realized volatility estimators in high-frequency settings.



# Setting up a virtual environment

Different views pertain re editors, settings, etc. If you have not used Python before, the following describes a procedure that has worked well for me.

1. Download the Python install package from [the official python website](https://www.python.org/downloads/windows/). Be sure that the version is Python 3.0.0 or above, and check the requirement of the [tensorflow library.](https://www.tensorflow.org/install) I prefer not to use a third party provider like [Anaconda](https://www.anaconda.com/). Although it is faster to get the popular IDEs via Anaconda, I have found it to be more difficult to debug once things progress beyond moderately complex.

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

9. Tip: Create a virtual environment. This will ensure that the packages you need to install are properly documented. It will also prevent packages from contaminating your original python installation.

   ```
   virtualenv .venv
   ```

2. Activate the virtual environment.

   ```
   .venv\Scripts\activate
   ```

3. Install jupyter notebook and jupyter lab. I tend to use lab.

   ```
   pip install jupyter
   pip install jupyterlab
   ```

4. Install the following machine learning libraries.

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

# Accompanying Jupyter notebook

The accompanying Jupyter notebook (see MSM_03.ipynb) applies ML estimation to two datasets. To establish a benchmark for the the model and validate that it is accurately constructed, I simulate the first dataset using the MSM framework, and test that the model computes parameters that are acceptably close to true.  The second dataset (see DEXJPUS.csv) allows replication of the results of Calvet & Fisher (2004, 2008) with the same data, and thus provides an anchor for subsequent analysis.

# MSM Python code

See MSM_03.py.



