# Background

The Markov-switching multifractal stochastic volatility model (MSM) of Calvet & Fisher (2004, 2008)) permits the parsimonious specification of a high-dimensional state space. In Collins (2020) I showed that MSM's out-of-sample performance improved when the state space was expanded, high-frequency data used, and microstructure noise taken into account. I enabled maximum likelihood estimation and analytical forecasting with up to 13 volatility frequencies and over 8,000 states, some eight times higher than previous literature, using a Python implementation of MSM, the code for which is presented in this repo (see MSM_03.py). To expand the state space in this way required a significantly faster implementation of MSM.  This Python implementation therefore used Numba, a just-in-time (JIT) compiler, to translate some of my Python functions to optimized machine code at runtime, in turn achieved via use of LLVM compiler libraries. It also used memoization, a technique for recording intermediate results to avoid repeated calculations, and introduced a stochastic algorithm that combined heuristic procedures with local searches to perform an enhanced exploration of the state space in conjunction with local optimization. This set up reduced the compute time versus a baseline Python implementation of the MATLAB code of Calvet & Fisher (2013) on the same machine from some minutes to just under 1 second (comparison based upon the benchmark JPYUSD (DEXJPUS) daily returns dataset with kbar = 4). In my work, rigorous preparation and cleansing of data, sparse sampling, and return innovations weighted by the respective depth of the best bid and ask, mitigated microstructure noise. These developments resulted in a well-specified model, better able to use the increased information provided to it by large high frequency (HF) datasets. 

In-sample model selection tests showed statistically significant monotonic improvement in the model as more volatility components were introduced. MSM(13) was compared to the relative accuracy of out-of-sample forecasts produced by a realized volatility measure using the heterogeneous autoregressive (HAR) model of Corsi (2009). MSM(13) provided comparatively better, statistically significant, forecasts than HAR most of the time at 1-hour, 1-day, and 2-day horizons for equity HF (Apple and J.P.Morgan) and foreign exchange HF (EURUSD) returns series. MZ regressions showed little sign of bias in the MSM(13) forecasts. These results suggest MSM may provide a viable alternative to established realized volatility estimators in high-frequency settings.

# This repo

This repo contains my Python implementation of MSM (MSM_03.py) along with a Jupyter notebook (see MSM_03.ipynb) containing a basic set up, for convenience.  In the notebook I apply ML estimation to two datasets.  Firstly, to establish a benchmark for the the model and validate that it is accurately constructed, I simulate a dataset (see MSM_Scripts.py) using the MSM framework, and test that the model returns parameters that are acceptably close to true.  The second dataset (see DEXJPUS.csv) allows replication of the results of Calvet & Fisher (2004, 2008) with one of these author's original datasets, and thus provides an anchor for subsequent analysis.

If you are new to Python, the following may be helpful.

# Python set up

1. Download a Python install package from [the official python website](https://www.python.org/downloads/windows/). I prefer not to use [Anaconda](https://www.anaconda.com/). Although it may be faster to get a working set up via Anaconda, I have found it to be more difficult to debug once things progress beyond moderately complex.

2. Download and install [gitbash](https://gitforwindows.org/).

3. Many Python libraries require Microsoft Visual Studio C++. These libraries may fail, with somewhat impenetrable error messages at times, in the abscence of it. IMHO save some pain later and install it at the outset [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

4. Install a Python script editor.  I like [Atom](https://atom.io/).  Two other popular options are [PyCharm](https://www.jetbrains.com/pycharm/) and [Sublime Text](https://www.sublimetext.com/).

5. Create a [Python virtual environment](https://www.python.org/dev/peps/pep-0405/).  A virtual environment has its own Python binary (allowing creation of environments with various Python versions) and can have its own independent set of installed Python packages in its site directories, whilst still sharing the standard library with the base installed Python.

6. Install [Jupyter Notebook](https://jupyter.org/) and/or [Jupyter Lab](https://jupyter.org/install.html). I tend to use Lab.

7. Install the following libraries.

   ```
   pip install numpy
   pip install pandas
   pip install python-dotenv
   pip install scikit-learn
   pip install scipy
   pip install matplotlib
   pip install seaborn
   pip install cupy
   pip install numba
   ```

# Jupyter notebook

The accompanying Jupyter notebook (see MSM_03.ipynb) illustrates a basic MSM set up and applies ML estimation to two datasets. To establish a benchmark for the the model and validate that it is accurately constructed, I simulate the first dataset (see MSM_Scripts.py) using the MSM framework, and test that the model computes parameters that are acceptably close to true.  The second dataset (see DEXJPUS.csv) allows replication of the results of Calvet & Fisher (2004, 2008) with the same data, and thus provides an anchor for subsequent analysis.

# Python code

See MSM_03.py.



