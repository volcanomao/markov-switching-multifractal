# 马尔可夫切换多重分形 (MSM)

Calvet 和 Fisher (2004, 2008) 的马尔可夫切换多重分形模型 (MSM) 允许对高维状态空间进行简约的规范。在 Collins (2020) 中，我展示了当扩展状态空间、使用高频数据并考虑微观结构噪声时，MSM 的样本外表现得到了改善。我通过对 MSM 的 Python 实现启用了最大似然估计和分析预测，支持多达 13 个波动频率和超过 8,000 个状态，这比以往文献高出八倍，相关代码在本仓库中提供（见 MSM_03.py）。状态空间的扩展需要对 MSM 进行更快的实现。因此，我使用了 Numba，一个即时编译器 (JIT)，将我的一些 Python 函数在运行时转换为优化的机器代码，这一过程通过使用 LLVM 编译器库实现。我还使用了备忘录化，这是一种记录中间结果以避免重复计算的技术，并引入了一种随机算法，将启发式程序与局部搜索相结合，以增强对状态空间的探索，并结合局部优化。通过这种设置，我将与 Calvet 和 Fisher (2013) 的 MATLAB 代码在同一台机器上的基线 Python 实现相比，计算时间从几分钟减少到大约 1 秒（比较基于基准 JPYUSD (DEXJPUS) 日收益数据集，kbar = 4）。在 Collins (2020) 中，我严格准备和清理了高频 (HF) 数据，进行了稀疏采样，计算了按最佳买入和卖出深度加权的收益创新，从而减轻了微观结构噪声。这导致了一个良好规范的模型，更好地利用了大型 HF 数据集提供的增量信息。

样本内模型选择测试显示，随着更多波动成分的引入，模型的单调改进在统计上显著。MSM(13) 与使用 Corsi (2009) 的异质自回归 (HAR) 模型生成的样本外预测的相对准确性进行了比较。MSM(13) 在大多数情况下在 1 小时、1 天和 2 天的时间范围内为股票 HF（苹果和摩根大通）和外汇 HF（EURUSD）收益系列提供了相对更好、统计上显著的预测。MZ 回归显示 MSM(13) 预测几乎没有偏差。这些结果表明，MSM 可能在高频环境中提供一个可行的替代方案，替代已建立的实现波动率估计器。

# 本仓库

本仓库包含我对 MSM 的 Python 实现 (MSM_03.py) 以及一个 Jupyter 笔记本 (见 MSM_03.ipynb)，其中包含基本设置，方便使用。在笔记本中，我对两个数据集应用了 ML 估计。首先，为了建立模型的基准并验证其准确构建，我使用 MSM 框架模拟了一个数据集 (见 MSM_Scripts.py)，并测试模型返回的参数是否接近真实值。第二个数据集 (见 DEXJPUS.csv) 允许复制 Calvet 和 Fisher (2004, 2008) 的结果，使用这些作者的原始数据集之一，从而为后续分析提供了锚点。

如果你是 Python 新手，以下内容可能会有所帮助。

# Python 设置

1. 从 [官方 Python 网站](https://www.python.org/downloads/windows/) 下载 Python 安装包。我更倾向于不使用 [Anaconda](https://www.anaconda.com/)。虽然通过 Anaconda 获取工作设置可能更快，但我发现一旦事情超出中等复杂度，调试会更困难。

2. 下载并安装 [gitbash](https://gitforwindows.org/)。

3. 许多 Python 库需要 Microsoft Visual Studio C++。在没有它的情况下，这些库可能会失败，有时会出现难以理解的错误信息。为了避免后续的麻烦，建议一开始就安装 [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)。

4. 安装一个 Python 脚本编辑器。我喜欢 [Atom](https://atom.io/)。另外两个流行的选择是 [PyCharm](https://www.jetbrains.com/pycharm/) 和 [Sublime Text](https://www.sublimetext.com/)。

5. 创建一个 [Python 虚拟环境](https://www.python.org/dev/peps/pep-0405/)。虚拟环境有自己的 Python 二进制文件（允许创建具有不同 Python 版本的环境），并且可以在其站点目录中拥有自己独立安装的 Python 包，同时仍与基础安装的 Python 共享标准库。

6. 安装 [Jupyter Notebook](https://jupyter.org/) 和/或 [Jupyter Lab](https://jupyter.org/install.html)。我倾向于使用 Lab。

7. 安装以下库。

   ```
   pip install numpy
   pip install pandas
   pip install python-dotenv
   pip install scikit-learn
   pip install matplotlib
   pip install seaborn
   pip install numba
   ```

# Jupyter 笔记本

随附的 Jupyter 笔记本 (见 MSM_03.ipynb) 说明了基本的 MSM 设置，并对两个数据集应用了 ML 估计。为了建立模型的基准并验证其准确构建，我使用 MSM 框架模拟了第一个数据集 (见 MSM_Scripts.py)，并测试模型计算的参数是否接近真实值。第二个数据集 (见 DEXJPUS.csv) 允许使用相同的数据复制 Calvet 和 Fisher (2004, 2008) 的结果，从而为后续分析提供了锚点。

# Python 代码

见 MSM_03.py.
