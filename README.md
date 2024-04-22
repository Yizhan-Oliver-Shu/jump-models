# Statistical Jump Models (JMs)

**Developer**: Yizhan Shu   ([@Yizhan-Oliver-Shu](https://github.com/Yizhan-Oliver-Shu))
Email: yizhans@princeton.edu

**Author**: Yizhan Shu, Onat Aydınhan ([Linkedin](https://www.linkedin.com/in/af%C5%9Far-onat-ayd%C4%B1nhan-ph-d-940ba7142/)) and Chenyu Yu [@chenyu-yu](https://github.com/chenyu-yu).

**Credit**: This project builds upon the open-source [code](https://www.sciencedirect.com/science/article/pii/S0957417421009647#appSB) from the paper by Nystrup et al. (2021).

---

This repository, named `jumpmodels`, offers a comprehensive implementation of statistical jump models (JMs) for non-parametric regime identification in financial time series. It includes the discrete jump model, continuous statistical jump models, and their sparse variants, designed for efficient feature selection. Each model is encapsulated within a class, adopting a coding style reminiscent of the popular [scikit-learn](https://github.com/scikit-learn/scikit-learn) library.

The `jumpmodels/` folder contains all the source code.  An example application focusing on the Nasdaq Composite Price Index series, featuring a simple feature set, is located in the `notebooks/nasdaq_example` directory.

I regret to inform you that complete documentation is not available at the moment. However, the code includes detailed docstrings, and the source code itself serves as the most comprehensive form of documentation for now.


## Dependencies (version in our environment)

- `Python (== 3.12.2)`
- `NumPy (== 1.26.4)` 
- `Pandas (== 2.2.1)`
- `scikit-learn (== 1.4.1.post1)`
- `matplotlib (== 3.8.3)`
- `hmmlearn (== 0.3.2)`
- `scipy (== 1.12.0)`
- `yfinance (== 0.2.38)`, only used in data retrieval for the example notebook.


---

## References

- **Continuous Statistical Jump Models**: Aydınhan, A. O., Kolm, P. N., Mulvey, J. M., and Shu, Y. (2024). Identifying patterns in financial markets: Extending the statistical jump model for regime identification. *Annals of Operations Research*. To appear. [Link](https://papers.ssrn.com/abstract=4556048)

- **Sparse Jump Models**: Nystrup, P., Kolm, P. N., and Lindström, E. (2021). Feature selection in jump models.  *Expert Systems with Applications*, 184:115558.  [Link](https://www.sciencedirect.com/science/article/pii/S0957417421009647)

- **Out-of-sample Online Prediction for JMs**:  Nystrup, P., Kolm, P. N., and Lindström, E. (2020). Greedy online classification of persistent market states using realized intraday volatility features. *The Journal of Financial Data Science*, 2(3):25–39. [Link](https://www.pm-research.com/content/iijjfds/2/3/25)

- **Original Jump Models**: Nystrup, P., Lindström, E., and Madsen, H. (2020). Learning hidden Markov models with persistent states by penalizing jumps. *Expert Systems with Applications*, 150:113307. [Link](https://www.sciencedirect.com/science/article/abs/pii/S0957417420301329)
