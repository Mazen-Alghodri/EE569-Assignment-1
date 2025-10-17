This folder contains the solution for EE569 Assignment 1 part a
LinearClass.py implements a single computation node that performs the linear function y=A.x+b for task 1.
The file contains sigmoid , BCE to use in task 2 , and it also contains MSE for testing.
LogisticRegressionNoBatching.py implement the LinearClass into the 03_logistic_regression.py from @Nuri-benbarka repository. for task 2.
The LogisticRegressionWithBatching.py introduces batching to the previous code.
the LinearClass.py backprop function was modified during this implementation to handle batch gradient.
at the end of the LogisticRegressionWithBatching.py we test  multiple batch sizes and plot their effect on training loss.
This code only uses NumPy, Matplotlib, and SciPy, in accordance with the assignment requirements.
