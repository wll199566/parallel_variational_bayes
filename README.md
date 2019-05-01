# How to run our code

The experiment environment: all experiments were done under `NetID@crunchy1.cims.nyu.edu`, using `python2` and other necessary modules, which are defaultly provided in this machine. Thus, it is possible to run the commands line provided below to reproduce the experiments. 

Since our project is machine learning algorithm, please keep the parameters in the following commands, except `--numthreads`. 

To run single-thread streaming variational Bayes on Wikipedia dataset, please use the following command:

`$ python onlinewikipedia.py --algorithmname=filtering --corpus=wiki --batchsize=1024 --eta=0.01 --max_iters=100 --threshold=1 `

To run distributed, multiprocessing streaming variational Bayes on Wikipedia dataset, please use the following command:

`$ python onlinewikipedia.py --algorithmname=filtering --corpus=wiki --batchsize=1024 --eta=0.01 --max_iters=100 --threshold=1 --numthreads=32`

In our experiments, we tested the performance and execution time for different numbers of threads, where the number can be `{1, 2, 4, 8, 16, 32}`.

Therefore, for `--numthreads=1`, please use the first command line, and for other numbers of threads, please use the second command line with `--numthreads = the number of threads` like the example provided above.

# How to interpret the result

Finally, after the completion of running code, there will be two numbers in a parenthesis, the running time (s) is the first one.  Additionally, there are also two numbers after `held-outerplexity estimate = `. The second one is the value for our measurement of performance. 

If there are several lines of output, then the total running time is the summation of all time mentioned above. And the value for the measurement of performance is the one in the first line.  