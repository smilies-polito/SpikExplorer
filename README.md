## REFERENCES

1. Hyperparameters Optimization:
	* https://en.wikipedia.org/wiki/Hyperparameter_optimization
	* https://neptune.ai/blog/hyperparameter-tuning-in-python-complete-guide
	* https://www.tomasbeuzen.com/deep-learning-with-pytorch/chapters/chapter6_cnns-pt2.html

2. Network Architecture Search: 
	* https://www.oreilly.com/content/what-is-neural-architecture-search/
	* https://dl.acm.org/doi/10.5555/2023011.2023014
	* https://en.wikipedia.org/wiki/Neural_architecture_search
	* https://www.automl.org/nas-overview/

3. Bayesan Optimization:
	* https://arxiv.org/abs/1910.11858 									%% BANANAS
	* https://research.facebook.com/blog/2021/07/optimizing-model-accuracy-and-latency-using-bayesian-multi-objective-neural-architecture-search/
	* https://arxiv.org/pdf/2109.10964.pdf								%% Multi-Objective High Dimensional Bayesian Optimization
	* https://botorch.org/docs/botorch_and_ax#when-not-to-use-ax		%% Design Spaces belongs to the cases in which is not optimal to use Ax, vague

4. Tools
	* AX: https://ax.dev/docs/why-ax.html
		%% Ax substantially simplifies the usage of BoTorch and does it through 3 different APIs, the one of interest is Dev API, the most customizable

5. Estimate energy consumption:
	* https://arxiv.org/pdf/2210.13107.pdf
	* https://arxiv.org/pdf/2210.01625.pdf
	* https://www.frontiersin.org/articles/10.3389/fnins.2020.00662/full
## TO DO

1. State-of-the-art review: start to write the first part of the thesis:
	* Overview of hyperparameter optimization, NAS, meta-learning etc.
	* Overview of the methods: how they work? Which are the PROs? Which are the CONs?
	* Available tools

2. Describe the constraints for the optimization:
	* Latency
	* Power
	* Area
	* Architecture: maximum number of neurons, synapses
	* Bit-width
	
	How can we describe them? How can we formalize them? Should we consider quantization at this level?

3. Keep track of whatever you think can be useful, like tools, papers, websites, books, blogs, whichever thing, here. Keep this readme updated.
