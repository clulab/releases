This is the resouce described in the LREC 2018 paper: 
"Grounding Gradable Adjectives through Crowdsourcing"

If you have any questions or issues, please contact Becky Sharp
(bsharp@email.arizona.edu).

The resource consists of four files, each corresponding to one version of the 
gradable adjective groundings:

 -- the full set of adjectives (98) with both mean (mu) and stdev (sigma)
 -- the full set of adjectives with only mean
 -- the high-frequency subset (30)  with both mean and stdev
 -- the high-frequency subset with only mean

The first line of each file is the header that describes the data organization.

Usage example:
 - if you have an known item (e.g., mean rainfall = 40 in/yr, stdev = 3in) and 
you want to know the impact of a gradable adjective (i.e., a *small* increase):
	1) choose a model 
		(ex: full model)
	2) find the row corresponding to the desired adjective
		(ex: small	1.034e-05	-0.001123	-1.7094)
	3) This row gives the linear model, so plug in your known mean 
	   (and stdev, if applicable):
		(ex: logDeviations = -1.7094 + (1.034e-05*40) + (-0.001123*3))
	4) Convert to the true predicted change:
		(ex: predChange = (e^(logDeviations) * stdev))
		(here, because it's an increase, this would then be added to 
		 the mean).

A python demo script is included.  To use:
> python demo_gradable.py
And follow the prompts.

DATA and CODE:
We include the original (unfiltered) data from the crowd-sourcing experiment.
	data/AdjMainR.csv
We additionally include the R code to reproduce all analyses and plots, though
please note that the code is not an end-to-end script, but rather a set of commands.
Also, paths to data will likely need to be modified depending on where/how you
store the released data.

Predictions of the NN model on both seen and unseen adjectives are included in data/.
The code for evaluating these models as well as for generating the input used
to make the MSE vs. Variance plot for unseen adjectives is included in 
	code/nn_eval_and_plot.py

We will soon release the code for the neural network model as well as all prompts
given to participants in the crowd-sourcing task.  Please contact 
Becky Sharp (bsharp@email.arizona.edu) for more information in the meantime.
 


