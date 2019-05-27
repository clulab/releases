# Lightly Supervised Representation Learning with Global Interpretability

This repository contains the data and code required to run Emboot.

## Installation notes for Emboot
- install `miniconda`
- add `miniconda` to your PATH
- `conda create -n emboot python=3 tensorflow` to create a conda environment with python3 and tensorflow
- `conda install scikit-learn matplotlib`
- Install keras: `pip install keras`
- In file `~/.keras/keras.json`, change line to `"floatx":"float64"`


## Running Emboot
- To run the basic Emboot model, run `python emboot_classifier.py` from the `emboot` directory.
- To run the interpretable Emboot model, run `python emboot_classifier_interpretable.py` from the `emboot` directory.
- To run the edited Emboot model, run `python emboot_classifier_interpretable_load_vectors.py` from the `emboot` directory.
	- In order to run this, you must have run the interpretable model first AND have a list of edited patterns in the right format. This list of patterns should follow the format of `pools_output.txt_patterns.txt` that is produced when running the other models, but without the epoch numbers.
- All of these will use the CoNLL dataset. To use the OntoNotes dataset, uncomment the lines `102-106` and comment out lines `111-116` in the relevant file.


## Converting the interpretable model for plotting
- Running the interpretable model will produce an interpretable model file (`pools_output_interpretable.txt_interpretable_model.txt`) that needs converted before it can be plotted alongside the other models.
- To convert the file, from the main release directory run `sbt run`. Then, select the number that will run `EmbootInterpretOutputProcessor.scala`.


## Plotting results
- To plot a single model, broken down into each category, run `python plot_results.py RESULTS OUTPUT`, where `RESULTS` is the Embood motel, and `OUTPUT` is the name of the file where you want the plot saved.
- To compare more than one model, run `python plot_results_multitrials_overallplot.py GOLD # MODEL1 MODEL2 ...`, where `GOLD` is the gold labelled data (in our case this is in `data/CONLL/entity_label_counts_emboot.filtered.txt` or `data/Ontonotes/entity_label_counts_emboot.filtered.txt`, depending on the dataset), `#` is the number of models to compare, `MODEL1` is the output of the first model, `MODEL2` the second model, etc.