# MonoTuned_RNNs
# TimingGenerationNeuralNets
Not yet published in journal: _Tuned responses spontaneously emerge in recurrent neural network models during timing generation_ 
(Evi Hendrikx, Daniel Manns, Nathan van der Stoep, Alberto Testolin, Marco Zorzi, Ben M. Harvey)
This pipeline was started by Daniel Manns and further adapted and developed by Evi Hendrikx

Preprint available: https://doi.org/10.1101/2024.08.29.610320

Scripts will likely be made a bit more legible before publication, but the core will remain the same.

Independent recurrent neural networks (indRNN, Li et al., 2018) are trained on a generative timing task: predict the next frame in a movie with a repeating event. This frame only exists of a single pixel. In an event pixels can be on (1) and off (0). An example movie could look like: 1 1 0 0 0 1 1 1 1 0 0 0 1 1 1 1 0 0 0

## Running the pipeline
In order to run everything: run "main.py". 
Movies with regular temporal intervals are created using "generate_dataset.py".
Networks are built using "network_models_new.py" and trained and evaluated using "pipeline.py".

## Evaluating accuracy
Accuracy is compared between different network depths (1-5 layers) in "accuracy_stats.py"

## Evaluating response functions in the network nodes
Monotonic and tuned functions are fitted on the per-event activations of the network nodes in "model_fitting.py".
Fits are compared between different network depths and different model layers within the same network in "model_fitting_stats.py"

## Evaluating response function properties
Properties of the response functions are started in "parameter_stats.py" 

## Requirements to run
provided are my version of the Anaconda installation and required packages for Linux ("networks_anaconda_packages.yml")
