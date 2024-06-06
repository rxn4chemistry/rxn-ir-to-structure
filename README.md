# IR to Structure
Scripts of the paper: Automatic structure elucidation from IR spectra


<p align='center'>
  <img src='figure/Graphical Abstract v3.png' width="1000px">
</p>

The dataset generated in this paper and on which all models were trained is available at: [10.5281/zenodo.7928396](https://doi.org/10.5281/zenodo.7928396)


## Installation guide

Install the package via Poetry:

```
poetry install
```

EMC can be downloaded from [here](https://sourceforge.net/projects/montecarlo/). Version 9.4.4 was used for all simulations. To setup EMC, add both `bin` and `scripts` to your PATH by running the commands below. Set `EMC_ROOT` environment variable to the path of the EMC folder (e.g. `~/emc/v9.4.4`):
```
export EMC_ROOT=<path to EMC folder>
export PATH=$PATH:$EMC_ROOT/bin:$EMC_ROOT/scripts
```

## Generating IR spectra

This section explains how to generate IR spectra using a molecular dynamics pipeline. The code requires both EMC and LAMMPS.

### Running the simulation pipeline

Running the pipeline requires a csv file with two columns: 'Indices' and 'Smiles'. The pipeline will create a separate folder for each row and runs the simulation for a given smiles in this folder. The name of the folder is a vale as specified in the column 'Indices'. This allows you to correlate a SMILES with a simulation folder once all simulations have finished. If the column 'Indices' is not present the script will use the number of the row.

Both the `emc_template.esh` and the `lammps_template.in` can be found in `templates/`.

```
poetry run run_md --smiles_csv <smiles_csv> --output_folder <output_folder> --field <force field> --steps <steps> --cores <cores> --emc_template <path to emc_template.esh> --lammps_template <path to lammps_template.in>
```
To test the script, run the following command with the `example.csv` provided:
```
poetry run run_md --smiles_csv examples/smiles.csv --output_folder examples/out --steps 50000 --emc_template templates/emc_template.esh --lammps_template templates/lammps_template.in
```
Note: For reliable spectra, do not run production runs with 50,000 steps; 500,000 is more appropriate.

### Gathering the data

Once the simulation pipeline has finished, you can gather the data using the `gather_data` script. Provide the script with the `simulation_folder`, the `smiles_path` (path to the smiles csv used for the simulations) and where to save the data (`output_folder`, saved as a pickled dataframe ready to train with).

For the example run as follows:

```
poetry run gather_data --simulation_folder examples/out --smiles_path examples/smiles.csv --output_folder examples/
```


## Training a model
This section covers preparing the data for the model, training the model, running inference and scoring the output of the model.

### Preparing the data

The `prepare_data` script creates the input data from the simulated data. It requires as input a pickled dataframe with the columns: 'formula', 'spectra' and 'smiles'. The spectra is expected as a 1D Numpy array with a range from 400-3982cm<sup>-1</sup> and a resolution of 2cm<sup>-1</sup> (i.e. 1792 values). 

Parameters of the script:
```
--data_path: Path to the pickled dataframe
--output_path: Path where the processed data is saved
--n_tokens: With how many tokens the spectrum is encoded (default 400)
--window: What window of the IR spectrum is used, Options: full: Full spectrum, fingerprint: 400-2000<sup>-1</sup>, 
umir: 400-3982<sup>-1</sup> and merged: 400-2000<sup>-1</sup> with 2800-3200<sup>-1</sup>
--augmentation: Augmentation options, Options: N/A, smooth, shift_horizontal, noise_vertical, all (only smooth 
and shift_horizontal)
--special: Special data preparation, Options: N/A, Formula: only the formula, Spectrum: only the spectrum
```

Running the script to generate training data from the example:

```
poetry run prepare_data --data_path examples/ir.pkl --output_path examples/train
```

### Training

Run a training of the model using the `run_training` script. This script requires a directory prepared by `prepare_data` with a subfolder called `data` which contains the training, validation and test data. Another requirement is a path to a OpenNMT configuration template. The template with which all trainings were performed in the template is provided in `templates/transformer_template.yaml`.

For the example try:
```
poetry run run_training --template_path templates/transformer_template.yaml --data_folder examples/train
```
Note: The template expects a GPU to be present and to run a real training much more data is required.

For testing purposes, training a model for only few steps with a tiny model on CPU only:
```
poetry run run_training --template_path templates/transformer_template_tiny_cpu.yaml --data_folder examples/train
```

### Inference

Run the following command to do inference:

```
onmt_translate -model <model_path> -src <src_path> -output <out_file> -beam_size 10 -n_best 10 -min_length 5 -gpu 0
```

For the example (dummy model, valid set, CPU only), this corresponds to the following:
```
onmt_translate -model examples/train/model_step_5.pt -src examples/train/data/src-val.txt -output examples/train/pred-val.txt -beam_size 10 -n_best 10 -min_length 5
```

### Scoring

Scoring can be done via the `score.py` script:

```
poetry run score --tgt_path <path to tgt file> --inference_path <out file from inference> 
```

For the example above:
```
poetry run score --tgt_path examples/train/data/tgt-val.txt --inference_path examples/train/pred-val.txt
```

Use `--n_beams` if the beam size is changed from 10 during inference.
