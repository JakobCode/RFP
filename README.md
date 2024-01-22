# Relevance Forward Propagation (RFP)

This repository contains the evaluation script for the FusionResNet18 model on the SEN12MS dataset. The evaluation script generates accuracy plots, classwise data source relevance values, and summaries for given models. 

<p align=center>
<img src="https://user-images.githubusercontent.com/77287533/228695768-bc673b4a-a540-4963-8604-189350671204.png" alt="teaser2" width="600">
</p>

## Prerequisites
This repository has been tested under ```Python 3.9.12``` in a *unix* development environment. <br> 
For a setup, clone the repository and ``cd``to the root of it. <br>
Create a new environment and activate it, for instance, on unix via
```
python -m venv venv && source venv/bin/activate
```
Then install the needed packages via:
```
pip install --upgrade pip
pip install -r requirements.txt
```

## Evaluation of pre-trained FusionResNet18 models
To use the pretrained model, 
* download the pre-trained models (see `pretrained_models/pretrained_models.txt`),
* download the SEN12MSCR data set (and adjust the root path in `cfg.py`),
* run the `evaluation.py` script in the command line:

```
python evaluation.py -ckpt <path_to_pretrained_FusionResNet18_checkpoint> [-cloud_mode <cloudy|clear>] [-print_id <int>]
```


The script takes in three optional command line arguments:

* `-ckpt`: Path to pretrained FusionResNet18 checkpoint.
* `-cloud_mode`: Mode of cloud images to evaluate. Choices are "cloudy" and "clear". Default is "clear".
* `-print_id`: If set to an integer value, only the specified id will be evaluated. If not set, the entire dataset will be evaluated.


## Outputs

The script generates the following output files and saves them in the `Figures` folder located in the same directory as the checkpoint file:
- Accuracy plots: Saved in the `Accuracy` subfolder.
- Classwise relevance values: Saved in the `Relevance` subfolder.
- Summaries: Saved in the `Summaries` subfolder (only generated when `-print_id` is set to a specific sample ID).


## Example Calls

To evaluate the entire dataset of cloudy images with the default checkpoint, run the following command:

```
python evaluate.py -cloud_mode "cloudy"
```


To evaluate only the image with id 123 with a custom checkpoint, run the following command:

```
python evaluate.py -ckpt "/path/to/checkpoint.pth" -cloud_mode "cloudy" -print_id 123
```

## Example Visualizations

Relevance of the optical and the SAR data source for the <i>forest</i> and the <i>urban</i> class for a network trained with cross-entropy loss and a network with the source-wise relevance loss. 
<p align=center>
<img src="https://user-images.githubusercontent.com/77287533/228696931-320d9662-4a11-4536-b63c-9f9fbb6a7224.png" alt="density_plot" width="900">
</p>


Example images and corresponding relevance scores for a <i>forest</i> sample and an <i>urban</i> sample for a network trained with cross-entropy loss. 
<p align=center>
<img src="https://user-images.githubusercontent.com/77287533/228696710-405c9d12-fbac-427d-83f1-da7b717ee506.png" alt="Sample_567_class_0" width="460"> <img src="https://user-images.githubusercontent.com/77287533/228696712-500eb650-09e4-4ddc-95f0-166d68e5f5ee.png" alt="Sample_2256_class_4" width="460">
</p>

## Citation

```
[1] Gawlikowski, J., (2023). Efficient Data Source Relevance Quantification
and Application to SAR-Optical Fusion Networks. TBA.
```

BibTeX format:
```
@article{gawlikowski2023efficient,
  title={Efficient Data Source Relevance Quantification and Application to SAR-Optical Fusion Networks},
  author={Jakob Gawlikowski},
  journal={TBA},
  year={2023}
}
```
