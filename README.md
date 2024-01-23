# Relevance Forward Propagation (RFP)

This repository contains the code for the paper <i>Efficient Data Source Relevance Quantification for Multi-Source Neural Networks</i> ([LINK](https:link_goes_here)).
<br><br>
The work proposes a Forward Relevance Propation approach to compute source-wise relevevances values within the neural network forward pass. 
<p align=center>
<img src="https://user-images.githubusercontent.com/77287533/228695768-bc673b4a-a540-4963-8604-189350671204.png" alt="teaser2" width="600">
</p>

## Prerequisites and Setup
This repository has been tested under ```Python 3.9.12``` in a *unix* development environment. <br> 
For a setup, clone the repository and ``cd``to the root of it. <br>
Create a new environment and activate it, for instance, on unix via
```
python -m venv venv && source venv/bin/activate
```
Then install the needed packages via the ```setup.py```:
```
pip install --upgrade pip
python setup.py install
```

## Content
The folder code contains Jupyter Notebooks to reproduce the experiments and to explore the approach on the Multi-Source MNIST data set. 
* <b>MNIST Playground</b><br>
The notebook ```code/01_MNISTPlayGround.ipynb``` provides a pipeline to explore the different Multi-Source MNSIT setups, train networks and evaluate the relevance values. 
* <b>Regression Example</b><br>
The notebook ```code/02_RegressionExample.ipynb``` provides an example of the Forward Relevance Propagation applied to a regression task.  

* <b>Evaluation MNIST</b><br>
The notebook ```code/11_Evaluate_MNIST.ipynb``` provides an example of the Forward Relevance Propagation applied to a regression task.  

* <b> Evaluation SEN12MS </b><br>
The notebook ```code/12_Evaluate_SEN12MS.ipynb``` provides a pipeline to apply Relevance Forward Propagation to pre-trained networks provided by the authors of the data set.<br>
The pretrained networks can be found [here](https://github.com/schmitt-muc/SEN12MS/blob/master/classification/linkToPreTrainedModels.txt).<br>
The original (clear) can be found [here](https://mediatum.ub.tum.de/1474000).<br>
The cloudy Version can be found [here](https://mediatum.ub.tum.de/1554803) and [here](https://patricktum.github.io/cloud_removal/sen12mscr/).<br>
We provide checkpoints with intermediate computation steps: ([link_goes_here](https:link_goes_here))


## Citation

```
[1] Names, Names, (2024). Efficient Data Source Relevance Quantification for Multi-Source Neural Networks. TBA.
```

BibTeX format:
```
@article{name2024efficient,
  title={Efficient Data Source Relevance Quantification for Multi-Source Neural Networks},
  author={Names Names},
  journal={TBA},
  year={2023}
}
```
