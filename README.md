

<a name="readme-top"></a>

![Size][size-shield]
[![notebooks][notebooks-shield]][notebooks-url]
[![Doi][DOI-shield]][DOI-url]





# elastML

<p align="justify"> Evaluating Generalized Feature Importance via Performance Assessment of Machine Learning Models for Predicting Elastic Properties of Materials</p>


## Table of Contents
- [Introduction](#Introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the code](#running-the-code)
- [Citation](#citation)
- [License](#license)





## Introduction

Featurization of crystal structures for the prediction of elastic modulus (Bulk and Shear) using different ML models ([Paper](https://doi.org/10.26434/chemrxiv-2023-07vcr)). The code contains 8 different trained ML models on ~10K materials' project crystal structures, comparing their overall performance. A generalized task-specific feature importance is therefore determined from the feature importance learned by each model. <br/>

<p align="justify">&emsp;&emsp;&emsp;&emsp;</p>


<p align="center"> <a href="url"><img src="https://github.com/sbanik2/elastML/blob/main/figs/workflow.png?raw=true" align="center" height="450" width="700" ></a> </p>


<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Prerequisites
This package requires:
- [dscribe](https://singroup.github.io/dscribe/latest/)
- [matminer](https://matminer.readthedocs.io/en/latest/)
- [pymatgen](https://pymatgen.org/)
- [scikit_learn](https://scikit-learn.org/stable/)
- [mrmr-selection](https://github.com/smazzanti/mrmr)


<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Installation

[Install the anaconda package](https://docs.anaconda.com/anaconda/install/). Then, 

```
conda create --name elastML python=3.10
conda activate elastML
```

```
git clone https://github.com/sbanik2/elastML.git
pip install -r requirements.txt
python setup.py install
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Running the code
The code can be used to generate the feature space as described in the [Paper](https://doi.org/10.26434/chemrxiv-2023-07vcr), or the prediction of bulk and shear modulus. To compute the initial feature relevance using mRMR, please refer to the [notebook](https://github.com/sbanik2/elastML/blob/main/notebooks/mrmr.ipynb).

To predict either bulk or shear modulus using the trained models, download them from:

```
https://doi.org/10.5281/zenodo.8067989
```
n example of predicting the elastic modulus using the trained models can be found in the [notebook](https://github.com/sbanik2/elastML/blob/main/notebooks/prediction.ipynb).

## Citation
```
@article{banik2023machine,
  title={Machine Learning for Elastic Properties of Materials: A predictive benchmarking study in a domain-segmented feature Space},
  author={Banik, Suvo and Balasubramanian, Karthik and Manna, Sukriti and Derrible, Sybil and Sankaranarayananan, Subramanian},
  year={2023}
}
```
    
<p align="right">(<a href="#readme-top">back to top</a>)</p>
        
## License
elastML is distributed under MIT License. See `LICENSE` for details.
    
    
<p align="right">(<a href="#readme-top">back to top</a>)</p>  
    
<!--LINKS -->



[size-shield]: https://img.shields.io/github/repo-size/sbanik2/elastML
[DOI-shield]: https://img.shields.io/badge/Paper-8A2BE2
[DOI-url]: https://doi.org/10.26434/chemrxiv-2023-07vcr
[notebooks-shield]: https://img.shields.io/badge/notebooks-2ECC71
[notebooks-url]: https://github.com/sbanik2/elastML/tree/main/notebooks
    
    
