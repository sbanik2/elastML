

<a name="readme-top"></a>

![Size][size-shield]

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


<p align="center"> <a href="url"><img src="https://github.com/sbanik2/elastML/blob/main/figs/workflow.png?raw=true" align="center" height="500" width="700" ></a> </p>


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
conda env create --name elastML
conda activate CASTING
git clone https://github.com/sbanik2/elastML.git
pip install -r requirements.txt
python setup.py install
```


<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Running the code
The code can be used to generate feature space as  described in the ([Paper](https://doi.org/10.26434/chemrxiv-2023-07vcr)), for the prediction of bulk and shear modulus. To compute the initial feature relevance using mRMR use the ([notebook](https://github.com/sbanik2/elastML/blob/main/notebooks/mrmr.ipynb)).

To predict either bulk or shear modulus using the trained modes, 1st download the trained models from 

```
https://doi.org/10.5281/zenodo.8067989
```

## Citation
```
@article{banik2022continuous,
  title={A Continuous Action Space Tree search for INverse desiGn (CASTING) Framework for Materials Discovery},
  author={Banik, Suvo and Loefller, Troy and Manna, Sukriti and Srinivasan, Srilok and Darancet, Pierre and Chan, Henry and Hexemer, Alexander and Sankaranarayanan, Subramanian KRS},
  journal={arXiv preprint arXiv:2212.12106},
  year={2022}
}
```
    
<p align="right">(<a href="#readme-top">back to top</a>)</p>
        
## License
CASTING is distributed under MIT License. See `LICENSE` for details.
    
    
<p align="right">(<a href="#readme-top">back to top</a>)</p>  
    
<!--LINKS -->



[size-shield]: https://img.shields.io/github/repo-size/sbanik2/elastML
[DOI-shield]: https://img.shields.io/badge/Paper-8A2BE2
[DOI-url]: https://doi.org/10.26434/chemrxiv-2023-07vcr
    
    
