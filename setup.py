from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")


setup(
    name="elastML",
    version="0.1.0",  
    description="Benchmarking Machine Learning Models for Predicting Elastic Properties of Materials",
    author="Suvo Banik",
    author_email="sbanik2@uic.edu", 
    packages=find_packages(),
    install_requires=[
                "dscribe==2.0.0",
                "joblib==1.2.0",
                "matminer==0.8.0",
                "numpy==1.23.4",
                "pandas==1.5.3",
                "pymatgen==2023.5.31",
                "scikit_learn==1.1.1",
                "tqdm>=4.64.0",
                "polars==0.18.3",
        ],
    
    
    scripts =[
        "elastML/features.py",
        "elastML/predict.py",
        "elastML/utils.py",
 
    ],
        
    package_data={
        'elastML.Bulk_models': [ 'model_*'],
        'elastML.Shear_models': [ 'model_*'],
        'elastML.transform': [ 'Transform_*',"Imputer"],
        'elastML.mrmr_relevance': [ '*.csv'] 
        
    },
    include_package_data=True,
    
    
    classifiers=[

    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.10',
    
    ],

    
    long_description=long_description,  
    long_description_content_type="text/markdown",
    url="https://github.com/sbanik2/elastML",  
    python_requires=">=3.9",

)

