<div style="background-color:white">
  <div align="center">
    <img src="./imgs/techfak_logo.jpg" width="700" height="250">
    <hr>
    <h1 style="color:black">Identifying a Trial Population for Clinical Studies on Diabetes Drug Testing with Neural Networks<h1>
    <h3 style="color:black">Tim Löhr<h3>
    <img src="./imgs/madi_logo.png" width="400">
  </div>
  <hr>
</div>

## Motivation
This project models an end-to-end workflow of implementing AI for the clinical environment. A possible use-case such as the selection of patients for a novel treatment or drug will be conducted, by estimating the hospitalization time with a Tensorflow regression Neural Network. Using a synthetic dataset from the UCI Diabetes readmission dataset, the expected days for a person being hospitalized after certain conditions or treatments will be predicted. This result is used to decide whether a patient is applicable to be included in the clinical trial. If so, there needs to be a clear explanation of the prediction and possible risk factors. This project shows the importance of splitting the data appropriately without data leakage and evaluating the results to make it transparent for the official use case, e.g. being accepted by the Arzneimittelbeho ̈rde or FDA as a decision support tool for hospitals or doctors.


### Structure

```

+-- Code
|   +-- Notebooks                        
|   |    +-- Clinical_EDA.ipynb
|   |    +-- Machine_Learning.ipynb
|   |    +-- Explainable_AI.ipynb
|   +-- Scripts                        
|   |    +-- model_preprocessing_utils.py
|   |    +-- tensorflow_modeling.py
|   |    +-- utils.py
|   +-- Source                      
|   |    +-- __init__.py
|   |    +-- main.py
|   +-- Tests             
|        +-- test_main.py    
+-- Paper
|   +-- Final Paper
|   +-- Related Work Paper
|   +-- Bibliography.bib
|
+-- Presentation
|   +-- Mid-Term Presentation
|   +-- Final Presentation
|   
+-- imgs                    
+-- requirements.txt                    
+-- README.md
+-- .gitignore              

```
## Links to Ressources

- Clinical_EDA as [iPython](https://github.com/Mavengence/Machine_Learning_in_the_Industry_4.0_Seminar_SS2020.FAU/blob/master/Code/Notebooks/Clinical_EDA.ipynb)
- Machine_Learning as [iPython](https://github.com/Mavengence/Machine_Learning_in_the_Industry_4.0_Seminar_SS2020.FAU/blob/master/Code/Notebooks/Machine_Learning.ipynb)
- Explainable_AI as [iPython](https://github.com/Mavengence/Machine_Learning_in_the_Industry_4.0_Seminar_SS2020.FAU/blob/master/Code/Notebooks/Explainable_AI.ipynb)
- Final Presentation as [PDF](https://github.com/Mavengence/Machine_Learning_in_the_Industry_4.0_Seminar_SS2020.FAU/blob/master/Presentation/L%C3%B6hr_Tim_MADI40SS20_final_presentation.pdf)
- Final Paper as [PDF](https://github.com/Mavengence/Machine_Learning_in_the_Industry_4.0_Seminar_SS2020.FAU/blob/master/Paper/L%C3%B6hr_Tim_MADI40SS20_paper.pdf)

## Ressources

- Coding Example: https://towardsdatascience.com/machine-learning-for-diabetes-562dd7df4d42
- Mapping Data: https://www.accessdata.fda.gov/scripts/cder/ndc/index.cfm
- Model building: https://www.tensorflow.org/tutorials
- Model building with Layers: https://blog.tensorflow.org/2019/03/regression-with-probabilistic-layers-in.html
- Evaluation: https://www.sciencedirect.com/science/article/pii/S1877050916323870
- Dataset UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008

### Prerequisites

```
The dependencies to this project are stored in the file:
   - requirements.txt

I use python version 3.7.4
```

## Author

* **Tim Löhr** - If you have questions you can contact me under timloehr@icloud.com

## License

This project was done during my Seminar Machine Learning in the Industry 4.0 from the Machine Learning and Data Analytics Lab at the Friedrich Alexander University in Erlangen-Nürnberg.

## Acknowledgments

* Thanks a lot to Philipp Schlieper from the Machine Learning and Data Analytics Lab for a really good supervising through all my project. I can totally recommend this seminar!
