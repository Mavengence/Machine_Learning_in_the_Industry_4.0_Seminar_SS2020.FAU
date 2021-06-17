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

## Abstract
This project aims to model an end-to-end workflow of implementing Artificial Intelligence (AI) for the clinical environment. A possible use-case such as the selection of patients for a novel treatment or drug will be conducted by estimating the hospitalization time with a Neural Network.
The diabetes readmission dataset from the University of California, Irvine (UCI) Diabetes was used for this project. The trial population is selected by predicting the expected days for a person being hospitalized. Then and arbitrary boundary is set for chosing whether or not this patient is shall be included or not. If so, a clear explanation of the how the prediction was calculated and additional possible risk factors will be given in order to make the workflow explainable. This project shows that given a proper explanatory approach, AI can be a useful tool for the modern clinical environment. The workflow finally reveals that AI can be a beneficial support tool for doctors, e.g. by effectively choose possibly suitable patients in the patient selection process.


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

This project was done during my Seminar Machine Learning in the Industry 4.0 from the Machine Learning and Data Analytics Lab at the Friedrich Alexander University in Erlangen-Nürnberg. Some parts of the code are under the licence of www.udacity.com. Those parts can mostly be found in the scripts section.

## Acknowledgments

* Thanks a lot to Philipp Schlieper from the Machine Learning and Data Analytics Lab for a really good supervising through all my project. I can totally recommend this seminar!
