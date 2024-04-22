# Set Environments

Execute the following commands:
- conda create --name mrm_env
- conda activate mrm_env
- conda install plotly pandas statsmodels numpy scikit-learn jupyterlab matplotlib seaborn
- conda install jupyter
- pip install yfinance==0.2.38
- pip freeze > requirements.txt
- conda env export > environment.yaml
- conda env create -f environment.yaml

