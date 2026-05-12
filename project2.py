import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import datetime as dt
from ISLP.models import \
(ModelSpec as MS, 
summarize, poly)
from ISLP import load_data
from ISLP.bart import BART
Auto = load_data('Auto')
from ISLP import confusion_table
from ISLP.models import contrast
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm
import sklearn.model_selection as skm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import sklearn.model_selection as skm
from sklearn.discriminant_analysis import \
(LinearDiscriminantAnalysis as LDA,
QuadraticDiscriminantAnalysis as QDA)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from functools import partial
from sklearn.model_selection import \
(cross_validate, 
KFold,
ShuffleSplit)
import sklearn as skl
from sklearn.base import clone
from ISLP.models import sklearn_sm
from functools import partial
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from ISLP.models import \
     (Stepwise,
      sklearn_selected,
      sklearn_selection_path)
from sklearn.tree import (DecisionTreeClassifier as DTC,
                          DecisionTreeRegressor as DTR,
                          plot_tree,
                          export_text)
from sklearn.metrics import (accuracy_score,
                             log_loss)
from sklearn.ensemble import \
     (RandomForestRegressor as RF,
      GradientBoostingRegressor as GBR)
import statsmodels.formula.api as smf
import sys
from sklearn.preprocessing import PolynomialFeatures
import sklearn.model_selection as skm
from sklearn.discriminant_analysis import \
(LinearDiscriminantAnalysis as LDA,
QuadraticDiscriminantAnalysis as QDA)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from functools import partial
from sklearn.model_selection import \
(cross_validate, 
KFold,
ShuffleSplit)
from sklearn.base import clone
from ISLP.models import sklearn_sm
from functools import partial
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from ISLP.models import \
     (Stepwise,
      sklearn_selected,
      sklearn_selection_path)
import warnings
warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
from statsmodels.api import OLS
import sklearn.model_selection as skm
import sklearn.linear_model as skl
from sklearn.preprocessing import StandardScaler
from ISLP import load_data
from ISLP.models import ModelSpec as MS
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
import streamlit as stlt
from streamlit_option_menu import option_menu
import os
from datetime import date
from datetime import time
from datetime import timedelta
import mlflow
import databricks.sdk
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog
import io

#python -m streamlit run "c:/Users/katek/Desktop/NCF/Spring 2026/Dist Computing/Project 2/project2.py"

w = WorkspaceClient(
    host = "https://dbc-75cc67cd-21ce.cloud.databricks.com",
    token = ""
)

house_train = w.files.download("/Volumes/compfinal/default/compfinal/house_train.csv/part-00000-tid-6899596336054846361-567385bd-f712-4743-bc28-4416e52c99ca-263-1-c000.csv")
htrn = pd.read_csv(io.BytesIO(house_train.contents.read()))
house_train = pd.DataFrame(htrn)

house_test = w.files.download("/Volumes/compfinal/default/compfinal/house_test.csv/part-00000-tid-6986609311974168513-9cb6b7d2-a77b-42b8-bd28-1600738c16b2-265-1-c000.csv")
htst = pd.read_csv(io.BytesIO(house_test.contents.read()))
house_train = pd.DataFrame(htst)

predictthis = pd.DataFrame({'median_listing_price_mm':[0], 'median_listing_price_yy':[0], 'active_listing_count_mm':[0],
'active_listing_count_yy':[0], 'median_days_on_market_mm':[0], 'median_days_on_market_yy':[0], 'new_listing_count_mm':[0],
'new_listing_count_yy':[0], 'price_increased_count':[0], 'price_increased_count_mm':[0], 'price_increased_count_yy':[0],
'price_reduced_count':[0], 'price_reduced_count_mm':[0], 'price_reduced_count_yy':[0], 'pending_listing_count':[0],
'pending_listing_count_mm':[0], 'pending_listing_count_yy':[0], 'median_listing_price_per_square_foot':[0],
'median_listing_price_per_square_foot_mm':[0], 'median_listing_price_per_square_foot_yy':[0], 'median_square_feet':[0],
'median_square_feet_mm':[0], 'median_square_feet_yy':[0], 'average_listing_price_mm':[0], 'average_listing_price_yy':[0],
'total_listing_count_mm':[0], 'total_listing_count_yy':[0], 'New England':[0], 'Middle Atlantic':[0], 'East North Central':[0],
'West North Central':[0], 'South Atlantic':[0], 'East South Central':[0], 'West South Central':[0], 'Mountain':[0], 'Pacific':[0]})

def update_value():
    stlt.session_state.my_number = 100
if 'my_number' not in stlt.session_state:
    stlt.session_state.my_number = 100

stlt.header("How Long Will Your House Be On The Market?")
price = stlt.number_input("Type the price:", value = stlt.session_state.my_number, key=f'priceinput', on_change = update_value,)

sqft = stlt.number_input("Type the square footage:", value = stlt.session_state.my_number, key=f'sqftinput', on_change = update_value,)

state =  stlt.selectbox('What state is it in?:', ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut',
'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana',
'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada'
'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania',
'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia',
'Wisconsin', 'Wyoming'])
datee = stlt.slider("What month are you planning to list your house?", 1, 12, step = 1, key = f'dateselect')
pushdate = stlt.button('Predict', key=f'pushdate')
if pushdate == True:    
      stlt.subheader("Calculating...")           
      if state in ['Connecticut', 'Maine', 'Massachusetts', 'New Hampshire', 'Rhode Island', 'Vermont']:
            hmpred = house_test[(house_test["month_date_yyyymm"] == f'2025-{datee}-01') & (house_test["New England"] == 1)]
            predictthis.iloc[0:,27] == 1
      elif state in ['New Jersey', 'New York', 'Pennsylvania']:
            hmpred = house_test[(house_test["month_date_yyyymm"] == f'2025-{datee}-01') & (house_test["Middle Atlantic"] == 1)]
            predictthis.iloc[0:,28] == 1
      elif state in ['Illinois', 'Indiana', 'Michigan', 'Ohio', 'Wisconsin']:
            hmpred = house_test[(house_test["month_date_yyyymm"] == f'2025-{datee}-01') & (house_test["East North Central"] == 1)]
            predictthis.iloc[0:,29] == 1
      elif state in ['Iowa', 'Kansas', 'Minnesota', 'Missouri', 'Nebraska', 'North Dakota', 'South Dakota']:
            hmpred = house_test[(house_test["month_date_yyyymm"] == f'2025-{datee}-01') & (house_test["West North Central"] == 1)]
            predictthis.iloc[0:,30] == 1
      elif state in ['Delaware', 'Florida', 'Georgia', 'Maryland', 'North Carolina', 'South Carolina', 'Virginia', 'West Virginia']:
            hmpred = house_test[(house_test["month_date_yyyymm"] == f'2025-{datee}-01') & (house_test["South Atlantic"] == 1)]
            predictthis.iloc[0:,31] == 1
      elif state in ['Alabama', 'Kentucky', 'Mississippi', 'Tennessee']:
            hmpred = house_test[(house_test["month_date_yyyymm"] == f'2025-{datee}-01') & (house_test["East South Central"] == 1)]
            predictthis.iloc[0:,32] == 1
      elif state in ['Arkansas', 'Louisiana', 'Oklahoma', 'Texas']:
            hmpred = house_test[(house_test["month_date_yyyymm"] == f'2025-{datee}-01') & (house_test["West South Central"] == 1)]
            predictthis.iloc[0:,33] == 1
      elif state in ['Arizona', 'Colorado', 'Idaho', 'Montana', 'Nevada', 'New Mexico', 'Utah', 'Wyoming']:
            hmpred = house_test[(house_test["month_date_yyyymm"] == f'2025-{datee}-01') & (house_test["Mountain"] == 1)]
            predictthis.iloc[0:,34] == 1
      elif state in ['Alaska', 'California', 'Hawaii', 'Oregon', 'Washington']:
            hmpred = house_test[(house_test["month_date_yyyymm"] == f'2025-{datee}-01') & (house_test["Pacific"] == 1)]
            predictthis.iloc[0:,35] == 1
      hmpred = hmpred.drop(columns = ['month_date_yyyymm'])
      for i in range(0, 26):
            predictthis.iloc[0:, i] = np.mean(hmpred.iloc[0:, i])

      predictthis.iloc[0:, 17] = (price/sqft)
      predictthis.iloc[0:, 20] = sqft

      xtrain = house_train.drop(columns = ['median_days_on_market', "month_date_yyyymm"])
      ytrain = house_train['median_days_on_market']

      model = GBR(n_estimators = 20, learning_rate = 0.15, max_depth = 14, random_state = 100, max_features = 22)

      results = model.fit(xtrain, ytrain)
                              
      ypred = results.predict(predictthis)
      ypred = round(ypred[0])
      stlt.write(f"Your house will be on the market for approximately {ypred} days.")

#      mlflow.set_tracking_uri("databricks")
#      mlflow.set_registry_uri("databricks-uc")
#      mlflow.set_experiment("/Users/s.baker24@ncf.edu/ml_experiment")

#      with mlflow.start_run(run_name="Boost"):

#            xtrain = house_train.drop(columns=['median_days_on_market', "month_date_yyyymm"])
#            ytrain = house_train['median_days_on_market']

#            model = GBR(
#                  n_estimators=20,
#                  learning_rate=0.15,
#                  max_depth=14,
#                  random_state=100,
#                  max_features=22
#            )

#            model.fit(xtrain, ytrain)

#            ypred = model.predict(predictthis)

#            mlflow.log_param("max_features", model.max_features)
#            mlflow.log_param("n_estimators", model.n_estimators)
#            mlflow.log_param("learning_rate", model.learning_rate)
#            mlflow.log_param("max_depth", model.max_depth)

#            mlflow.log_metric("predicted_days", float(ypred[0]))

#            mlflow.sklearn.log_model(model, "model")
