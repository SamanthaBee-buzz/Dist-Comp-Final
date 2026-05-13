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
    host = "",
    token = ""
)
os.environ["DATABRICKS_HOST"] = ""
os.environ["DATABRICKS_TOKEN"] = ""

house_train = w.files.download("/Volumes/compfinal/default/compfinal/house_train.csv")
house_train = pd.read_csv(io.BytesIO(house_train.contents.read()))
house_train = pd.DataFrame(house_train)
house_train["month_date_yyyymm"] = pd.to_datetime(house_train["month_date_yyyymm"])
house_test = w.files.download("/Volumes/compfinal/default/compfinal/house_test.csv")
house_test = pd.read_csv(io.BytesIO(house_test.contents.read()))
house_test = pd.DataFrame(house_test)
house_test["month_date_yyyymm"] = pd.to_datetime(house_test["month_date_yyyymm"])

predictthis = pd.DataFrame({'median_listing_price_mm':[0], 'median_listing_price_yy':[0], 'active_listing_count_mm':[0],
'active_listing_count_yy':[0], 'median_days_on_market':[0], 'median_days_on_market_mm':[0], 'median_days_on_market_yy':[0], 'new_listing_count_mm':[0],
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
      hmpred = house_test
      stlt.subheader("Calculating...")           
      if state in ['Connecticut', 'Maine', 'Massachusetts', 'New Hampshire', 'Rhode Island', 'Vermont']:
            hmpred = house_test[(house_test["month_date_yyyymm"] == f'2025-{datee}-01') & (house_test["New England"] == 1)]
            predictthis["New England"] == 1
      elif state in ['New Jersey', 'New York', 'Pennsylvania']:
            hmpred = house_test[(house_test["month_date_yyyymm"] == f'2025-{datee}-01') & (house_test["Middle Atlantic"] == 1)]
            predictthis["Middle Atlantic"] == 1
      elif state in ['Illinois', 'Indiana', 'Michigan', 'Ohio', 'Wisconsin']:
            hmpred = house_test[(house_test["month_date_yyyymm"] == f'2025-{datee}-01') & (house_test["East North Central"] == 1)]
            predictthis["East North Central"] == 1
      elif state in ['Iowa', 'Kansas', 'Minnesota', 'Missouri', 'Nebraska', 'North Dakota', 'South Dakota']:
            hmpred = house_test[(house_test["month_date_yyyymm"] == f'2025-{datee}-01') & (house_test["West North Central"] == 1)]
            predictthis["West North Central"] == 1
      elif state in ['Delaware', 'Florida', 'Georgia', 'Maryland', 'North Carolina', 'South Carolina', 'Virginia', 'West Virginia']:
            hmpred = house_test[(house_test["month_date_yyyymm"] == f'2025-{datee}-01') & (house_test["South Atlantic"] == 1)]
            predictthis["South Atlantic"] == 1
      elif state in ['Alabama', 'Kentucky', 'Mississippi', 'Tennessee']:
            hmpred = house_test[(house_test["month_date_yyyymm"] == f'2025-{datee}-01') & (house_test["East South Central"] == 1)]
            predictthis["East South Central"] == 1
      elif state in ['Arkansas', 'Louisiana', 'Oklahoma', 'Texas']:
            hmpred = house_test[(house_test["month_date_yyyymm"] == f'2025-{datee}-01') & (house_test["West South Central"] == 1)]
            predictthis['West South Central'] == 1
      elif state in ['Arizona', 'Colorado', 'Idaho', 'Montana', 'Nevada', 'New Mexico', 'Utah', 'Wyoming']:
            hmpred = house_test[(house_test["month_date_yyyymm"] == f'2025-{datee}-01') & (house_test["Mountain"] == 1)]
            predictthis["Mountain"] == 1
      elif state in ['Alaska', 'California', 'Hawaii', 'Oregon', 'Washington']:
            hmpred = house_test[(house_test["month_date_yyyymm"] == f'2025-{datee}-01') & (house_test["Pacific"] == 1)]
            predictthis["Pacific"] == 1
      hmpred = hmpred.drop(columns = ['month_date_yyyymm'])
      for i in range(0, 27):
            predictthis.iloc[0, i] = np.mean(hmpred.iloc[0:, i])
      predictthis.iloc[0:, 18] = (price/sqft)
      predictthis.iloc[0:, 21] = sqft
      xtrain = house_train.drop(columns = ['median_days_on_market'])
      x = ['median_listing_price_mm', 'median_listing_price_yy', 'active_listing_count_mm', 'active_listing_count_yy', 'median_days_on_market_mm', 'median_days_on_market_yy',  'new_listing_count_mm','new_listing_count_yy', 'price_increased_count', 'price_increased_count_mm', 'price_increased_count_yy','price_reduced_count', 'price_reduced_count_mm', 'price_reduced_count_yy', 'pending_listing_count', 'pending_listing_count_mm', 'pending_listing_count_yy', 'median_listing_price_per_square_foot','median_listing_price_per_square_foot_mm', 'median_listing_price_per_square_foot_yy', 'median_square_feet','median_square_feet_mm', 'median_square_feet_yy', 'average_listing_price_mm','average_listing_price_yy','total_listing_count_mm', 'total_listing_count_yy', 'New England', 'Middle Atlantic', 'East North Central','West North Central', 'South Atlantic', 'East South Central', 'West South Central', 'Mountain', 'Pacific']
      design = MS(x)
      xtrain = design.fit_transform(xtrain)
      ytrain = house_train['median_days_on_market']
      ytrain = house_train['median_days_on_market']
      model = GBR(n_estimators = 20, learning_rate = 0.15, max_depth = 14, random_state = 100, max_features = 22)

      results = model.fit(xtrain, ytrain)

      xtest = predictthis.drop(columns = ['median_days_on_market'])
      xtest = design.fit_transform(xtest)
      ytest = predictthis['median_days_on_market']
      ypredtest = results.predict(xtest)
      ypred = round(ypredtest[0])
      stlt.write(f"Your house will be on the market for approximately {ypred} days.")

      mlflow.set_tracking_uri("databricks")
      mlflow.set_registry_uri("databricks-uc")
      mlflow.set_experiment("/Users/s.baker24@ncf.edu/marketpredictions")

      with mlflow.start_run(run_name = "Market predictor"):

            xtrain = house_train.drop(columns = ['median_days_on_market'])
            x = ['median_listing_price_mm', 'median_listing_price_yy', 'active_listing_count_mm', 'active_listing_count_yy', 'median_days_on_market_mm', 'median_days_on_market_yy',  'new_listing_count_mm','new_listing_count_yy', 'price_increased_count', 'price_increased_count_mm', 'price_increased_count_yy','price_reduced_count', 'price_reduced_count_mm', 'price_reduced_count_yy', 'pending_listing_count', 'pending_listing_count_mm', 'pending_listing_count_yy', 'median_listing_price_per_square_foot','median_listing_price_per_square_foot_mm', 'median_listing_price_per_square_foot_yy', 'median_square_feet','median_square_feet_mm', 'median_square_feet_yy', 'average_listing_price_mm','average_listing_price_yy','total_listing_count_mm', 'total_listing_count_yy', 'New England', 'Middle Atlantic', 'East North Central','West North Central', 'South Atlantic', 'East South Central', 'West South Central', 'Mountain', 'Pacific']
            design = MS(x)
            xtrain = design.fit_transform(xtrain)
            ytrain = house_train['median_days_on_market']
            ytrain = house_train['median_days_on_market']

            model = GBR(n_estimators = 20, learning_rate = 0.15, max_depth = 14, random_state = 100, max_features = 22)

            results = model.fit(xtrain, ytrain)
            
            ypredtrain = results.predict(xtrain)
            msetrain = np.mean((ytrain - ypredtrain)**2)
            mlflow.log_metric("Training MSE", msetrain)

            xtest = predictthis.drop(columns = ['median_days_on_market'])
            xtest = design.fit_transform(xtest)
            ytest = predictthis['median_days_on_market']
            ypredtest = results.predict(xtest)
            msetest = np.mean((ytest - ypredtest)**2)
            mlflow.log_metric("Test MSE", msetest)

            mlflow.log_metric('R2', r2_score(ytrain, ypredtrain))

            mlflow.log_metric('Max features', model.max_features)
            mlflow.log_metric('Estimators', model.n_estimators)
            mlflow.log_metric('Learning rate', model.learning_rate)
            mlflow.log_metric('Max depth', model.max_depth)

            mlflow.sklearn.log_model(model, "Market predictor")
