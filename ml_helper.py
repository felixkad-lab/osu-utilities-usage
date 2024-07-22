import sys
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay


# Do Linear regression using scipy
def do_scipy_linregress(x, y):
    result = stats.linregress(x, y)
    return result

# Perform linear regression for x and y
def do_linear_regression(x, y):
    # Do linear regression with scipy to get slope, intercept, 
    # and related errors
    scipy_linreg = do_scipy_linregress(x, y)
    
    # Reshape data for sklearn linear regression
    x = x.values.reshape(-1, 1)
    y = y.values.reshape(-1, 1)
    
    # Create model, fit data, and calcuate predicted values
    linreg = LinearRegression()
    linreg.fit(x, y)
    y_pred = linreg.predict(x)

    # Slope, intercept, rmse, and r^2 calculation
    m = linreg.coef_[0][0]
    b = linreg.intercept_[0]
    rmse = root_mean_squared_error(y, y_pred.flatten())
    r2 = r2_score(y, y_pred.flatten())    
    print("Slope: {:.2f}".format(m))
    print("Intercept: {:.2f}".format(b))
    print("RMSE: {:.2f}".format(rmse))
    print("R^2: {:.4f}\n".format(r2))

    return y_pred.reshape(-1), m, b, rmse, r2, scipy_linreg

# Do Linear Regression by parts                                       
def regression_by_parts(parts, xcol, ycol, part_col):
    # initiliaze lists
    len_parts = len(parts)
    
    x_list = [None] * len_parts
    y_list = [None] * len_parts
    y_pred = [None] * len_parts
    scipy_linreg = [None] * len_parts

    m_list = []
    b_list = []
    mse_list = []
    r2_list = []

    # Loop through different parts
    for i in range(len_parts):
        # Get variables
        x = parts[i][xcol]
        y = parts[i][ycol]
        
        # Sklearn linear regression
        print(f"Regression for Part {i+1}")
        pred_y, m, b, mse, r2, scipy_res = do_linear_regression(x, y)

        ## Add values to list
        x_list[i] = x
        y_list[i] = y
        y_pred[i] = pred_y
        scipy_linreg[i] = scipy_res

        m_list.append(m)
        b_list.append(b)
        mse_list.append(mse)
        r2_list.append(r2)

    return (
        x_list, y_list, y_pred, m_list, b_list, mse_list, r2_list, scipy_linreg
    )

# Calculate standard error of slope
def calc_slope_error(x, y, y_pred):
    # Calculate residuals
    residuals =  y - y_pred

    # Calculate the standard error of the slope
    mse = np.mean(residuals**2)
    variance_of_slope = mse /np.sum((x - np.mean(x))**2)
    std_error_of_slope = np.sqrt(variance_of_slope)

    return std_error_of_slope, residuals

# AC use classication prediction
def predict_ac_use(data=None, figname=None, outname=None):
    print("="*60)
    print("Doing some ML to predict AC use\n")
    df = data[['ghs_used', 'ac_use']].copy()    

    # Remove last row because no ghs_used
    df = df.iloc[:-1]
    print(f"df shape:{df.shape}\n")
    
    # Split features from target
    X = df.drop(['ac_use'], axis=1)
    y = df['ac_use']
    X_columns = X.columns.tolist()
    
    # Scale X values to be in the range of (0, 1)
    scaler = MinMaxScaler(feature_range = (0, 1))
    X = scaler.fit_transform(X)
    
    # Split data into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.1, random_state=42
    )

    y_test.name = 'y_test'

    print(f"Shapes:\nX_train = {X_train.shape}\ty_train = {y_train.shape}")
    print(f"X_test = {X_test.shape}\ty_test = {y_test.shape}\n")

    # Create Model and fit
    model = XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Convert X_test, y_pred and y_test to dataframes and merge them
    x_test_df = pd.DataFrame(
        data=X_test, columns=X_columns
    ).reset_index(drop=True)
    y_pred_df = pd.DataFrame(data=y_pred, columns=['y_pred'])
    y_test_df = y_test.copy()
    y_test_df = pd.DataFrame(
        y_test_df, columns=['y_test']
    ).reset_index(drop=True)
    out_df = x_test_df.join([y_test_df, y_pred_df])
    out_df.to_csv(outname, sep='\t', header=True, index=True)
    
    # Evaluate model
    error_rate = 1 - accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    print(f"Error rate of XGB classifier = {error_rate}\n")

    # Make graph of confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig = disp.plot()
    plt.savefig(figname, dpi=300, bbox_inches='tight')
    #plt.close(fig)

# Classification algorithms
def do_classification(model_choice=None, param_dict=None):
    if model_choice in ['XGBC', 'xgboost', 'XGBClassifier']:
        # xgboost
        model = XGBClassifier(**param_dict)
    elif model_choice in ['LGBMC', 'lightgbm', 'LGBMClassifier']:
        # LightGBM classifier        
        model = LGBMClassifier(**param_dict)        
    elif model_choice in ['Logis', 'LogisticRegression']:
        # Logistic Regression Classifier
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(**param_dict)
    elif model_choice in ['RandForC', 'RandomForest', 'randomforest', 'random_forest', 'RandomForestClassifier']:
        ## Random Forest classifier
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**param_dict) 
    elif model_choice in ['SVC', 'support_vector', 'support_vector_machine', 'svm', 'svc', 'SVM']:
        ## Support Vector Machine (SVM) 
        from sklearn.svm import SVC
        model = SVC(**param_dict) 
    elif model_choice in [
        'KNN', 'nearest_neighbors', 'KNeighborsClassifier', 'KNNC', 'KNNClassifier'
        ]:
        # K-Nearest Neighbor (KNN)
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(**param_dict)
    elif model_choice in ['Neural', 'neural_network', 'deep_learning', 'deepL']:
        # Neureal Network (Deep Learning)
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(**param_dict)
    elif model_choice in [
        'DeciTree', 'DecisionTreeClassifier', 'decision_tree', 'DecisionTree'
        ]:
        # Decision Tree Classifier
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(**param_dict)        
    else:
        print(f"I don't know {model_choice} model. Quitting.")
        sys.exit(1)
        
    return model
                      
# Classification algorithms
def tune_hyperparameters(
        model=None, X_train=None, y_train=None, algo_type=None, 
        param_grid=None):
    if algo_type in ['grid_search', 'GridSearch', 'GridSearchCV']:
        # GridSearchCV from sklearn
        from sklearn.model_selection import GridSearchCV

        #model = GridSearchCV(model, param_grid, n_jobs=-1, cv=5)
        clf = GridSearchCV(model, param_grid, n_jobs=-1, cv=5)
        search = clf.fit(X_train, y_train)
    elif algo_type in [
        'random_search', 'RandomSearch', 'RandomSearchCV', 'RandomizedSearchCV'
        ]:
        from sklearn.model_selection import RandomizedSearchCV   
    else:
        print(f"I don't know {algo_type} algorithm for tuning hyperparameters. Quitting.")
        sys.exit(1)

    return search.best_params_
