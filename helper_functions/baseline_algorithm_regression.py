# +
from sklearn.metrics import mean_squared_error

def baseline_algorithm_regression(df, target, feature, test):
    """baseline algorithm for regression"""

    output_values = df[target].values
    
    my_dict = df.groupby(feature)[target].mean().to_dict()
    
    df_modified = df.copy()
    for key in my_dict:
        df_modified.loc[df[feature] == key, target] = int(round(my_dict[key]))

    predicted = df_modified[target].values
    MSE = mean_squared_error(output_values,predicted)
    print('True Values={}\nPredicted Values={}\nMean Squared Error={}'.format(output_values, predicted, MSE))
