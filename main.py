from model import Prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split

def make_prediction(inputs, outputs, input_value, plot = False):
    if len(inputs) != len(outputs):
        raise Exception('Length of "inputs" and "outputs" must be same.')
    
    # Lets create dataframe for our data 
    df = pd.DataFrame({'inputs': inputs, 'outputs': outputs})
    
    # Reshape the data using numpy (X : Input, y : Output)
    X = np.array(df['inputs']).reshape(-1,1)
    y = np.array(df['outputs']).reshape(-1,1)
    
    # Spllit the data into training data to test our model
    train_X,test_X,train_y,test_y = train_test_split(X,y, random_state=0,test_size=.20)
    
    
    # Initialize the model and test it
    model = LinearRegression()
    model.fit(train_X, train_y)
    
    # Prediction
    y_prediction = model.predict([[input_value]])
    y_line = model.predict(X)
    
    # Testing for accuracy
    y_test_prediction = model.predict(test_X)
    
    # Plot
    if plot:
        display_plot(inputs=X, outputs=y, y_line=y_line)
    
    return Prediction(value=y_prediction[0][0],
                      r2_score = r2_score(test_y,y_test_prediction),
                      slope = model.coef_[0][0],
                      intercept=model.intercept_[0],
                      mean_absolute_error= mean_absolute_error(test_y,y_test_prediction))
    
    
def display_plot(inputs,outputs,y_line):
    plt.scatter(inputs,outputs,s=12)
    plt.xlabel('Inputs')
    plt.ylabel('Outputs')
    plt.plot(inputs, y_line, color = 'r')
    plt.show()
    
if __name__ == '__main__':
    years = [1,2,3,4,5,6,7,8,9,10]  
    earnings = [10000,12000,15000,20000,26000,30000,34000,40000,42000,45000]
    my_inputs = 20
    prediction = make_prediction(inputs=years, outputs=earnings,input_value=my_inputs, plot = False)
    print('Inputs:', my_inputs)
    print(prediction)
    
    print('Year 26:', prediction.slope*26)
    
    print(prediction.mean_absolute_error)