import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker


if __name__ == '__main__':
    # read history data
    # extract date and USD/NTD
    # save data as pandas dataframe
    data = pd.read_csv('history.csv', encoding= 'unicode_escape', header=[0])
    data = data[['date','USD/NTD']]

    # get train data to train ai model
    # use last 30 days data to predict current data
    # train data range from 2010/04/01 to 2018/12/28
    # use spread rather than price to improve accuracy
    x_train = []
    y_train = []
    for i in range(0, 2136):
        temp = []
        for j in range(0, 31):
            if j == 30:
                y_train.append(data['USD/NTD'][i+j] - data['USD/NTD'][i])
            else:
                temp.append(data['USD/NTD'][i+j] - data['USD/NTD'][i])
        x_train.append(temp)

    # get test data to predict price
    # test data range from 2019/01/02 to 2022/05/26
    x_test = []
    y_test = []
    x_base = []
    for i in range(2136, 2960):
        temp = []
        x_base.append(data['USD/NTD'][i])
        for j in range(0, 31):
            if j == 30:
                y_test.append(data['USD/NTD'][i+j] - data['USD/NTD'][i])
            else:
                temp.append(data['USD/NTD'][i+j] - data['USD/NTD'][i])
        x_test.append(temp)
    
    # use RandomForest as model to train ai
    # use ai model to predict 
    # x_train is last 30 days spread and y_train is now spread
    model = RandomForestRegressor(n_estimators=200).fit(x_train, y_train)        
    y_predict = model.predict(x_test)

    # add spread and base price to get predict price
    for i in range(len(x_base)):
        y_predict[i] = y_predict[i] + x_base[i]

    # make plot
    date_plot = data['date'][2166:2990]
    data_plot = data['USD/NTD'][2166:2990]
    date_plot.reset_index(inplace = True, drop = True)
    data_plot.reset_index(inplace = True, drop = True)
    plt.plot(date_plot, y_predict)
    plt.plot(date_plot, data_plot)
    plt.legend(['Predict','Origin'])
    ax = plt.subplot(111)
    plt.title("RandomForest Prediction")    
    ax.set_xlabel('date')
    ax.set_ylabel('USD/NTD', rotation = 0)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(51))
    plt.show()
    
    # count deviation
    # add all abs(true price - predict price)
    # the goal is to reduce this value
    error = 0
    for i in range(len(data_plot)):
        error += abs(data_plot[i] - y_predict[i])
    print(error)

    # (optional) output ai model as pickle file
    with open('save/Random_ForestPrediction.pickle', 'wb') as f:
        pickle.dump(model, f)