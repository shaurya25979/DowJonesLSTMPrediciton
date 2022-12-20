# Building training set, test set and LSTM model

def new_dataset(dataset, step_size):
	data_X, data_Y = [], []
	for i in range(len(dataset)-step_size-1):
		a = dataset[i:(i+step_size), 0]
		data_X.append(a)
		data_Y.append(dataset[i + step_size, 0])
	return np.array(data_X), np.array(data_Y)
 
BullStock = [] # For stocks showing upward trend
BearStock = [] # Fow stocks showing downward trend

for Eachstock,ticker in zip(StockData, tickers):
# Selecting the necessary attributes for prediction 
    OHLC_avg = Eachstock[['Open','High', 'Low', 'Close']].mean(axis = 1)
    HLC_avg = Eachstock[['High', 'Low', 'Close']].mean(axis = 1)
    close_val= Eachstock[['Close']]
    
# Converting to time series    
    OHLC_avg = np.reshape(OHLC_avg.values, (len(OHLC_avg),1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    OHLC_avg = scaler.fit_transform(OHLC_avg)
    
# Splitting
    train_OHLC = int(len(OHLC_avg) * 0.75)
    test_OHLC = len(OHLC_avg) - train_OHLC
    train_OHLC, test_OHLC = OHLC_avg[0:train_OHLC,:], OHLC_avg[train_OHLC:len(OHLC_avg),:]

# TIME-SERIES DATASET FOR TIME T, VALUES FOR TIME T+1
    trainX, trainY = new_dataset(train_OHLC, 1)
    testX, testY = new_dataset(test_OHLC, 1)

# Reshaping
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    step_size = 1

# LSTM MODEL
    model = Sequential()
    model.add(LSTM(32, input_shape=(1, step_size), return_sequences = True))
    model.add(LSTM(16))
    model.add(Dense(1))
    model.add(Activation('linear'))
    print(ticker.capitalize())
# Training
    model.compile(loss='mean_squared_error', optimizer='adagrad') 
    model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)

# Predicting 
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    print('\n\n\n')
    
    
# Output, Error and Plotting 
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])


# Trainng RMSE
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train RMSE of '+ticker+' %.2f' % (trainScore))

# Test RMSE
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test RMSE of '+ticker+' %.2f' % (testScore))

# Plotting
    trainPredictPlot = np.empty_like(OHLC_avg)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[step_size:len(trainPredict)+step_size, :] = trainPredict

    testPredictPlot = np.empty_like(OHLC_avg)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(step_size*2)+1:len(OHLC_avg)-1, :] = testPredict

    OHLC_avg = scaler.inverse_transform(OHLC_avg)


    plt.figure(figsize=(15,5))
    subplot(2,1,1)
    plt.plot(OHLC_avg, 'g', label = 'original dataset')
    subplot(2,1,2)
    plt.title(number)
    plt.plot(trainPredictPlot, 'r', label = 'training set')
    plt.plot(testPredictPlot, 'b', label = 'predicted stock price/test set')
    plt.legend(loc = 'upper right')
    plt.xlabel('Number of days')
    plt.ylabel('OHLC Value')
    plt.show()
    # PREDICT FUTURE VALUES
    last_val = testPredict[-1]
    last_val_scaled = last_val/last_val
    next_val = model.predict(np.reshape(last_val_scaled, (1,1,1)))
    next_val = scaler.inverse_transform(next_val)
    
    #print(f"Raw value of next val is {next_val}")
    #print(f"Scaled value of next val is {}")
    
    last_val = np.asscalar(last_val)
    #next_val = np.asscalar(last_val*next_val)
    print ("Last Day Value of "+ticker, last_val)
    print ("Next Day Value of"+ticker, next_val)
    print('\n')
    if last_val > next_val:
        print('Negative cluster')
        BearStock.append(ticker)
    else:
        print('Positive cluster')
        BullStock.append(ticker)
    print('\n\n\n')

