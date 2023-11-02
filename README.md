# House_Price_Prediction
House Price Prediction based on various factor.

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    data1 = pd.read_csv('House Price India.csv')
    data=data1.copy()
    data.isnull().sum()
    data=pd.DataFrame(data)
    data=data.drop(['id','Date','waterfront present','number of views','condition of the house','grade of the house','Renovation Year', 'Postal Code', 'Lattitude','Longitude','living_area_renov', 'lot_area_renov',
    'Number of schools nearby', 'Distance from the airport'],axis=1)
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    from sklearn.linear_model import LinearRegression
    multi_regressor = LinearRegression()
    multi_regressor.fit(x_train, y_train)
    y_pred1=multi_regressor.predict(x_test)
    from sklearn.metrics import r2_score
    print(r2_score(y_test, y_pred1))
    new_data=pd.DataFrame({
        'number of bedrooms':3,
        'number of bathrooms':1,
        'living area':4000,
        'lot area':9080,
        'number of floors':1,
        'Area of the house(excluding basement)':1556,
        'Area of the basement':0,
        'Built Year':2000,
    },index=[0])
    multi_regressor = LinearRegression()
    multi_regressor.fit(x, y)
    predicted_price = multi_regressor.predict(new_data)
    print(predicted_price)
    
    
    import matplotlib.pyplot as plt
    x = data['number of bathrooms']
    y = data['Price']
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.5)
    plt.title('Number of Bedrooms vs. Price')
    plt.xlabel('Number of Bedrooms')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    sns.countplot(x='number of bedrooms',hue='Price',data=data1)
    plt.subplot(1,2,2)
    sns.countplot(x='number of bathrooms',hue='Price',data=data1)
