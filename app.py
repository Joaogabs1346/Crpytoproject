#Base
import pandas as pd
import numpy as np

#graph
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,10)
import seaborn as sns

#TimeFunction
import time

#Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, render_template, request

#Warnings
import warnings
warnings.filterwarnings('ignore')



sc = StandardScaler()
# opening the DF
df  = pd.read_csv('data/crypto_markets.csv').drop(['market','close_ratio','spread','slug','symbol','ranknow','volume'],axis=1)


#-------------------------------------------------------------------------------------------
#Starting the flask application and choosing the templates folder
app = Flask(__name__, template_folder='template', static_folder='template/assets')
    
    
   
#INDEX/HOME PAGE
@app.route('/')
def index():
    return render_template('index.html')

#CLIENT FORM
@app.route('/dados_cliente')
def dados_cliente():
    return render_template("form.html")

#Capturing form data
def get_data():
    data = request.form.get('Cryptocurrency')    
    return data

#Result page to show the ML
@app.route('/send', methods=['POST'])
def show_data():
    #Accessing the data collected from the form 
    df_crpyto = get_data()
    df_crypto = str(df_crpyto)
    #Query using the chosen cryptocurrency
    Crypto_df = df.loc[df['name']== df_crpyto]
    Crypto_df = Crypto_df.drop('name',axis=1).set_index('date')
    #taking the average
    Crypto_df['ohlc_average'] = (Crypto_df['open'] + Crypto_df['high'] + Crypto_df['low'] + Crypto_df['close']) / 4
    Crypto_df['Price_After_Month']= Crypto_df['close'].shift(-30)   
    Crypto_df.dropna(inplace=True)
    #Separating data between training and testing
    X = Crypto_df.drop('Price_After_Month',axis=1)
    X = sc.fit_transform(X)#We need to scale our values to input them in our model
    y = Crypto_df['Price_After_Month']
    
    # Train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 42)

    # RandomFOrestRegressor(Ensemble)
    reg=RandomForestRegressor(bootstrap=True, max_depth=20, min_samples_leaf=2,min_samples_split= 5,
                          n_estimators=500,random_state=101)
    reg.fit(X_train,y_train)

    #Accuracy of the model
    accuracy=reg.score(X_test,y_test)
    accuracy=accuracy*100
    accuracy = float("{0:.2f}".format(accuracy))
    
    #Making the 30-day forecast
    X_30=X[-30:]
    forecast=reg.predict(X_30)    
    last_date=Crypto_df.iloc[-1].name
    modified_date = last_date 
    date=pd.date_range(modified_date,periods=30,freq='D')    
    df1=pd.DataFrame(forecast,columns=['Forecast'],index=date)    
    #Adding forecast column to original dataframe
    Crypto_df = Crypto_df.append(df1)
    #Plotting the graph
    Crypto_df['close'].plot(figsize=(12,6),label='Close')
    Crypto_df['Forecast'].plot(label='forecast')
    plt.legend()
    #Saving the plot with the CryptoName
    plt.savefig(f'template/assets/image/{df_crypto}.png')
    #Showing the results in the html page
    crypto = df_crypto
    outcome = accuracy       
    imagem = f'assets/image/{df_crypto}.png'    
    return render_template('result.html',result=outcome, imagem = imagem, crypto=crypto)

if __name__ == "__main__":
    app.run(debug=True)