# Importing the Libraries
#-------------------------------------------------------------------
import os
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.patches as mpatches
import numpy as np
import tensorflow as tf
import keras.backend as K
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import base64
from flask import Flask, render_template, request
from flask import redirect
import pandas as pd
import os
import seaborn as sns
import matplotlib.image as mpimg
from werkzeug.utils import secure_filename
import keras

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
image_path="/Output.png"
full_filename=""
app = Flask(__name__)
output=""

IMG_PLOT_DIR='D:/CanDataAnalytics/CanDataAnalytics/static/img/plot.png'
app.config['UPLOAD_DIRECTORY'] = 'D:/CanDataAnalytics/CanDataAnalytics/static/'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1024MB
ALLOWED_EXTENSIONS = {'csv'}
filepath = "D:/CanDataAnalytics/CanDataAnalytics/static/CAN.csv"
FILE_UPLOAD_FLAG=False
FILE_UPLOAD_FLAG_1=False
file_path=""
TIME_STEPS =20

app.secret_key = 'This is your secret key to utilize session in Flask'
#----------------------------------------------------------------------------

model = keras.models.load_model('D:/CanDataAnalytics/CanDataAnalytics/Epchos_28_4_22_mem_20__30_Epochs_Last')

#----------mainpage----------
@app.route('/', methods = ['GET','POST'])

def index():
    if request.method == "GET":
        global FILE_UPLOAD_FLAG_1
         #-----------------------------------------------------------
        if FILE_UPLOAD_FLAG_1:
            TIME_COL=pd.read_csv("D:/CanDataAnalytics/CanDataAnalytics/static/CAN.csv",usecols=["Time[s]"],nrow=5000)
            anomalies = pd.read_csv("D:/CanDataAnalytics/CanDataAnalytics/static/anomalies.csv")
            TP = 0
            FN = 0
            for i in range(1720, 1721):
                temp = TIME_COL.iloc[i, 0]
                if temp in anomalies['Time[s]'].values:
                    TP = TP + 1
                else:
                    FN = FN + 1
            for i in range(2720, 2721):
                temp = TIME_COL.iloc[i, 0]
                if temp in anomalies['Time[s]'].values:
                    TP = TP + 1
                else:
                    FN = FN + 1
            print("True Positives  :", TP)
            print("False Nagative  :", FN)
            FP = len(anomalies) - (TP + FN)
            print("False Positives :", FP)
            TN = len(TIME_COL) - len(anomalies)
            print("True Nagative   :", TN)
            precision = (TP) / (TP + FP)
            recall = (TP) / (TP + FN)
            f1_score = 2 * ((precision * recall) / (precision + recall))
            accuracy = (TP + TN) / (TP + FN + FP + TN)
            #----------------------------------------------------------------------------------------
            Normal_anomalies_mean = pd.read_csv("D:/CanDataAnalytics/CanDataAnalytics/static/Normal_anomalies_mean.csv")
            Anomalies_df_mean = pd.read_csv("D:/CanDataAnalytics/CanDataAnalytics/static/Anomalies_df_mean.csv")

            print("Anomaly signals are")
            print("--------------------")
            Anomaly_Signals=[]
            for i in range(len(Normal_anomalies_mean)):
                normal_row = Normal_anomalies_mean.iat[i, 0]  # [row,col]
                anomaly_row = Anomalies_df_mean.iat[i, 0]

                if normal_row > anomaly_row:
                    diff = normal_row - anomaly_row
                else:
                    diff = anomaly_row - normal_row
                # print(diff)

                if diff > 0.0001:
                    temp = Anomalies_df_mean.iloc[[i], [0]]
                    print(temp.index[0])
                    Anomaly_Signals.append(temp.index[0])
            FILE_UPLOAD_FLAG_1=False
        else:
            anomalies=pd.DataFrame()
            Normal_anomalies_mean=anomalies.copy()
            TP=0
            FP=0
            Anomaly_Signals=NA
            
            
        return render_template('index.html',value=len(Normal_anomalies_mean), value1=len(anomalies),value2=TP,value3=FP,value4=Anomaly_Signals, tables=[anomalies.to_html(classes='data')], titles=anomalies.columns.values)
    
#--------input_page--------    
@app.route('/form',methods = ['GET'])
def Test_Report():
        
        return render_template('form.html')
    
    
#---------Analysis---------
@app.route('/table', methods=['GET'])
def Tables():
    global file_path
    global FILE_UPLOAD_FLAG
    if FILE_UPLOAD_FLAG:
# Reading the uoloaded file any removing empty columns
        uploaded_df = pd.read_csv(file_path,encoding= 'unicode_escape',nrows=300)
        uploaded_df = uploaded_df.dropna(axis=1, how='all', inplace=False)
        uploaded_df = uploaded_df.set_index('Time[s]')
        # Creating empty dataframe to store frequency of each signal
        count_df = pd.DataFrame(columns=['count'])
        # Extracting column names
        col_names = uploaded_df.columns.values
        col_names_df = pd.DataFrame(col_names)
        # Finding the frequency of all signals
        for k in range(len(uploaded_df.columns)):
            count = 0

            for i in range(len(uploaded_df) - 1):
                num1 = uploaded_df.iat[i, k]  # i rows ,k col
                num2 = uploaded_df.iat[i + 1, k]
                if (num1 > num2) or (num1 < num2):
                    count = count + 1

            count_df.loc[k] = count
        # concatenating the signals names with their frequency
        freq_count = pd.concat([col_names_df, count_df], join='outer', axis=1)
        freq_count.columns.values[0] = "signals"
        # Finding High frequency Signals
        high_freq_signals = pd.DataFrame()  # creating df with column name signals
        for i in range(len(uploaded_df.columns)):
            temp = freq_count.iat[i, 1]  # i row 1 st col
            if temp >= (len(uploaded_df) * 0.4):
                data = uploaded_df[freq_count.iat[i, 0]]
                high_freq_signals = pd.concat([high_freq_signals, data], join='outer', axis=1)
        high_freq_signals.iloc[:, :5].plot(subplots=True, figsize=(15, 25))
        plt.xlabel('Time[s]', fontsize=15)
        plt.ylabel('Signal_Value', fontsize=15)
        plt.savefig('D:/CanDataAnalytics/CanDataAnalytics/static/img/High_freq_plot.png')

        # Finding Mid frequency Signals
        mid_freq_signals = pd.DataFrame()  # creating df with column name signals
        for i in range(len(uploaded_df.columns)):
            temp = freq_count.iat[i, 1]  # i row 1 st col
            if temp < (len(uploaded_df) * 0.4) and temp >= (len(uploaded_df) * 0.1) :
                data = uploaded_df[freq_count.iat[i, 0]]
                mid_freq_signals = pd.concat([mid_freq_signals, data], join='outer', axis=1)
        mid_freq_signals.iloc[:, :5].plot(subplots=True, figsize=(15, 25))
        plt.xlabel('Time[s]', fontsize=15)
        plt.ylabel('Signal_Value', fontsize=15)
        plt.savefig('D:/CanDataAnalytics/CanDataAnalytics/static/img/Mid_freq_plot.png')

        # Finding Low frequency Signals
        low_freq_signals = pd.DataFrame()  # creating df with column name signals
        for i in range(len(uploaded_df.columns)):
            temp = freq_count.iat[i, 1]  # i row 1 st col
            if temp < (len(uploaded_df) * 0.1):
                data = uploaded_df[freq_count.iat[i, 0]]
                low_freq_signals = pd.concat([low_freq_signals, data], join='outer', axis=1)
        low_freq_signals.iloc[:, :5].plot(subplots=True, figsize=(15, 25))
        plt.xlabel('Time[s]', fontsize=15)
        plt.ylabel('Signal_Value', fontsize=15)
        plt.savefig('D:/CanDataAnalytics/CanDataAnalytics/static/img/Low_freq_plot.png')
        
        FILE_UPLOAD_FLAG=False

    return render_template('table.html')


#-----output curve-----
@app.route('/chart', methods=['GET'])
def Charts():

    output_image = os.path.join('D:/CanDataAnalytics/CanDataAnalytics/static/img', 'output' + '.png')
    return render_template('chart.html',user_image = output_image)



#-----upload----------
@app.route('/upload',methods=['POST'])
def upload():
    global FILE_UPLOAD_FLAG
    global FILE_UPLOAD_FLAG_1
    global file_path
    file = request.files['file']

    if file:
        file_path=os.path.join(app.config['UPLOAD_DIRECTORY'],file.filename)
        file.save(file_path)
        FILE_UPLOAD_FLAG = True
        FILE_UPLOAD_FLAG_1 = True
        normal_mean(file_path)
        anomaly_mean(file_path)
                   
                
    return redirect('/table')

#-----High_Mid_low_freq_page------
@app.route('/highfreq', methods=['GET'])
def high():

    # input_image = os.path.join('static/img', 'high_freq' + '.png')
    return render_template('High_freq.html')

@app.route('/midfreq', methods=['GET'])
def mid():

    # input_image = os.path.join('static/img', 'high_freq' + '.png')
    return render_template('Mid_freq.html')

@app.route('/lowfreq', methods=['GET'])
def low():

    # input_image = os.path.join('static/img', 'high_freq' + '.png')
    return render_template('Low_freq.html')



def sequence(x,y,seq_size=TIME_STEPS):
    x_values=[]
    y_values=[]
        
    for i in range (len(x)-seq_size):
        x_values.append(x.iloc[i:(i+seq_size)].values)
        y_values.append(y.iloc[i+seq_size])
        
    return np.array(x_values), np.array(y_values)

#--------normmal mean calcuation------
def normal_mean(filepath):
    global model
    data = pd.read_csv(filepath, encoding='unicode_escape')

    # Taking the Time Column
    TIME_COL = data.iloc[:, [0]]
    DataFrame = data.copy()
    No_of_Signal = len(DataFrame.columns)

    # Taking the 10 signals
    DataFrame = DataFrame[['UB_BrakePressure_HS', 'ABSChecksum_HS[Counter]',
                                   'BrakePressure_HS[Bar]', 'UB_VehRefSpeed_HS',
                                   'HillDescentMode_HS', 'RDiffTorqLimit_HS[Nm]', 'BrakePressureQF_HS',
                                   'UB_WheelSpeedFrL_HS',
                                   'UB_TractionControlFault_HS', 'UB_BrakePressureQF_HS']]
    # Taking only 5000 rows
    DataFrame = DataFrame.iloc[1:4990]
    PredictDataFrame = DataFrame.copy()

    # Standard Scaling the data
    scaler = StandardScaler()
    DataFrame = scaler.fit_transform(DataFrame)
    DataFrame = pd.DataFrame(DataFrame, columns=PredictDataFrame.columns)
    Normal_DataFrame = DataFrame.copy()

    X_train_noramal, Y_train_normal = sequence(Normal_DataFrame, Normal_DataFrame)

    X_predict = model.predict(X_train_noramal)
    X_predict_normal = pd.DataFrame(X_predict, columns=Normal_DataFrame.columns)

    Loss_normal = np.mean(np.abs(X_predict_normal - Normal_DataFrame), axis=1)
    Loss_normal_df = pd.DataFrame(Loss_normal, columns=['Normal_Loss'])
    Loss_normal_df['Time[s]'] = TIME_COL['Time[s]']
    Loss_normal_df = Loss_normal_df.set_index('Time[s]')
    Loss_normal_df = Loss_normal_df.dropna(axis=0)


    Loss_normal_df_th = Loss_normal_df.copy()
    Loss_normal_df_th = Loss_normal_df_th.reset_index(drop=True)
    Loss_normal_df_th['Time[s]'] = TIME_COL['Time[s]']
    Loss_normal_df_th['anomaly'] = Loss_normal_df_th['Normal_Loss'] > 0.077

    Loss_normal_df_th = pd.concat([Loss_normal_df_th, Normal_DataFrame],
                                  join='outer', axis=1)
    Normal_anomalies = Loss_normal_df_th.loc[Loss_normal_df_th['anomaly'] == True]

    Normal_anomalies = Normal_anomalies.set_index('Time[s]')
    Normal_anomalies = Normal_anomalies.drop(['Normal_Loss', 'anomaly'], axis=1)

    Normal_anomalies_mean = Normal_anomalies.mean()
    Normal_anomalies_mean = pd.DataFrame(Normal_anomalies_mean)
    Normal_anomalies_mean.to_csv('D:/CanDataAnalytics/CanDataAnalytics/static/Normal_anomalies_mean.csv', index=False)
    return 0

    
#--------Finding Anomaly signal ------------
def anomaly_mean(filepath):
    global model
    data = pd.read_csv(filepath,encoding= 'unicode_escape')

    # Taking the Time Column
    TIME_COL = data.iloc[:, [0]]
    DataFrame= data.copy()
    No_of_Signal=len(DataFrame.columns)

    # Taking the 10 signals
    DataFrame=DataFrame[['UB_BrakePressure_HS','ABSChecksum_HS[Counter]',
                                        'BrakePressure_HS[Bar]','UB_VehRefSpeed_HS',
                                         'HillDescentMode_HS','RDiffTorqLimit_HS[Nm]','BrakePressureQF_HS','UB_WheelSpeedFrL_HS',
                                         'UB_TractionControlFault_HS','UB_BrakePressureQF_HS']]
    # Taking only 5000 rows
    DataFrame=DataFrame.iloc[1:4990]
    PredictDataFrame=DataFrame.copy()
    
    #Introducing Anomaly

    for i in range(1720, 1721):
        DataFrame.iat[i, 1] = (DataFrame.iat[i, 1] + 50)
    for i in range(2720, 2721):
        DataFrame.iat[i, 2] = (DataFrame.iat[i, 2] - 10)

    # Standard Scaling the data
    scaler = StandardScaler()
    DataFrame=scaler.fit_transform(DataFrame)
    DataFrame=pd.DataFrame(DataFrame,columns=PredictDataFrame.columns)

    # Generating the sequence
    X_train, Y_train= sequence(DataFrame,DataFrame)

    # Loading trained model

    X_predict = model.predict(X_train)
    X_predict_DataFrame = pd.DataFrame(X_predict, columns=DataFrame.columns)
    # Finding Loss using Threshold
    Loss = np.mean(np.abs(X_predict_DataFrame - DataFrame), axis=1)
    Loss_df = pd.DataFrame(Loss)
    Loss_df['Time[s]'] = TIME_COL['Time[s]']
    pred_error_threshold = 0.077
    Loss_df.columns.values[0] = 'Loss'
    Loss_df['anomaly'] = Loss_df['Loss'] > pred_error_threshold
    Loss_df.drop([0], axis=0, inplace=True)
    Loss_df['Threshold'] = pred_error_threshold
    # Extracting only anomaly rows
    anomalies = Loss_df.loc[Loss_df['anomaly'] == True]
    # saving anomalies to csv
    anomalies.to_csv('D:/CanDataAnalytics/CanDataAnalytics/static/anomalies.csv',
                                 header =True, index = False)
    
     # Plotting loss v/s Time with Anomaly points
    sns.set(rc={'figure.figsize': (15, 5)})
    sns.lineplot(x=Loss_df['Time[s]'], y=Loss_df['Loss'], label='Loss')
    sns.lineplot(x=Loss_df['Time[s]'], y=Loss_df['Threshold'], label='Threshold', color='r')
    # Plot anomalies
    plot = sns.scatterplot(x=anomalies['Time[s]'], y=anomalies['Loss'], color='r', label='Anomalies')
    fig = plot.get_figure()
    # Saving the figure
    fig.savefig("D:/CanDataAnalytics/CanDataAnalytics/static/img/loss_plot_1.png")

    #---------------------------------------------------------------------------------------
    
    # finding mean of anomaly signals
    Loss_anomaly_df_th = Loss_df.copy()
    Loss_anomaly_df_th = pd.concat([Loss_anomaly_df_th, DataFrame],
                                   join='outer', axis=1)
    Anomalies_df = Loss_anomaly_df_th.loc[Loss_anomaly_df_th['anomaly'] == True]

    Anomalies_df = Anomalies_df.set_index('Time[s]')
    Anomalies_df = Anomalies_df.drop(['Loss', 'anomaly', 'Threshold'], axis=1)
    #print(Anomalies_df)
    Anomalies_df_mean = Anomalies_df.mean()
    Anomalies_df_mean = pd.DataFrame(Anomalies_df_mean)
    Anomalies_df_mean.to_csv('D:/CanDataAnalytics/CanDataAnalytics/static/Anomalies_df_mean.csv',
                             header =True, index = False)
    return 0
    
        
       

if __name__ == '__main__':
    app.run(debug=True,)