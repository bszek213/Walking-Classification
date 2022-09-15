#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classification of walking and non-walking data via machine learning 
classifiers. 
@author: brianszekely
"""
import pupil_recording_interface as pri
import os
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from pandas import Series
from datetime import datetime
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from scipy import stats
from sklearn.preprocessing import StandardScaler
"""
TODO: Create method that can append multiple takes together into one massive 
timeseries.
"""
class WalkPredictor():
    def __init__(self):
        print('initialize WalkPredictor')
    def readin(self,folder='/media/brianszekely/TOSHIBA EXT/UNR/2021_07_13_15_27_51/'):
        #TODO: readin and concatenate multiple recordings
        isdir = os.path.isdir(folder) 
        if isdir == False:
            print('Folder does not exist. Check the path')
        elif isdir == True:
            self.folder = folder
            self.odometry = pri.load_dataset(folder,odometry='recording',cache=False)
            self.accel = pri.load_dataset(folder,accel='recording',cache=False)
    def encode_binary(self):
        #create an array of zeros for the label data 
        zero_arr = np.zeros(len(self.odometry.orientation.values),dtype = int)
        self.odometry['labels'] = (['time'],  zero_arr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.odometry.time.values,
            y=self.odometry.angular_velocity[:, 0],
            name="pitch"       # this sets its legend entry
            ))
        fig.update_layout(
            title="Angular velocity odometry",
            xaxis_title="Time stamps (datetime)",
            yaxis_title="Angular Velocity (radians/second)",
            )
        fig.show()
        done = False
        times = []
        while not done:
            pitch_start = input('Pitch Timestamp Start (HH:mm:ss format): ')
            pitch_end = input('Pitch Timestamp End (HH:mm:ss format): ')
            df_time = Series(self.odometry.time[0].values)
            pitch_start = datetime.combine(df_time.dt.date.values[0],
                                            datetime.strptime(pitch_start, '%H:%M:%S').time())
            pitch_end = datetime.combine(df_time.dt.date.values[0],
                                            datetime.strptime(pitch_end, '%H:%M:%S').time())
            tmp = {'walking':
                {'start': pitch_start,
                    'end': pitch_end,
                    }}
            times.append(tmp)
            next_calibration = input('Continue for more labeling? (y/n) ')
            done = next_calibration != 'y'
        for num in times:
            temp_array = self.odometry.labels.sel(time=slice(num['walking']['start'],num['walking']['end']))
            temp_arr_start = self.odometry.labels.where(self.odometry.time == temp_array.time[0])
            temp_arr_end = self.odometry.labels.where(self.odometry.time == temp_array.time[-1])
            start_loc = np.where(temp_arr_start == 0)[0]
            end_loc = np.where(temp_arr_end == 0)[0]
            self.odometry.labels[start_loc[0]:end_loc[0]] = 1
        #Check where the ones and zeros are in the array
        # fig = go.Figure()
        # fig.add_trace(go.Scatter(
        #     x=self.odometry.time.values,
        #     y=self.odometry.labels.values,
        #     name="1s"       # this sets its legend entry
        #     ))
        # fig.show()
    def split(self):
        #Drop unnecessary features
        self.odometry = self.odometry.drop_vars('confidence')
        variables = ['position','linear_velocity','angular_velocity',
                     'linear_acceleration','angular_acceleration','orientation']
        self.y = self.odometry.labels
        self.x = self.odometry[variables]
        self.x = DataFrame(list(zip(self.x.position[:,0].values,self.x.position[:,1].values,self.x.position[:,2].values,
                                self.x.linear_velocity[:,0].values,self.x.linear_velocity[:,1].values,self.x.linear_velocity[:,2].values,
                                self.x.angular_velocity[:,0].values,self.x.angular_velocity[:,1].values,self.x.angular_velocity[:,2].values,
                                self.x.linear_acceleration[:,0].values,self.x.linear_acceleration[:,1].values,self.x.linear_acceleration[:,2].values,
                                self.x.angular_acceleration[:,0].values,self.x.angular_acceleration[:,1].values,self.x.angular_acceleration[:,2].values,
                                self.x.orientation[:,0].values,self.x.orientation[:,1].values,self.x.orientation[:,2].values,self.x.orientation[:,3].values
                                )),
                    columns =['position_x','position_y','position_z',
                              'linear_velocity_x','linear_velocity_y','linear_velocity_z',
                              'angular_velocity_x','angular_velocity_y','angular_velocity_z',
                              'linear_acceleration_x','linear_acceleration_y','linear_acceleration_z',
                              'angular_acceleration_x','angular_acceleration_y','angular_acceleration_z',
                              'orientation_w','orientation_x','orientation_y', 'orientation_z'
                              ])
        self.y = DataFrame(self.odometry.labels.values, columns=['labels'])
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x,self.y, train_size=0.8)
    def pre_process(self):
        # #Scale data 
        scaler = StandardScaler()
        scaled_data = scaler.fit(self.x_train).transform(self.x_train)
        self.x_train = DataFrame(scaled_data, columns = self.x_train.columns)
        scaled_data_test = scaler.fit(self.x_test).transform(self.x_test)
        self.x_test = DataFrame(scaled_data_test, columns = self.x_test.columns)
        #Log Transform - Should I transform or scale first?
        # for col_name in self.x_train.columns:
        #     self.x_train[col_name] = np.log10(self.x_train[col_name])
        #     self.x_test[col_name] = np.log10(self.x_test[col_name])
        # Find features with correlation greater than 0.90
        corr_matrix = np.abs(self.x_train.astype(float).corr())
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] >= 0.75)]
        self.x_train = self.x_train.drop(columns=to_drop)
        self.x_test = self.x_test.drop(columns=to_drop)
        self.drop_cols = to_drop
        print(f'features to drop (coef > 0.75): {to_drop}')
        #Probability plots - normal distribution
        for col_name in self.x_train.columns:
            self.prob_plots(col_name)
        #plot heat map
        top_corr_features = corr_matrix.index
        plt.figure(figsize=(20,20))
        plt.title('Correlation matrix of all t265 data')
        g=sns.heatmap(corr_matrix[top_corr_features],annot=True,cmap="RdYlGn")
        plt.savefig('correlations.png')
        plt.close()

    def prob_plots(self,col_name):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        stats.probplot(self.x_train[col_name], dist=stats.norm, plot=ax1)
        title = f'probPlot of training data against normal distribution, feature: {col_name}'  
        ax1.set_title(title,fontsize=10)
        save_name = 'probplot_' + col_name + '.png'
        plt.tight_layout()
        plt.savefig(os.path.join(os.getcwd(), 'prob_plots',save_name), dpi=200)

    def learn(self):
        """
        check the base case classifiers and then tune the one with the highest accuracy
        TODO: tune the model with the highest accuracy
        """
        #check the base case
        print('training base casee for all models')
        kneighbor = KNeighborsClassifier().fit(self.x_train,self.y_train)
        log_regress = LogisticRegression().fit(self.x_train,self.y_train)
        grad_class = GradientBoostingClassifier().fit(self.x_train,self.y_train)
        mlpClass = MLPClassifier().fit(self.x_train,self.y_train)
        randFor = RandomForestClassifier().fit(self.x_train,self.y_train)
        print('kneighbor acc: ',accuracy_score(self.y_test, kneighbor.predict(self.x_test)))
        print('logisticRegression acc: ',accuracy_score(self.y_test, log_regress.predict(self.x_test)))
        print('gradientBoostingClassifier acc: ',accuracy_score(self.y_test, grad_class.predict(self.x_test)))
        print('mlpClassifier acc: ',accuracy_score(self.y_test, mlpClass.predict(self.x_test)))
        print('randomForestClassifier acc: ',accuracy_score(self.y_test, randFor.predict(self.x_test)))
        dict_models = {'kneighbor': accuracy_score(self.y_test, kneighbor.predict(self.x_test)),
                       'logisticRegression': accuracy_score(self.y_test, log_regress.predict(self.x_test)),
                       'gradientBoostingClassifier': accuracy_score(self.y_test, grad_class.predict(self.x_test)),
                       'mlpClassifier': accuracy_score(self.y_test, mlpClass.predict(self.x_test)),
                       'randomForestClassifier': accuracy_score(self.y_test, randFor.predict(self.x_test)),
                       }
        models_highest = max(dict_models, key=dict_models.get)
        print(f'Model with the highest accuracy: {models_highest}')
        if models_highest == 'kneighbor':
            self.model = kneighbor
        elif models_highest == 'logisticRegression':
            self.model = log_regress
        elif models_highest == 'gradientBoostingClassifier':
            self.model = grad_class
        elif models_highest == 'mlpClassifier':
            self.model = mlpClass
        elif models_highest == 'randomForestClassifier':
            self.model = randFor
        # Gradclass = GradientBoostingClassifier()
        # Grad_perm = {
        #     'loss' : ['log_loss', 'exponential'],
        #     'learning_rate': np.arange(0.1, .5, 0.1, dtype=float),
        #     'n_estimators': range(100,500,100),
        #     'criterion' : ['friedman_mse', 'squared_error'],
        #     'max_depth': np.arange(1, 5, 1, dtype=int),
        #     'max_features' : [1, 'sqrt', 'log2']
        #     }
        # clf = GridSearchCV(Gradclass, Grad_perm, scoring=['accuracy'],
        #                     refit='accuracy', verbose=4, n_jobs=-1)
        # search_Grad = clf.fit(self.x_train,self.y_train)
        # print('GradientBoostingClassifier - best params: ',search_Grad.best_params_)
        # Gradclass_err = accuracy_score(self.y_test, search_Grad.predict(self.x_test))
        # print('GradientBoostingClassifier accuracy',Gradclass_err)
        print('check the amount of Walks (1) and Non-Walks (0) training  data: ',self.y_train.value_counts())
        print('check the amount of Walks (1) and Non-Walks (0) test  data: ',self.y_test.value_counts())
    def predict_data(self):
        """
        Predict any input data.
        possibly graph the data as well to see what is labeled correctly
        """
        temp_df = self.x.drop(columns=self.drop_cols)
        temp_df['labels'] = np.zeros(len(temp_df))
        for i in range(len(temp_df)):
            temp_df['labels'].iloc[i] = self.model.predict(temp_df.iloc[i].to_numpy().reshape(-1, 1))
            if temp_df['labels'].iloc[i] == 0:
                plt.plot(temp_df.index[i],temp_df['angular_velocity_x'].iloc[i],linestyle='--', marker='o', color='r')
            elif temp_df['labels'].iloc[i] == 1:
                plt.plot(temp_df.index[i],temp_df['angular_velocity_x'].iloc[i],linestyle='--', marker='o', color='g')
        plt.show()
    def feature_importances(self,model):
        pass

def main():
    walk = WalkPredictor()
    walk.readin()
    walk.encode_binary()
    walk.split()
    walk.pre_process()
    walk.learn()
    walk.predict_data()
if __name__=='__main__':
    main()