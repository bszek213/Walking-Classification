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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
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
        #TODO: these are all matrices wtih x, y, and z axes. change this to a dataframe with each feature being an axis
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
        # Find features with correlation greater than 0.90
        corr_matrix = np.abs(self.x_train.astype(float).corr())
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] >= 0.88)]
        self.x_train = self.x_train.drop(columns=to_drop)
        self.x_test = self.x_test.drop(columns=to_drop)
        print(f'features to drop: {to_drop}')
        #plot heat map
        top_corr_features = corr_matrix.index
        plt.figure(figsize=(20,20))
        plt.title('Correlation matrix of all t265 data')
        g=sns.heatmap(corr_matrix[top_corr_features],annot=True,cmap="RdYlGn")
        plt.savefig('correlations.png')
        plt.close()
        pass
    def learn(self):
        Gradclass = GradientBoostingClassifier()
        Grad_perm = {
            'loss' : ['log_loss', 'exponential'],
            'learning_rate': np.arange(0.1, .5, 0.1, dtype=float),
            'n_estimators': range(100,500,100),
            'criterion' : ['friedman_mse', 'squared_error'],
            'max_depth': np.arange(1, 5, 1, dtype=int),
            'max_features' : [1, 'sqrt', 'log2']
            }
        clf = GridSearchCV(Gradclass, Grad_perm, scoring=['accuracy'],
                            refit='accuracy', verbose=4, n_jobs=-1)
        search_Grad = clf.fit(self.x_train,self.y_train)
        print('GradientBoostingClassifier - best params: ',search_Grad.best_params_)
        Gradclass_err = accuracy_score(self.y_test, search_Grad.predict(self.x_test))
        print('GradientBoostingClassifier accuracy',Gradclass_err)
        pass
    def prob_plots(self,col_name):
        pass
    def feature_importances(self,model):
        pass

def main():
    walk = WalkPredictor()
    walk.readin()
    walk.encode_binary()
    walk.split()
    walk.pre_process()
    walk.learn()
if __name__=='__main__':
    main()
