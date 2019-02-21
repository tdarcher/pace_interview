#!/usr/bin/env python

# The function predict() takes a pandas vector containing dates and
# the historical avarage as arguments.  
# a numpy vector contining probability for each date is returned
############## usage example ############################### 
#mindate = datetime.datetime(2017, 3, 6)
#maxdate = mindate +  datetime.timedelta(days=100)
#dates=pd.date_range(mindate, maxdate, freq='D').to_series()
#av=0.5
#probs=predict(dates,av)
#print probs




import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import datetime 
import pickle



def read_model():
    f=open('y1data.pickle','r')
    string=f.read()
    f.close
    y1data=pickle.loads(string)
    return y1data


def predict(dates,av):
    # dates is a pandas data frame containing datetime vaues
    # av should be the mean occupancy for the hotel of interest

    y1data=read_model()
    days=dates.dt.dayofweek
    weeks=dates.dt.week
    hist_predicted=np.zeros((len(dates)))
    for j in range(len(dates)):
        day = days[j]
        week = weeks[j]
        hist_predicted[j]=y1data[day][week] + av
        # prevent occupancy > 1 or < 0
        if hist_predicted[j]>1:
            hist_predicted[j]=1
        if hist_predicted[j]<0.0:
            hist_predicted[j]=0.0            
    return hist_predicted    


