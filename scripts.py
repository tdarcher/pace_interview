#!/usr/bin/env python

# Code here was used to process the dataset of hotel room prices
# 
# The final fitting function 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
from matplotlib.pyplot import cm
import datetime 
import math
import pickle
import scipy.fftpack


def read_fitting_dataset():
    fname='home_exercise_dataset_52.csv'
    cols=['sales_date','reservation_night_date','hotelroom_id','display_price','type']
    parsedates=['sales_date','reservation_night_date']
    df = pd.read_csv(fname, usecols=cols,parse_dates=parsedates,engine='python' )
    df = df[(df.hotelroom_id != 519) ]
    df = df[(df.hotelroom_id == 520) ]
    df = df.sort_values(by=['reservation_night_date'])
    df = df[(df.type == 'booking') ]
    df = df.reset_index(drop=True)
    return df

def read_fitting_dataset_cancelled():
    fname='home_exercise_dataset_52.csv'
    cols=['sales_date','reservation_night_date','hotelroom_id','display_price','type']
    parsedates=['sales_date','reservation_night_date']
    df = pd.read_csv(fname, usecols=cols,parse_dates=parsedates,engine='python' )
    df = df[(df.hotelroom_id != 519) ]
    df = df[(df.hotelroom_id == 520) ]
    df = df.sort_values(by=['reservation_night_date'])
    df = df[(df.type == 'cancellation') ]
    df = df.reset_index(drop=True)
    return df

def read_capacity():
    fname = 'home_exercise_hotelroom_capacities_52.csv'
    df = pd.read_csv(fname,usecols=['id','capacity'],engine='python')
    df = df[(df.id != 519) ]
    df = df.reset_index(drop=True)
    return df



def plot_year_histogram(what):
    # Plots histogram plots either
    #"all" where each room id has its own series
    #"sum" data from each room i
    capacity=read_capacity()
    df = read_fitting_dataset();
    df_cancelled = read_fitting_dataset_cancelled();
    mindate = datetime.datetime(2017, 3, 6)
    maxdate = mindate +  datetime.timedelta(days=730)
    #maxdate = df.reservation_night_date.max()
    #mindate = df.reservation_night_date.min()
    delta = (maxdate-mindate).days
    hist=np.zeros((delta))
    dates= np.arange(mindate,maxdate, dtype='datetime64[D]')
    dates_df = pd.date_range(mindate, maxdate, freq='D').to_series()
    hist_all=np.zeros((len(capacity.values),delta))
    print capacity
    total=capacity.capacity.sum()
    print total
    
    for i in range(delta):
        date =  mindate +  datetime.timedelta(days=i)
        if what!='all':
            cancelled = df_cancelled[ (df.reservation_night_date == date)].shape[0] 
            booked =    df[(df.reservation_night_date == date) ].shape[0]
            hist[i] = float(booked-cancelled)/total
        if what=='all':
            for j in range(len(capacity.values)) :
                hotelroom_id=capacity.values[j][0]
                cancelled = df_cancelled[(df.hotelroom_id == hotelroom_id) & (df.reservation_night_date == date) ].shape[0]
                booked =    df[(df.hotelroom_id == hotelroom_id) & (df.reservation_night_date == date) ].shape[0]
                hist_all[j][i]=float(booked - cancelled)/capacity.values[j][1]
    if what=='all':
        colors =     ['b','g','r','c','m','y','g','b','k','r','c','m','y','k']
        linestyles = ['--','--','--','--','--','--','-','-','-','-','-','-','-']
        for j in range(len(capacity.values)) :
            plt.plot(dates,hist_all[j], label=capacity.id.values[j],color=colors[j],linestyle=linestyles[j])
        plt.legend()
        plt.gcf().autofmt_xdate()
        plt.ylabel("P(Occupied)")
        plt.show()
    if what=='sum':
        plt.plot(dates, hist)
        plt.gcf().autofmt_xdate()
        plt.ylabel("P(Occupied)")
        plt.show()
    if what=='fft':
        sum_hist=0.0
        for i in range(delta):
            sum_hist+=hist[i]/len(hist)
        hist=hist-sum_hist
        histf=scipy.fftpack.fft(hist)
        N=int(len(histf))
        print N
        xf = np.linspace(0.0, 1.0/(2.0), N/2)
        plt.plot(xf,2.0/N * np.abs(histf[:N/2]))
        plt.title('FFT of total occupancy')
        plt.ylabel("Spectral Amplitude")
        plt.xlabel("Days^-1")
        plt.show()
    


        
def plot_day_histogram():
    capacity=read_capacity()
    df = read_fitting_dataset();
    df_cancelled = read_fitting_dataset_cancelled();
    mindate = datetime.datetime(2017, 3, 6)
    maxdate = mindate +  datetime.timedelta(days=42)
    delta = (maxdate-mindate).days
    hist=np.zeros((delta))
    dates= np.arange(mindate,maxdate, dtype='datetime64[D]')
    dates_df = pd.date_range(mindate, maxdate, freq='D').to_series()
    hist_all=np.zeros((len(capacity.values),delta))
    print capacity
    total=capacity.capacity.sum()
    print total
    
    for i in range(delta):
        date =  mindate +  datetime.timedelta(days=i)
        cancelled = df_cancelled[ (df.reservation_night_date == date)].shape[0] 
        booked =    df[(df.reservation_night_date == date) ].shape[0]
        hist[i] = float(booked-cancelled)/total
    days=dates_df.dt.dayofweek
    hist_days = np.zeros((7))
    error_days = np.zeros((7))
    for j in range(len(hist)):  # Calculate the mean
        day = days[j]
        hist_days[day] +=  hist[j]/(len(hist)/7)
    for j in range(len(hist)):  # calculate the variance
        day = days[j]
        error_days[day] += (hist[j] - hist_days[day]) * (hist[j] - hist_days[day])/(len(hist)/7)
    for j in range(len(error_days)): # sum variance and calculate sd
        error_days[j]=math.sqrt(error_days[j])        
    plt.errorbar(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'],hist_days,yerr=error_days) 
    plt.ylabel("P(Occupied)")
    plt.xlabel("reservation night")
    plt.title('42 day sample')
    plt.show()



 
    
def process_historical_data(what):
    capacity=read_capacity()
    df = read_fitting_dataset()
    df_cancelled = read_fitting_dataset_cancelled()
    mindate = datetime.datetime(2017, 3, 6)
    maxdate = mindate +  datetime.timedelta(days=365)
    delta = (maxdate-mindate).days 
    hist=np.zeros((delta))
    dates= np.arange(mindate,maxdate, dtype='datetime64[D]')
    dates_df = pd.date_range(mindate, maxdate-datetime.timedelta(days=1), freq='D').to_series()
    hist_all=np.zeros((len(capacity.values),delta))
    total=np.sum(capacity.values[:][1])
    for i in range(delta):
        date =  mindate +  datetime.timedelta(days=i)   
        cancelled = df_cancelled[ (df.reservation_night_date == date)].shape[0] 
        booked =    df[(df.reservation_night_date == date) ].shape[0]
        hist[i] = float(booked-cancelled)/total
    days=dates_df.dt.dayofweek
    weeks=dates_df.dt.week
    week_vec= np.arange(0,53)
    hist_weeks = np.zeros((7,53))
    weekdays=['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    for j in range(len(hist)):
        day = days[j]
        week = weeks[j] 
        hist_weeks[day][week] +=  hist[j]
    if what=="plot":
        for day in [0,1,2,3,4,5,6]:
            plt.scatter(week_vec,hist_weeks[day], label=weekdays[day])
        plt.legend()    
        plt.ylabel("P(Occupied)")
        plt.xlabel("Week number")
        plt.title('Week and day breakdown of training data')
        plt.show()
    #save historical data in new format
    y1data=hist_weeks
    f=open('y1data.pickle','w')
    string=pickle.dumps(y1data)
    f.write(string)
    f.close()
    return y1data 



def process_historical_data_variation(what):
    capacity=read_capacity()
    df = read_fitting_dataset()
    df_cancelled = read_fitting_dataset_cancelled()
    mindate = datetime.datetime(2017, 3, 6)
    maxdate = mindate +  datetime.timedelta(days=365)
    delta = (maxdate-mindate).days 
    hist=np.zeros((delta))
    dates= np.arange(mindate,maxdate, dtype='datetime64[D]')
    dates_df = pd.date_range(mindate, maxdate-datetime.timedelta(days=1), freq='D').to_series()
    hist_all=np.zeros((len(capacity.values),delta))
    total=np.sum(capacity.values[:][1])
    sum_hist=0.0
    for i in range(delta):
        date =  mindate +  datetime.timedelta(days=i)   
        cancelled = df_cancelled[ (df.reservation_night_date == date)].shape[0] 
        booked =    df[(df.reservation_night_date == date) ].shape[0]
        hist[i] = float(booked-cancelled)/total
        sum_hist+=hist[i]/len(hist)
    hist=hist-sum_hist
    days=dates_df.dt.dayofweek
    weeks=dates_df.dt.week
    week_vec= np.arange(0,53)
    hist_weeks = np.zeros((7,53))
    weekdays=['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    for j in range(len(hist)):
        day = days[j]
        week = weeks[j] 
        hist_weeks[day][week] +=  hist[j]
    if what=="plot":
        for day in [0,1,2,3,4,5,6]:
            plt.scatter(week_vec,hist_weeks[day], label=weekdays[day])
        plt.legend()    
        plt.ylabel("P(Occupied)")
        plt.xlabel("Week number")
        plt.title('Week and day breakdown of training data')
        plt.show()
    #save historical data in new format
    y1data=hist_weeks
    f=open('y1data.pickle','w')
    string=pickle.dumps(y1data)
    f.write(string)
    f.close()
    return y1data 


def read_model():
    f=open('y1data.pickle','r')
    string=f.read()
    f.close
    y1data=pickle.loads(string)
    return y1data

def read_occ():
    f=open('occ.pickle','r')
    string=f.read()
    f.close
    occ=pickle.loads(string)
    return occ


def test_model1():
    capacity=read_capacity()
    df = read_fitting_dataset()
    df_cancelled = read_fitting_dataset_cancelled()
    mindate = datetime.datetime(2018, 2, 6)
    maxdate = mindate +  datetime.timedelta(days=365)
    delta = (maxdate-mindate).days 
    hist=np.zeros((delta))
    hist_predicted=np.zeros((delta))
    dates= np.arange(mindate,maxdate, dtype='datetime64[D]')
    dates_df = pd.date_range(mindate, maxdate-datetime.timedelta(days=1), freq='D').to_series()
    total=np.sum(capacity.values[7][1])

    for i in range(delta):
        date =  mindate +  datetime.timedelta(days=i)   
        cancelled = df_cancelled[ (df.reservation_night_date == date)].shape[0] 
        booked =    df[(df.reservation_night_date == date) ].shape[0]
        hist[i] = float(booked-cancelled)/total
    days=dates_df.dt.dayofweek
    weeks=dates_df.dt.week
    y1data=read_model()
    occ=read_occ()
    diff = 0.0
    for j in range(15,len(hist)):
        av=0.0
        for p in range(j-15,j-1):
            av += hist[p]/14
        day = days[j]
        week = weeks[j]
        hist_predicted[j]=y1data[day][week] + av
        if hist_predicted[j]>1:
            hist_predicted[j]=1
        if hist[j]>1:
            hist[j]=1
        diff += math.sqrt((hist_predicted[j]-hist[j])*(hist_predicted[j]-hist[j]))/365
    print "residual = "
    print diff
    plt.plot(dates,hist,'k',label='Recorded')
    plt.plot(dates,hist_predicted,'r--',label='Predicted')
    plt.gcf().autofmt_xdate()
    plt.ylabel("P(Occupied)")
    plt.title('Hotel 520')
    plt.legend() 
    plt.show()
    
        


plot_year_histogram('fft')
