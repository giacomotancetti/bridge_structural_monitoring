# -*- coding: utf-8 -*-
"""
Created on Thu May 16 08:35:55 2019

@author: giacomo tancetti
"""

import pandas as pd
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats

# read CSV files in folder "download" and store data into DataFrame
def ReadCSV(path,l_csv):
    
    df_letture=pd.DataFrame(columns=["Data/Ora"])
    df_letture=df_letture.set_index("Data/Ora")
    d_letture={}
    
    for csv_file in l_csv:
        csv_file=csv_file.strip()

        df_excel=pd.read_csv(path+'/'+csv_file,decimal=',',sep=';')

        # add n_centralina column data
        n_centralina=csv_file[:-4]
        
        # purge columns names from units 
        l_col=[]
        l_col_units=df_excel.columns.tolist()
        for name in l_col_units:    
            name = name.split()[0]   
            l_col.append(name)
        df_excel.columns=l_col
        
        l_datetime_obj=[]      
        for elem in df_excel["Data/Ora"]:
            datetime_obj = datetime.strptime(elem, '%d/%m/%Y %H:%M:%S')
            l_datetime_obj.append(datetime_obj)
        
        df_excel["Data/Ora"]=l_datetime_obj
        df_excel=df_excel.set_index("Data/Ora")
        df_excel=df_excel.sort_index()
        
        # d_letture[n_centralina:df_letture]
        d_letture[n_centralina]=df_excel
    
    return(d_letture)
     
# create dictionary ZeroLett with zero measurement    
def ZeroLett(d_letture):
    d_letture_zero={}
    
    for n_centralina in d_letture.keys():
        df_letture=d_letture[n_centralina]
        df_letture=df_letture.sort_index()
        d_letture_zero[n_centralina]=df_letture.iloc[0]

    return(d_letture_zero)    

# calculate delta between zero lecture and time t lecture 
def Delta(d_letture,d_letture_zero):
    d_delta_instr={}
    
    for n_centralina in d_letture.keys():
        df_delta=d_letture[n_centralina]-d_letture_zero[n_centralina]
        d_delta_instr[n_centralina]=df_delta
    
    return(d_delta_instr)
    
# calculate daily average temperature
def DailyAverageTemp(d_delta):
    zero_temp={'TEMP1N':19.4,'TEMP1S':20.8,'TEMP2N':19.2,'TEMP2S':20.6,
               'TEMP3N':20.0,'TEMP3S':22.7,'TEMP4N':18.5,'TEMP4S':18.0}
    df_temp=pd.DataFrame()
    for n_centralina in d_delta.keys():
        df_delta_davg = d_delta[n_centralina].resample('D').mean()
        l_col_temp=[n_col for n_col in df_delta_davg.columns if 'TEMP' in n_col]
        for col_temp in l_col_temp:
            temp=df_delta_davg[col_temp]+zero_temp[col_temp]
            df_temp=pd.concat([df_temp, temp], axis=1)
    
    s_temp=df_temp.mean(axis = 1, skipna = True)
    
    return(s_temp)

# create dictionary with instruments location names
def AssignLocationName(d_delta_instr):
    d_loc_names={'CE1N':{'FESS1':'FESS_GIUNTO_1','FESS2':'FESS_GIUNTO_2'},
             'CE1S':{'FESS1':'FESS_GIUNTO_1','FESS2':'FESS_GIUNTO_2','FESS100':'FESS_MURO_DX_SPALLA','INCL1':'INCL_SPALLA_1','INCL2':'INCL_PILA_1','INCL3':'INCL_PILA_2'},
             'CE2N':{'FESS3':'FESS_GIUNTO_3','FESS4':'FESS_GIUNTO_4'},
             'CE2S':{'FESS3':'FESS_GIUNTO_3','FESS4':'FESS_GIUNTO_4','INCL4':'INCL_PILA_3','INCL5':'INCL_PILA_4'},
             'CE3N':{'FESS5':'FESS_GIUNTO_5','FESS6':'FESS_GIUNTO_6'},
             'CE3S':{'FESS5':'FESS_GIUNTO_5','FESS6':'FESS_GIUNTO_6','INCL6':'INCL_PILA_5'},
             'CE4N':{'FESS7':'FESS_GIUNTO_7','FESS8':'FESS_GIUNTO_8'},
             'CE4S':{'FESS7':'FESS_GIUNTO_7','FESS8':'FESS_GIUNTO_8','INCL7':'INCL_PILA_6','INCL8':'INCL_PILA_7','INCL9':'INCL_SPALLA_2'}}
    
    d_delta=d_delta_instr.copy()
    for n_centralina in d_loc_names.keys():
        for n_str in d_loc_names[n_centralina].keys():
            loc_name=d_loc_names[n_centralina][n_str]
            for n_col in d_delta[n_centralina].columns:
                if n_str in n_col:
                    new_name=n_col.replace(n_str,loc_name)
                    d_delta[n_centralina]=d_delta[n_centralina].rename(columns = {n_col:new_name})
                    
    return(d_delta)
        
#calculate Ddaily average of each measure   
def DailyAverage(d_delta):
    d_delta_av={}
    l_days_meas=[]
    
    for n_centralina in d_delta.keys():
        l_days_meas=[]
        l_datetime_meas=d_delta[n_centralina].index.tolist()
        
        for meas_time in l_datetime_meas:
            if meas_time.date() not in l_days_meas:
                l_days_meas.append(meas_time.date())
        
        # initialize DataFrame average delta
        df_delta_av=pd.DataFrame(columns=d_delta[n_centralina].columns)
        
        for day_meas in l_days_meas:
            s_av=d_delta[n_centralina][d_delta[n_centralina].index.date==day_meas].mean()
            df_delta_av=df_delta_av.append(s_av,ignore_index=True)
        
        df_delta_av['date']=l_days_meas
        df_delta_av=df_delta_av.set_index('date')
        d_delta_av[n_centralina]=df_delta_av
        
    return(d_delta_av)
    
#calculate delta Moving Average of each measure   
def DeltaMovingAverage(d_delta):
    d_delta_av_t={}
    
    for n_centralina in d_delta.keys():
        df_delta_i=d_delta[n_centralina].rolling(3).mean()
        
        d_delta_av_t[n_centralina]=df_delta_i
        
    return(d_delta_av_t)
    
def Fit_T_delta(d_delta):
    d_fit_par={}
    ref_date=datetime(2018,12,3)
    d_ref_dates={'CE1N':{'FESS_GIUNTO_1XN':ref_date,'FESS_GIUNTO_1YN':ref_date,'FESS_GIUNTO_1ZN':ref_date,'FESS_GIUNTO_2XN':ref_date,'FESS_GIUNTO_2YN':ref_date,'FESS_GIUNTO_2ZN':ref_date,'TEMP1N':ref_date},
                 'CE1S':{'FESS_GIUNTO_1XS':ref_date,'FESS_GIUNTO_1YS':ref_date,'FESS_GIUNTO_1ZS':ref_date,'FESS_GIUNTO_2XS':ref_date,'FESS_GIUNTO_2YS':ref_date,'FESS_GIUNTO_2ZS':ref_date,'FESS_GIUNTO_100P':ref_date,'FESS_MURO_DX_SPALLA':ref_date,'INCL_SPALLA_1Y':datetime(2018,10,13),'INCL_SPALLA_1X':ref_date,'INCL_PILA_1Y':ref_date,'INCL_PILA_1X':ref_date,'INCL_PILA_2Y':datetime(2019,4,30),'INCL_PILA_2X':ref_date,'TEMP1S':ref_date},
                 'CE2N':{'FESS_GIUNTO_3XN':ref_date,'FESS_GIUNTO_3YN':ref_date,'FESS_GIUNTO_3ZN':ref_date,'FESS_GIUNTO_4XN':ref_date,'FESS_GIUNTO_4YN':ref_date,'FESS_GIUNTO_4ZN':ref_date,'TEMP2N':ref_date},
                 'CE2S':{'FESS_GIUNTO_3XS':ref_date,'FESS_GIUNTO_3YS':ref_date,'FESS_GIUNTO_3ZS':ref_date,'FESS_GIUNTO_4XS':ref_date,'FESS_GIUNTO_4YS':ref_date,'FESS_GIUNTO_4ZS':ref_date,'INCL_PILA_3X':ref_date,'INCL_PILA_3Y':ref_date,'INCL_PILA_4X':ref_date,'INCL_PILA_4Y':ref_date,'TEMP2S':ref_date},
                 'CE3N':{'FESS_GIUNTO_5XN':ref_date,'FESS_GIUNTO_5YN':ref_date,'FESS_GIUNTO_5ZN':ref_date,'FESS_GIUNTO_6XN':ref_date,'FESS_GIUNTO_6YN':ref_date,'FESS_GIUNTO_6ZN':ref_date,'TEMP3N':ref_date},
                 'CE3S':{'FESS_GIUNTO_5XS':ref_date,'FESS_GIUNTO_5YS':ref_date,'FESS_GIUNTO_5ZS':ref_date,'FESS_GIUNTO_6XS':ref_date,'FESS_GIUNTO_6YS':ref_date,'FESS_GIUNTO_6ZS':ref_date,'INCL_PILA_5X':ref_date,'INCL_PILA_5Y':ref_date,'TEMP3S':ref_date},
                 'CE4N':{'FESS_GIUNTO_7XN':ref_date,'FESS_GIUNTO_7YN':ref_date,'FESS_GIUNTO_7ZN':ref_date,'FESS_GIUNTO_8XN':ref_date,'FESS_GIUNTO_8YN':ref_date,'FESS_GIUNTO_8ZN':ref_date,'TEMP4N':ref_date},
                 'CE4S':{'FESS_GIUNTO_7XS':ref_date,'FESS_GIUNTO_7YS':ref_date,'FESS_GIUNTO_7ZS':ref_date,'FESS_GIUNTO_8XS':ref_date,'FESS_GIUNTO_8YS':ref_date,'FESS_GIUNTO_8ZS':ref_date,'INCL_PILA_6X':ref_date,'INCL_PILA_6Y':ref_date,'INCL_PILA_7X':ref_date,'INCL_PILA_7Y':ref_date,'INCL_SPALLA_2X':ref_date,'INCL_SPALLA_2Y':ref_date,'TEMP4S':ref_date}}

    for n_centralina in d_delta.keys():
        # initialize DataFrame fit parameters
        df_fit_par=pd.DataFrame(columns=d_delta[n_centralina].columns)
        for column_name in d_delta[n_centralina].columns:
            # find temperature column
            if "TEMP"==column_name[:4]:
                T_col_name=column_name
                
        for column_name in d_delta[n_centralina].columns:
            #find reference date for interpolation
            ref_date_i=d_ref_dates[n_centralina][column_name]
            x=d_delta[n_centralina][T_col_name][d_delta[n_centralina].index<ref_date_i].values  
            y=d_delta[n_centralina][column_name][d_delta[n_centralina].index<ref_date_i].values
            z=np.polyfit(x,y,1)
            df_fit_par[column_name]=z
        
        d_fit_par[n_centralina]=df_fit_par
        
    return(d_fit_par)
    
def Delta_T_comp(d_delta,d_fit_par):
    d_delta_comp={}
   
    for n_centralina in d_delta.keys():        
        # initialize DataFrame average delta
        df_delta_comp=pd.DataFrame(columns=d_delta[n_centralina].columns, index=d_delta[n_centralina].index)
        # find column temperature name
        for column_name in d_delta[n_centralina].columns:
            if "TEMP"==column_name[:4]:
                T_col_name=column_name
               
        for column_name in d_delta[n_centralina].columns:
            lett_temp=d_delta[n_centralina][T_col_name].values            
            lett=d_delta[n_centralina][column_name].values        
            lett_comp=lett-d_fit_par[n_centralina][column_name][0]*lett_temp
            df_delta_comp[column_name]=lett_comp
            
        d_delta_comp[n_centralina]=df_delta_comp
        
    return(d_delta_comp)
  
def Plot_t_delta_incl(d_delta,d_delta_av,d_delta_comp):
       
    for n_centralina in d_delta:
        # find temperature column name and tiltimeter columns names
        l_col_incl=[]
        l_n_incl=[]
        for column_name in d_delta[n_centralina].columns:
            if "TEMP"==column_name[:4]:
                T_col_name=column_name
            elif "INCL"==column_name[:4]:
                l_col_incl.append(column_name)
                n_incl=int(''.join(x for x in column_name if x.isdigit()))
                if n_incl not in l_n_incl:
                    l_n_incl.append(n_incl)
                    
        if l_n_incl:               # check if list is not empty
            for n_incl in l_n_incl:
                l_data=[name for name in l_col_incl if str(n_incl) in name]
                l_data_pila=[name for name in l_data if "PILA" in name]
                l_data_spalla=[name for name in l_data if "SPALLA" in name]

                if l_data_pila:
                    t=d_delta[n_centralina][l_data_pila[0]].index
                    t_av=d_delta_av[n_centralina][l_data_pila[0]].index
                    data1 = d_delta[n_centralina][l_data_pila[0]]
                    data2 = d_delta[n_centralina][l_data_pila[1]]
                    data3=d_delta_av[n_centralina][l_data_pila[0]]
                    data4=d_delta_av[n_centralina][l_data_pila[1]]
                    #data5=d_delta_comp[n_centralina][l_data_pila[0]]
                    #data6=d_delta_comp[n_centralina][l_data_pila[1]]
                    dataT = d_delta[n_centralina][T_col_name].values
                
                    fig, ax1 = plt.subplots()
                    color0 = 'tab:blue'
                    color1 = 'tab:orange'
                    ax1.set_xlabel('time')
                    ax1.set_ylabel('rotation [°]')
                    ax1.plot(t, data1, color=color0, label=l_data_pila[0],linewidth=0.5)
                    ax1.plot(t, data2, color=color1, label=l_data_pila[1],linewidth=0.5)
                    ax1.plot(t_av, data3, color=color0, label=l_data_pila[0]+'av',linewidth=1.0)
                    ax1.plot(t_av, data4, color=color1, label=l_data_pila[1]+'av',linewidth=1.0)
                    #ax1.plot(t, data5, color=color0, label=l_data_pila[0]+'comp',linewidth=1.0,linestyle="--")
                    #ax1.plot(t, data6, color=color1, label=l_data_pila[1]+'comp',linewidth=1.0,linestyle="--")
                    ax1.tick_params(axis='y',labelsize=10,labelrotation=0)
                    ax1.tick_params(axis='x',labelsize=10,labelrotation=-90)

                    # set y axis limits            
                    ax1.set_ylim([-0.2,0.2])
                    
                    plt.grid(True, which='major',axis='both', linestyle='--',dashes=[10, 10], linewidth=0.5)
                    leg=plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                                   ncol=2, mode="expand", borderaxespad=0.,fontsize=10)
               
                    for line in leg.get_lines():
                        line.set_linewidth(2)
            
                    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            
                    color = 'tab:grey'
                    ax2.set_ylabel('T [°C]', color=color) 
                    ax2.plot(t, dataT, color=color, label="TEMPERATURE", linewidth=0.5,alpha=0.4)
                    ax2.tick_params(axis='y', labelcolor=color)
                    ax2.set_ylim([-25,25])
                    
                    fig.tight_layout()
                    fig.set_size_inches((16,9))
                    fname=l_data_pila[0][:-1]
                    fig.canvas.set_window_title(l_data_pila[0][:-1])
                    
                    #plt.show()
                    plt.savefig(fname, dpi=300, facecolor='w', edgecolor='w',
                                orientation='landscape', papertype=None, format=None,
                                transparent=False, bbox_inches='tight', pad_inches=None,
                                frameon=None, metadata=None)
                    
                if l_data_spalla:
                    t=d_delta[n_centralina][l_data_spalla[0]].index
                    t_av=d_delta_av[n_centralina][l_data_spalla[0]].index
                    data1 = d_delta[n_centralina][l_data_spalla[0]]
                    data2 = d_delta[n_centralina][l_data_spalla[1]]
                    data3=d_delta_av[n_centralina][l_data_spalla[0]]
                    data4=d_delta_av[n_centralina][l_data_spalla[1]]
                    #data5=d_delta_comp[n_centralina][l_data_spalla[0]]
                    #data6=d_delta_comp[n_centralina][l_data_spalla[1]]
                    dataT = d_delta[n_centralina][T_col_name].values
                
                    fig, ax1 = plt.subplots()
                    color0 = 'tab:blue'
                    color1 = 'tab:orange'
                    ax1.set_xlabel('time')
                    ax1.set_ylabel('rotation [°]')
                    ax1.plot(t, data1, color=color0, label=l_data_spalla[0],linewidth=0.5)
                    ax1.plot(t, data2, color=color1, label=l_data_spalla[1],linewidth=0.5)
                    ax1.plot(t_av, data3, color=color0, label=l_data_spalla[0]+'av',linewidth=1.0)
                    ax1.plot(t_av, data4, color=color1, label=l_data_spalla[1]+'av',linewidth=1.0)
                    #ax1.plot(t, data5, color=color0, label=l_data_spalla[0]+'comp',linewidth=1.0,linestyle="--")
                    #ax1.plot(t, data6, color=color1, label=l_data_spalla[1]+'comp',linewidth=1.0,linestyle="--")
                    ax1.tick_params(axis='y',labelsize=10,labelrotation=0)
                    ax1.tick_params(axis='x',labelsize=10,labelrotation=-90)

                    # set y axis limits            
                    ax1.set_ylim([-0.2,0.2])
                    
                    plt.grid(True, which='major',axis='both', linestyle='--',dashes=[10, 10], linewidth=0.5)
                    leg=plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                                   ncol=2, mode="expand", borderaxespad=0.,fontsize=10)
               
                    for line in leg.get_lines():
                        line.set_linewidth(2)
            
                    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            
                    color = 'tab:grey'
                    ax2.set_ylabel('T [°C]', color=color) 
                    ax2.plot(t, dataT, color=color, label="TEMPERATURE", linewidth=0.5,alpha=0.4)
                    ax2.tick_params(axis='y',labelsize=10, labelcolor=color)
                    ax2.set_ylim([-25,25])
                    
                    fig.tight_layout()
                    fig.set_size_inches(16,9)
                    fig.canvas.set_window_title(l_data_spalla[0][:-1])
                    plt.show()
                    fname=l_data_spalla[0][:-1]
                    plt.savefig(fname, dpi=300, facecolor='w', edgecolor='w',
                                orientation='landscape', papertype=None, format=None,
                                transparent=False, bbox_inches='tight', pad_inches=None,
                                frameon=None, metadata=None)

# plot time - inclinometer delta compensed by temperature effect                    
def Plot_t_delta_comp_incl(d_delta_comp, d_delta_av_comp):
       
    for n_centralina in d_delta_comp:
        # find temperature column name and tiltimeter columns names
        l_col_incl=[]
        l_n_incl=[]
        for column_name in d_delta_comp[n_centralina].columns:
            if "TEMP"==column_name[:4]:
                T_col_name=column_name
            elif "INCL"==column_name[:4]:
                l_col_incl.append(column_name)
                n_incl=int(''.join(x for x in column_name if x.isdigit()))
                if n_incl not in l_n_incl:
                    l_n_incl.append(n_incl)
                    
        if l_n_incl:                    # check if list is not empty
            for n_incl in l_n_incl:
                l_data=[name for name in l_col_incl if str(n_incl) in name]
                l_data_pila=[name for name in l_data if "PILA" in name]
                l_data_spalla=[name for name in l_data if "SPALLA" in name]

                if l_data_pila:
                    t=d_delta_comp[n_centralina][l_data_pila[0]].index
                    t_av=d_delta_av_comp[n_centralina][l_data_pila[0]].index
                    data1 = d_delta_comp[n_centralina][l_data_pila[0]]
                    data2 = d_delta_comp[n_centralina][l_data_pila[1]]
                    data3=d_delta_av_comp[n_centralina][l_data_pila[0]]
                    data4=d_delta_av_comp[n_centralina][l_data_pila[1]]
                    dataT = d_delta_comp[n_centralina][T_col_name].values
                
                    fig, ax1 = plt.subplots()
                    color0 = 'tab:blue'
                    color1 = 'tab:orange'
                    ax1.set_xlabel('time')
                    ax1.set_ylabel('rotation [°]')
                    ax1.plot(t, data1, color=color0, label=l_data_pila[0]+'comp',linewidth=0.5)
                    ax1.plot(t, data2, color=color1, label=l_data_pila[1]+'comp',linewidth=0.5)
                    ax1.plot(t_av, data3, color=color0, label=l_data_pila[0]+'av'+'comp',linewidth=1.0)
                    ax1.plot(t_av, data4, color=color1, label=l_data_pila[1]+'av'+'comp',linewidth=1.0)
                    ax1.tick_params(axis='y',labelsize=10,labelrotation=0)
                    ax1.tick_params(axis='x',labelsize=10,labelrotation=-90)

                    # set y axis limits            
                    ax1.set_ylim([-0.2,0.2])
                    
                    plt.grid(True, which='major',axis='both', linestyle='--',dashes=[10, 10], linewidth=0.5)
                    leg=plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                                   ncol=2, mode="expand", borderaxespad=0.,fontsize=10)
               
                    for line in leg.get_lines():
                        line.set_linewidth(2)
                    '''
                    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
                    
                    color = 'tab:grey'
                    ax2.set_ylabel('T [°C]', color=color) 
                    ax2.plot(t, dataT, color=color, label="TEMPERATURE", linewidth=0.5,alpha=0.4)
                    ax2.tick_params(axis='y', labelcolor=color)
                    ax2.set_ylim([-25,25])
                    '''
                    fig.tight_layout()
                    fig.set_size_inches(16,9)
                    fig.canvas.set_window_title(l_data_pila[0][:-1]+"comp")
                    fname=l_data_pila[0][:-1]+"comp"
                    plt.savefig(fname, dpi=300, facecolor='w', edgecolor='w',
                                orientation='landscape', papertype=None, format=None,
                                transparent=False, bbox_inches='tight', pad_inches=None,
                                frameon=None, metadata=None)
                    plt.show()
                    
                if l_data_spalla:
                    t=d_delta_comp[n_centralina][l_data_spalla[0]].index
                    t_av=d_delta_av_comp[n_centralina][l_data_spalla[0]].index
                    data1 = d_delta_comp[n_centralina][l_data_spalla[0]]
                    data2 = d_delta_comp[n_centralina][l_data_spalla[1]]
                    data3=d_delta_av_comp[n_centralina][l_data_spalla[0]]
                    data4=d_delta_av_comp[n_centralina][l_data_spalla[1]]
                    dataT = d_delta_comp[n_centralina][T_col_name].values
                
                    fig, ax1 = plt.subplots()
                    color0 = 'tab:blue'
                    color1 = 'tab:orange'
                    ax1.set_xlabel('time')
                    ax1.set_ylabel('rotation [°]')
                    ax1.plot(t, data1, color=color0, label=l_data_spalla[0]+'comp',linewidth=0.5)
                    ax1.plot(t, data2, color=color1, label=l_data_spalla[1]+'comp',linewidth=0.5)
                    ax1.plot(t_av, data3, color=color0, label=l_data_spalla[0]+'av'+'comp',linewidth=1.0)
                    ax1.plot(t_av, data4, color=color1, label=l_data_spalla[1]+'av'+'comp',linewidth=1.0)
                    ax1.tick_params(axis='y',labelsize=10,labelrotation=0)
                    ax1.tick_params(axis='x',labelsize=10,labelrotation=-90)

                    # set y axis limits            
                    ax1.set_ylim([-0.2,0.2])
                    
                    plt.grid(True, which='major',axis='both', linestyle='--',dashes=[10, 10], linewidth=0.5)
                    leg=plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                                   ncol=2, mode="expand", borderaxespad=0.,fontsize=8)
               
                    for line in leg.get_lines():
                        line.set_linewidth(2)
                    '''
                    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            
                    color = 'tab:grey'
                    ax2.set_ylabel('T [°C]', color=color) 
                    ax2.plot(t, dataT, color=color, label="TEMPERATURE", linewidth=0.5,alpha=0.4)
                    ax2.tick_params(axis='y', labelcolor=color)
                    ax2.set_ylim([-25,25])
                    '''
                    fig.canvas.set_window_title(l_data_spalla[0][:-1]+"comp")
                    fig.tight_layout()
                    fig.set_size_inches(16,9)
                    fname=l_data_spalla[0][:-1]+"comp"
                    plt.savefig(fname, dpi=300, facecolor='w', edgecolor='w',
                                orientation='landscape', papertype=None, format=None,
                                transparent=False, bbox_inches='tight', pad_inches=None,
                                frameon=None, metadata=None)
                    plt.show()
                                      
# calculate Paerson index
def PearsonCorr(d_delta):
    d_coeff_Pear={}
    for n_centralina in d_delta:
        df_coeff=pd.DataFrame(columns=d_delta[n_centralina].columns)
        # find temperature column name and crack width gauge columns names 
        l_col_fess=[]
        l_n_fess=[]
        for column_name in d_delta[n_centralina].columns:
            if "TEMP"==column_name[:4]:
                T_col_name=column_name
            elif "FESS"==column_name[:4]:
                l_col_fess.append(column_name)
                n_fess=int(''.join(x for x in column_name if x.isdigit()))
                if n_fess not in l_n_fess:
                    l_n_fess.append(n_fess)
                    
        if l_n_fess:     # check if list is not empty
            for n_fess in l_n_fess:
                if n_fess != 100:
                    l_data=[name for name in l_col_fess if str(n_fess) in name]  
                    T = d_delta[n_centralina][T_col_name]
                    data1 = d_delta[n_centralina][l_data[0]]
                    data2 = d_delta[n_centralina][l_data[1]]
                    data3 = d_delta[n_centralina][l_data[2]]
                    c1=scipy.stats.pearsonr(T, data1)
                    c2=scipy.stats.pearsonr(T, data2)
                    c3=scipy.stats.pearsonr(T, data3)
                    df_coeff[l_data[0]]=c1
                    df_coeff[l_data[1]]=c2
                    df_coeff[l_data[2]]=c3
                    
            d_coeff_Pear[n_centralina]=df_coeff
            
    return(d_coeff_Pear)


# plot time delta displacement transducer
def Plot_t_delta_fess(d_delta,d_delta_av,d_delta_comp,d_coeff_Pear):

    for n_centralina in d_delta:        
        # find temperature column name and crack width gauge columns names 
        l_col_fess=[]
        l_n_fess=[]
        for column_name in d_delta[n_centralina].columns:
            if "TEMP"==column_name[:4]:
                T_col_name=column_name
            elif "FESS"==column_name[:4]:
                l_col_fess.append(column_name)
                n_fess=int(''.join(x for x in column_name if x.isdigit()))
                if n_fess not in l_n_fess:
                    l_n_fess.append(n_fess)
        if l_n_fess:     # check if list is not empty
            for n_fess in l_n_fess:
                if n_fess != 100:
                    l_data=[name for name in l_col_fess if str(n_fess) in name]
                    t = d_delta[n_centralina][l_data[0]].index
                    t_av=d_delta_av[n_centralina][l_data[0]].index
                    data1 = d_delta[n_centralina][l_data[0]]
                    data2 = d_delta[n_centralina][l_data[1]]
                    data3 = d_delta[n_centralina][l_data[2]]
                    data4 = d_delta_av[n_centralina][l_data[0]]
                    data5 = d_delta_av[n_centralina][l_data[1]]
                    data6 = d_delta_av[n_centralina][l_data[2]]
                    dataT = d_delta[n_centralina][T_col_name].values
              
                    fig, ax1 = plt.subplots()

                    color = 'tab:orange'
                    ax1.set_xlabel('time')
                    ax1.set_ylabel('displacement [mm]')
                    ax1.plot(t, data1, color=color, label=l_data[0],linewidth=0.5)
                    ax1.plot(t_av, data4, color=color, label=l_data[0]+'av',linewidth=0.5)
                    ax1.tick_params(axis='y', labelcolor=color)

                    color = 'tab:blue'
                    ax1.set_xlabel('time')
                    ax1.set_ylabel('displacement [mm]')
                    ax1.plot(t, data2, color=color, label=l_data[1],linewidth=0.5)
                    ax1.plot(t_av, data5, color=color, label=l_data[1]+'av',linewidth=0.5)
                    ax1.tick_params(axis='y', labelcolor=color)
            
                    color = 'tab:green'
                    ax1.set_xlabel('time')
                    ax1.set_ylabel('displacement [mm]')
                    ax1.plot(t, data3, color=color, label=l_data[2],linewidth=0.5)
                    ax1.plot(t_av, data6, color=color, label=l_data[2]+'av',linewidth=0.5)
                    ax1.tick_params(axis='y', labelcolor=color)
            
                    ax1.set_ylim([-40,40])
                    
                    textstr1=l_data[0]+' '+'c_Pear='+str(round(d_coeff_Pear[n_centralina][l_data[0]][0],2))
                    textstr2=l_data[1]+' '+'c_Pear='+str(round(d_coeff_Pear[n_centralina][l_data[1]][0],2))
                    textstr3=l_data[2]+' '+'c_Pear='+str(round(d_coeff_Pear[n_centralina][l_data[2]][0],2))
                    ax1.text(0.05, 0.95,'%s\n%s\n%s'%(textstr1,textstr2,textstr3),
                             transform=ax1.transAxes,fontsize=10,
                             verticalalignment='top')
        
                    plt.grid(True, which='major',axis='both', linestyle='--',dashes=[10, 10], linewidth=0.5)
                    leg=plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                               ncol=3, mode="expand", borderaxespad=0.,fontsize=10)
                    
                    for line in leg.get_lines():
                        line.set_linewidth(2)
       
                    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

                    color = 'tab:grey'
                    ax2.set_ylabel('T [°C]', color=color) 
                    ax2.plot(t, dataT, color=color, label="TEMPERATURE", linewidth=0.5, alpha=0.4)
                    ax2.tick_params(axis='y', labelcolor=color)
                    ax2.set_ylim([-30,30])

                    fig.canvas.set_window_title(l_data[0][:-2]+l_data[0][-1])
                    fig.tight_layout()
                    fig.set_size_inches((16,9))
                    fname=l_data[0][:-2]+l_data[0][-1]
                    plt.savefig(fname, dpi=300, facecolor='w', edgecolor='w',
                                orientation='landscape', papertype=None, format=None,
                                transparent=False, bbox_inches='tight', pad_inches=None,
                                frameon=None, metadata=None)
                    plt.show()
                    
                if n_fess == 100:
                    l_data=[name for name in l_col_fess if str(n_fess) in name]
                    n_data=l_data[0]
                    t = d_delta[n_centralina][n_data].index
                    t_av=d_delta_av[n_centralina][n_data].index
                    data1 = d_delta[n_centralina][n_data]
                    data2 = d_delta_av[n_centralina][n_data]
                    dataT = d_delta[n_centralina][T_col_name].values
                    
                    fig, ax1 = plt.subplots()

                    color = 'tab:orange'
                    ax1.set_xlabel('time')
                    ax1.set_ylabel('displacement [mm]')
                    ax1.plot(t, data1, color=color, label=l_data[0],linewidth=0.5)
                    ax1.plot(t_av, data2, color=color, label=l_data[0]+'av',linewidth=0.5)
                    ax1.tick_params(axis='y', labelcolor=color)
                    
                    ax1.set_ylim([-40,40])
                    
                    plt.grid(True, which='major',axis='both', linestyle='--',dashes=[10, 10], linewidth=0.5)
                    leg=plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                               ncol=3, mode="expand", borderaxespad=0.,fontsize=10)
                    
                    for line in leg.get_lines():
                        line.set_linewidth(2)
       
                    ax2 = ax1.twinx()
                    
                    color = 'tab:grey'
                    ax2.set_ylabel('T [°C]', color=color) 
                    ax2.plot(t, dataT, color=color, label="TEMPERATURE", linewidth=0.5, alpha=0.4)
                    ax2.tick_params(axis='y', labelcolor=color)
                    ax2.set_ylim([-30,30])

                    fig.canvas.set_window_title(n_data)
                    fig.tight_layout()
                    fig.set_size_inches((16,9))
                    fname=n_data
                    plt.savefig(fname, dpi=300, facecolor='w', edgecolor='w',
                                orientation='landscape', papertype=None, format=None,
                                transparent=False, bbox_inches='tight', pad_inches=None,
                                frameon=None, metadata=None)
                    plt.show()

# plot temperature - delta scatter graph                   
def Plot_T_delta(d_delta,d_delta_av,d_fit_par):
    
    for n_centralina in d_delta:
        
        # find temperature column name and tiltimeter columns names 
        l_col_incl=[]
        l_n_incl=[]
        for column_name in d_delta[n_centralina].columns:
            if "TEMP"==column_name[:4]:
                T_col_name=column_name
            elif "INCL"==column_name[:4]:
                l_col_incl.append(column_name)
                n_incl=int(''.join(x for x in column_name if x.isdigit()))
                if n_incl not in l_n_incl:
                    l_n_incl.append(n_incl)
                    
        if l_n_incl:               # check if list is not empty
            for n_incl in l_n_incl:
                l_data=[name for name in l_col_incl if str(n_incl) in name]
                l_data_pila=[name for name in l_data if "PILA" in name] 
                l_data_spalla=[name for name in l_data if "SPALLA" in name]

                if l_data_pila:
                    T = d_delta[n_centralina][T_col_name]
                    data1 = d_delta[n_centralina][l_data_pila[0]]
                    data2= d_delta[n_centralina][l_data_pila[1]]
    
                    p1 = np.poly1d(d_fit_par[n_centralina][l_data_pila[0]])
                    p2 = np.poly1d(d_fit_par[n_centralina][l_data_pila[1]])
                    xp = np.linspace(min(T), max(T), 100)
 
                    fig, ax1 = plt.subplots()
    
                    color0 = 'tab:blue'
                    color1 = 'tab:orange'
                    ax1.set_xlabel('temperature [°C]')
                    ax1.set_ylabel('rotation [°]', color=color0)
                    ax1.scatter(T, data1, s=1, color=color0, label=l_data_pila[0])
                    ax1.scatter(T, data2, s=1, color=color1, label=l_data_pila[1])
                    ax1.plot(xp,p1(xp),color=color0)
                    ax1.plot(xp,p2(xp), color=color1)
                    ax1.tick_params(axis='y')
                    #ax1.set_ylim()
                    
                    textstr1='c_x='+str(round(d_fit_par[n_centralina][l_data_pila[1]][0],4))
                    textstr2='c_y='+str(round(d_fit_par[n_centralina][l_data_pila[0]][0],4))               
                    ax1.text(0.05, 0.95,'%s\n%s'%(textstr1,textstr2),
                             transform=ax1.transAxes,fontsize=10,
                             verticalalignment='top')
   
                    plt.grid(True, which='major',axis='both', linestyle='--',dashes=[10, 10], linewidth=0.5)
                    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                               ncol=2, mode="expand", borderaxespad=0.)
                    
                    fig.canvas.set_window_title(l_data_pila[0][:-1]+"T-delta")
                    fig.tight_layout()      # otherwise the right y-label is slightly clipped
                    fig.set_size_inches(16,9)
                    fname=l_data_pila[0][:-1]+"T-delta"
                    plt.savefig(fname, dpi=300, facecolor='w', edgecolor='w',
                                orientation='landscape', papertype=None, format=None,
                                transparent=False, bbox_inches='tight', pad_inches=None,
                                frameon=None, metadata=None)
                    plt.show()
                    
                if l_data_spalla:
                    T = d_delta[n_centralina][T_col_name]
                    data1 = d_delta[n_centralina][l_data_spalla[0]]
                    data2= d_delta[n_centralina][l_data_spalla[1]]
    
                    p1 = np.poly1d(d_fit_par[n_centralina][l_data_spalla[0]])
                    p2 = np.poly1d(d_fit_par[n_centralina][l_data_spalla[1]])
                    xp = np.linspace(min(T), max(T), 100)
 
                    fig, ax1 = plt.subplots()
    
                    color0 = 'tab:blue'
                    color1 = 'tab:orange'
                    ax1.set_xlabel('temperature [°C]')
                    ax1.set_ylabel('rotation [°]', color=color0)
                    ax1.scatter(T, data1, s=1, color=color0, label=l_data_spalla[0])
                    ax1.scatter(T, data2, s=1, color=color1, label=l_data_spalla[1])
                    ax1.plot(xp,p1(xp),color=color0)
                    ax1.plot(xp,p2(xp), color=color1)
                    ax1.tick_params(axis='y')
                    #ax1.set_ylim()
                    
                    textstr1='c_x='+str(round(d_fit_par[n_centralina][l_data_spalla[1]][0],4))
                    textstr2='c_y='+str(round(d_fit_par[n_centralina][l_data_spalla[0]][0],4))                  
                    ax1.text(0.05, 0.95,'%s\n%s'%(textstr1,textstr2),
                             transform=ax1.transAxes,fontsize=10,
                             verticalalignment='top')
   
                    plt.grid(True, which='major',axis='both', linestyle='--',dashes=[10, 10], linewidth=0.5)
                    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                               ncol=2, mode="expand", borderaxespad=0.)
                    
                    fig.canvas.set_window_title(l_data_spalla[0][:-1]+" T-delta")
                    fig.tight_layout()      # otherwise the right y-label is slightly clipped
                    fig.set_size_inches(16,9)
                    fname=l_data_spalla[0][:-1]+" T-delta"
                    plt.savefig(fname, dpi=300, facecolor='w', edgecolor='w',
                                orientation='landscape', papertype=None, format=None,
                                transparent=False, bbox_inches='tight', pad_inches=None,
                                frameon=None, metadata=None)
                    plt.show()

def main():
    
    path="./download"
    l_csv=os.listdir(path)
    d_letture=ReadCSV(path,l_csv)
    d_letture_zero=ZeroLett(d_letture)
    d_delta_instr=Delta(d_letture,d_letture_zero)
    d_delta=AssignLocationName(d_delta_instr)
    s_temp=DailyAverageTemp(d_delta)
    d_delta_av=DailyAverage(d_delta)
    d_delta_av_t=DeltaMovingAverage(d_delta)
    d_fit_par=Fit_T_delta(d_delta)
    d_delta_comp=Delta_T_comp(d_delta,d_fit_par)
    d_delta_av_comp=DailyAverage(d_delta_comp)
    d_coeff_Pear=PearsonCorr(d_delta)
    #Plot_t_delta_incl(d_delta,d_delta_av,d_delta_comp)
    #Plot_t_delta_comp_incl(d_delta_comp, d_delta_av_comp)
    #Plot_t_delta_fess(d_delta,d_delta_av,d_delta_comp,d_coeff_Pear)
    #Plot_T_delta(d_delta,d_delta_av,d_fit_par)
    
# call the main function
if __name__ == "__main__":
    main()