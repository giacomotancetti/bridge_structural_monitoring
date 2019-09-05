# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 16:37:01 2019

@author: tancetti
"""

import os
import pandas as pd
import math
import matplotlib.pyplot as plt
import datetime
from as3d_py3 import as3d

# lettura dati topografici
def ReadXLSMan(filename,path_topo):
    xl = pd.ExcelFile(path_topo+'/'+filename)
    l_sheet_names=xl.sheet_names
    l_sheet_names_un=["installazione", "report","INSTALLAZIONE","Foglio1","VV","scheda-installazione","Elaborazione","CALCOLO"]
    l_sheet_names_points=[]
    df_excel=pd.DataFrame()

    for sheet_name in l_sheet_names:
        splitted=sheet_name.split()
        if splitted[0] not in l_sheet_names_un:
            l_sheet_names_points.append(sheet_name)

    for sheet_name in l_sheet_names_points:
        df_excel_i = pd.read_excel(path_topo+'/'+filename, sheet_name, usecols = 'K:O')
        df_excel_i=df_excel_i.dropna()
        df_excel_i['PUNTI'] = df_excel_i['PUNTI'].apply(int).apply(str)
        df_excel=df_excel.append(df_excel_i)
        
    df_excel=df_excel.rename(columns={'PUNTI': 'Nome Punto', 'GIORNI': 'Data Misura','NORD':'N','EST':'E','QUOTA':'H'})
    df_excel=df_excel.set_index('Nome Punto')
    
    return(df_excel)

# creazione variabile tipo DataFrame coordinate punti di misura
def CreateDataFrameCoord(path_topo):
    l_files=os.listdir(path_topo)
    df_coord=pd.DataFrame(columns=["Data Misura","N","E","H"])
    for filename in l_files:
        df_excel=ReadXLSMan(filename,path_topo)
        df_coord=df_coord.append(df_excel)

    l_date=[]
    for elem in df_coord['Data Misura']:
        l_date.append(elem.date())
    df_coord['Data Misura']=l_date
    
    return(df_coord)

# define list of all points names measured
def NomiPti(df_coord):
    l_nomi_pti=df_coord.index.unique().tolist()
    l_nomi_pti.sort()
    return(l_nomi_pti)

# define list of all measure dates
def Dates(df_coord):
    l_dates=df_coord["Data Misura"].unique().tolist()
    l_dates.sort()
    return(l_dates)

# define Series of zero-measure dates of all points
def DatesZero(df_coord,l_nomi_pti):
    l_dates_zero=[]
    for nome_pto in l_nomi_pti:
        date_misu_i=df_coord.loc[nome_pto]["Data Misura"]
        if isinstance(date_misu_i, pd.Series):
            l_dates_zero.append((sorted(date_misu_i.tolist()))[0])
        else:
            l_dates_zero.append(date_misu_i)
    s_dates_zero=pd.Series(l_dates_zero,index=l_nomi_pti)
    return(s_dates_zero)

# calculate zero-coordinates DataFrame
def ZeroCoord(df_coord,l_nomi_pti,s_dates_zero):
    l_E=[]
    l_N=[]
    l_H=[]
    for nome_pto in l_nomi_pti:
        data_zero=s_dates_zero.loc[nome_pto]
        coord_i=df_coord[df_coord["Data Misura"]==data_zero].loc[nome_pto]
        # calculation of average coordinates for each point
        E=coord_i['E'].mean()
        N=coord_i['N'].mean()
        H=coord_i['H'].mean()
        l_E.append(E)
        l_N.append(N)
        l_H.append(H)
       
    d_coord_zero={'Nome Punto':l_nomi_pti,'Data Misura':s_dates_zero.values.tolist(),'E':l_E,'N':l_N,'H':l_H}
    df_coord_zero=pd.DataFrame(d_coord_zero,columns=['Data Misura',"E","N","H"],index =l_nomi_pti)   
    return(df_coord_zero)    

# calculate delta DataFrame
def DeltaCoord(df_coord,df_coord_zero,l_dates):
    df_coord_mean=pd.DataFrame(columns=["Data Misura","N","E","H"])
    df_delta=pd.DataFrame(columns=["Data Misura","N","E","H"])
    for date in l_dates:
        l_E=[]
        l_N=[]
        l_H=[]
        l_nomi_pti_misu=[]
        # find for each date the list of measured points names
        l_nomi_pti_misu=df_coord[df_coord["Data Misura"]==date].index.unique().tolist()

        for nome_pto in l_nomi_pti_misu:
            coord_i=df_coord[df_coord["Data Misura"]==date].loc[nome_pto]
            E=coord_i['E'].mean()
            N=coord_i['N'].mean()
            H=coord_i['H'].mean()
            l_E.append(E)
            l_N.append(N)
            l_H.append(H)

        lenght=len(l_nomi_pti_misu)
        d_coord_mean={'Nome Punto':l_nomi_pti_misu,'Data Misura':[date for i in range(0,lenght)],'E':l_E,'N':l_N,'H':l_H}
        df_coord_mean_i=pd.DataFrame(d_coord_mean,columns=['Data Misura',"E","N","H"],index =l_nomi_pti_misu)
        df_coord_mean=df_coord_mean.append(df_coord_mean_i,ignore_index=False)

    for date in l_dates:
        l_nomi_pti_misu=df_coord[df_coord["Data Misura"]==date].index.unique().tolist()
        df_delta_i=df_coord_mean[df_coord_mean['Data Misura']==date][['E','N','H']].loc[l_nomi_pti_misu]-df_coord_zero[['E','N','H']].loc[l_nomi_pti_misu]
        # add date column to df DataFrame
        lenght=len(df_delta_i["E"])
        l_dates_i=[date for i in range(0,lenght)]
        s_date=pd.Series(l_dates_i,index=l_nomi_pti_misu)
        df_delta_i.loc[:,"Data Misura"]=s_date
                      
        df_delta=df_delta.append(df_delta_i,ignore_index=False)     
      
    return(df_delta,df_coord_mean)

# calcolo rotazioni pile da osservazioni topografia
def RotTopo(df_coord):
    alpha_deg=9.7407                # angolo fra sezione trasversale e asse E
    alpha_rad=math.radians(alpha_deg)
    
    # dizionario punti topografici di cui vengono calcolati i delta longitudinali e trasversali
    d_pile={'PILA_1':['1693106','1693107'],'PILA_2':['1693112','1693113'],'PILA_3':['1693118','1693119'],
            'PILA_4':['1693124','1693125'],'PILA_5':['1693243','1693246'],'PILA_6':['1693240','1693238'],
            'PILA_7':['1693232','1693231']}
    
    df_rot=pd.DataFrame(columns=["Pila","Data Misura","Rot Long","Rot Trasv"])
    
    for pila in d_pile.keys():
        
        pt_base=d_pile[pila][0]
        pt_pul=d_pile[pila][1]
        
        date_an_pt_base=set(df_coord["Data Misura"][pt_base])
        date_an_pt_pul=set(df_coord["Data Misura"][pt_pul])
        l_date_an=list(date_an_pt_base.intersection(date_an_pt_pul))
        l_date_an.sort()
        date_an_zero=l_date_an[3]   # settaggio data misura di zero corrispondente con l'installazione dei clinometri
        for i in range(0,len(l_date_an)):
            if l_date_an[i]==date_an_zero:
                index_zero=i
        d_rot={}
        l_alpha_trasv=[]
        l_alpha_long=[]
        l_pila=[]
        
        for i in range(index_zero,len(l_date_an)):
            delta_E_t0=df_coord[df_coord["Data Misura"]==l_date_an[index_zero]].loc[pt_pul]['E']-df_coord[df_coord["Data Misura"]==l_date_an[index_zero]].loc[pt_base]['E']
            delta_N_t0=df_coord[df_coord["Data Misura"]==l_date_an[index_zero]].loc[pt_pul]['N']-df_coord[df_coord["Data Misura"]==l_date_an[index_zero]].loc[pt_base]['N']
            delta_H_t0=df_coord[df_coord["Data Misura"]==l_date_an[index_zero]].loc[pt_pul]['H']-df_coord[df_coord["Data Misura"]==l_date_an[index_zero]].loc[pt_base]['H']

            delta_E_t1=df_coord[df_coord["Data Misura"]==l_date_an[i]].loc[pt_pul]['E']-df_coord[df_coord["Data Misura"]==l_date_an[i]].loc[pt_base]['E']
            delta_N_t1=df_coord[df_coord["Data Misura"]==l_date_an[i]].loc[pt_pul]['N']-df_coord[df_coord["Data Misura"]==l_date_an[i]].loc[pt_base]['N']
            delta_H_t1=df_coord[df_coord["Data Misura"]==l_date_an[i]].loc[pt_pul]['H']-df_coord[df_coord["Data Misura"]==l_date_an[i]].loc[pt_base]['H']    
    
            delta_E=delta_E_t1-delta_E_t0
            delta_N=delta_N_t1-delta_N_t0
            delta_H=delta_H_t1-delta_H_t0

            delta_trasv=delta_N*math.sin(alpha_rad)+delta_E*math.cos(alpha_rad) # delta_trasv + verso mare
            delta_long=delta_E*math.sin(alpha_rad)-delta_N*math.cos(alpha_rad)  # delta_long + verso sud

            alpha_trasv=math.degrees(math.atan(delta_trasv/delta_H_t1))
            alpha_long=math.degrees(math.atan(delta_long/delta_H_t1))
            
            l_alpha_trasv.append(alpha_trasv)
            l_alpha_long.append(alpha_long)
            l_pila.append(pila)
            
        d_rot={"Pila":l_pila,"Data Misura":l_date_an[index_zero:],"Rot Long":l_alpha_long,"Rot Trasv":l_alpha_trasv}
        df_rot_i=pd.DataFrame(d_rot)
        df_rot=df_rot.append(df_rot_i)
    
    df_rot=df_rot.set_index("Pila")
    
    return(df_rot)

# confronto rotazioni misurati da misure topografiche e elettrolivelle
def ConfrontoRot(d_delta_av_comp,df_rot_topo):
    l_nomi_pila=df_rot_topo.index.unique().tolist()
    for pila in l_nomi_pila:
        t=df_rot_topo["Data Misura"].loc[pila].values
        rot_trasv_topo=df_rot_topo["Rot Trasv"].loc[pila].values
        rot_long_topo=df_rot_topo["Rot Long"].loc[pila].values
        
        d_pos={}
        l_n_column=[]
        for n_centralina in d_delta_av_comp.keys(): 
            for n_column in d_delta_av_comp[n_centralina]:
                if pila in n_column:
                    l_n_column.append(n_column)
                    n_centr=n_centralina
        d_pos[n_centr]=l_n_column

        rot_trasv_incl=d_delta_av_comp[str(d_pos.keys())[-7:-3]][l_n_column[0]].reindex(t)
        rot_long_incl=d_delta_av_comp[str(d_pos.keys())[-7:-3]][l_n_column[1]].reindex(t)

        color0 = 'tab:orange'
        color1 = 'tab:blue'
        fig, ax1 = plt.subplots()
        ax1.plot(t,rot_trasv_topo,color=color0, label=pila+"_trasv_"+"topo",linewidth=0.8)
        ax1.plot(t,rot_trasv_incl, color=color1,label=pila+"_trasv_"+"incl",linewidth=0.8)
        plt.grid(True, which='major', linestyle='--',dashes=[10, 10], linewidth=0.5)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)
        ax1.set_ylim([-0.08,0.08])
        ax1.set_xlabel('time')
        ax1.set_ylabel('rotation [°]')
        fig.canvas.set_window_title(pila+"_trasv")
        fig.tight_layout()
        plt.show()
    
        color0 = 'tab:orange'
        color1 = 'tab:blue'
        fig, ax1 = plt.subplots()
        ax1.plot(t,rot_long_topo,color=color0, label=pila+"_long_"+"topo",linewidth=0.8)
        ax1.plot(t,rot_long_incl, color=color1,label=pila+"_long_"+"incl",linewidth=0.8)
        plt.grid(True, which='major', linestyle='--',dashes=[10, 10], linewidth=0.5)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)
        ax1.set_ylim([-0.08,0.08])
        ax1.set_xlabel('time')
        ax1.set_ylabel('rotation [°]')
        fig.canvas.set_window_title(pila+"_long")
        fig.tight_layout()
        plt.show()

# calcolo delta longitudinale e trasversale ad data di misura
def CalcDeltaLongTrasv(df_delta):
    alpha_deg=9.7407                # angolo fra sezione trasversale e asse E
    alpha_rad=math.radians(alpha_deg)
    
    d=[]
    for i in range(0,len(df_delta)):
        nome_pto=df_delta.iloc[i].name
        data_misu=df_delta.iloc[i]['Data Misura']
        delta_E=df_delta.iloc[i]['E']
        delta_N=df_delta.iloc[i]['N']
        delta_trasv=delta_N*math.sin(alpha_rad)+delta_E*math.cos(alpha_rad) # delta_trasv + verso mare
        delta_long=delta_E*math.sin(alpha_rad)-delta_N*math.cos(alpha_rad)  # delta_long + verso sud
        d.append({'Nome Punto':nome_pto,'Data Misura':data_misu,'Delta Long':delta_long,'Delta Trasv':delta_trasv})
    df_delta_long_trasv=pd.DataFrame(d)
    df_delta_long_trasv=df_delta_long_trasv.set_index('Nome Punto')
    
    return(df_delta_long_trasv)

# analisi spostamenti relativi fra strutture di elevazione contigue    
def AnSpostElev(df_coord,df_delta_long_trasv,df_coord_zero):
    # dizionario punti topografici di cui vengono calcolati i delta longitudinali e trasversali
    # punti alla base
    d_pile={'01_SPALLA_1':['1693102','1693203'],'02_PILA_1':['1693105','1693204'],
            '03_PILA_2':['1693111','1693210'],'04_PILA_3':['1693117','1693216'],
            '05_PILA_4':['1693123','1693222'],'06_PILA_5':['1693141','1693242'],
            '07_PILA_6':['1693137','1693239'],'08_PILA_7':['1693134','1693233'],
            '09_SPALLA_2':['1693128','1693229']}
    # punti testa
    d_pti_testa={'01_SPALLA_1':['1693102'],'02_PILA_1':['1693107'],
            '03_PILA_2':['1693113'],'04_PILA_3':['1693119'],
            '05_PILA_4':['1693125'],'06_PILA_5':['1693246'],
            '07_PILA_6':['1693238'],'08_PILA_7':['1693231'],
            '09_SPALLA_2':['1693230']}
    
    
    # calcolo distanza reciproca fra punti di riferimento per calcolo torsione
    d_dist={}
    
    for nome_pila in d_pile.keys():
        nome_pt_1=d_pile[nome_pila][0]
        nome_pt_2=d_pile[nome_pila][1]
        dist_1_2=math.sqrt((df_coord_zero.loc[nome_pt_2]['E']-df_coord_zero.loc[nome_pt_1]['E'])**2+(df_coord_zero.loc[nome_pt_2]['N']-df_coord_zero.loc[nome_pt_1]['N'])**2)
        d_dist[nome_pila]=dist_1_2
    
    d=[]
    d1=[]
    for nome_pila in d_pile.keys():
        # cerca date di misura in comune fra i punti della base di misura
        nome_pt_1=d_pile[nome_pila][0]
        nome_pt_2=d_pile[nome_pila][1]
    
        date_misu_pt_1=set(df_delta_long_trasv.loc[nome_pt_1]['Data Misura'].tolist())
        date_misu_pt_2=set(df_delta_long_trasv.loc[nome_pt_1]['Data Misura'].tolist())
        s_date_misu=date_misu_pt_1.intersection(date_misu_pt_2)
        l_date_misu=list(s_date_misu)
        l_date_misu.sort()
    
        for data_misu in l_date_misu:
            delta_long_pt_1=df_delta_long_trasv[df_delta_long_trasv['Data Misura']==data_misu].loc[nome_pt_1]['Delta Long']
            delta_long_pt_2=df_delta_long_trasv[df_delta_long_trasv['Data Misura']==data_misu].loc[nome_pt_2]['Delta Long']
            delta_long=delta_long_pt_2-delta_long_pt_1
            delta_long_medio=(delta_long_pt_1+delta_long_pt_2)/2
            alpha=math.degrees(math.atan(delta_long/d_dist[nome_pila]))
            d.append({"Nome Pila":nome_pila,"Data Misura":data_misu,"Rot Torsionale":alpha})
            d1.append({"Nome Pila":nome_pila,"Data Misura":data_misu,"Delta Long Medio":delta_long_medio})
        
    df_rot_tors=pd.DataFrame(d)
    df_rot_tors=df_rot_tors.set_index('Nome Pila')
    df_spost_long_medio=pd.DataFrame(d1)
    df_spost_long_medio=df_spost_long_medio.set_index('Nome Pila')
    
    l_pile=df_spost_long_medio.index.unique().tolist()
    l_pile.sort()
    
    d2=[]
    for i in range(1,len(l_pile)):
        nome_pila_1=l_pile[i-1]
        nome_pila_2=l_pile[i]
        date_misu_pila_1=set(df_spost_long_medio.loc[nome_pila_1]['Data Misura'].tolist())
        date_misu_pila_2=set(df_spost_long_medio.loc[nome_pila_2]['Data Misura'].tolist())
        s_date_misu_pila_1_2=date_misu_pila_1.intersection(date_misu_pila_2)
        l_date_misu_pila_1_2=list(s_date_misu_pila_1_2)
        l_date_misu_pila_1_2.sort()
        
        for data_misu in l_date_misu_pila_1_2:
            delta_long_pila_1=df_spost_long_medio[df_spost_long_medio['Data Misura']==data_misu]['Delta Long Medio'].loc[nome_pila_1]
            delta_long_pila_2=df_spost_long_medio[df_spost_long_medio['Data Misura']==data_misu]['Delta Long Medio'].loc[nome_pila_2]
            delta_long_pila_21=delta_long_pila_2-delta_long_pila_1
            d2.append({"Nome Pile":nome_pila_1[3:]+nome_pila_2[3:],"Data Misura":data_misu,"Delta Diff":delta_long_pila_21})
    
    df_spost_long_diff= pd.DataFrame(d2)
    df_spost_long_diff=df_spost_long_diff.set_index('Nome Pile')
    
    # calcolo spostamenti relativi punti testa pile
    d3=[]
    for i in range(1,len(l_pile)):
        nome_pila_1=l_pile[i-1]
        nome_pila_2=l_pile[i]
        date_misu_pila_1=set(df_delta_long_trasv.loc[d_pti_testa[nome_pila_1][0]]['Data Misura'].tolist())
        date_misu_pila_2=set(df_delta_long_trasv.loc[d_pti_testa[nome_pila_2][0]]['Data Misura'].tolist())
        s_date_misu_pila_1_2=date_misu_pila_1.intersection(date_misu_pila_2)
        l_date_misu_pila_1_2=list(s_date_misu_pila_1_2)
        l_date_misu_pila_1_2.sort()
        
        for data_misu in l_date_misu_pila_1_2:
            delta_long_pila_1=df_delta_long_trasv[df_delta_long_trasv['Data Misura']==data_misu]['Delta Long'].loc[d_pti_testa[nome_pila_1][0]]
            delta_long_pila_2=df_delta_long_trasv[df_delta_long_trasv['Data Misura']==data_misu]['Delta Long'].loc[d_pti_testa[nome_pila_2][0]]
            delta_long_pila_21=delta_long_pila_2-delta_long_pila_1
            d3.append({"Nome Pile":nome_pila_1[3:]+nome_pila_2[3:],"Data Misura":data_misu,"Delta Diff":delta_long_pila_21})

    df_spost_long_diff_testa= pd.DataFrame(d3)
    df_spost_long_diff_testa=df_spost_long_diff_testa.set_index('Nome Pile')    
    
    return(df_rot_tors,df_spost_long_diff,df_spost_long_diff_testa)
 
# rappresentazione grafica spostamenti in direzione longitudinale e trasversale       
def GraphDeltaLongTrasv(df_delta_long_trasv):
    
    # dizionario punti topografici di cui vengono graficati i delta longitudinali e trasversali
    # punti alla base
    d_pile={'PILA_1':['1693105','1693106','1693204','1693107'],'PILA_2':['1693111','1693112','1693210','1693113'],
            'PILA_3':['1693117','1693118','1693216','1693119'],'PILA_4':['1693123','1693124','1693222','1693125'],
            'PILA_5':['1693141','1693242','1693243','1693246'],'PILA_6':['1693137','1693239','1693240','1693238'],
            'PILA_7':['1693134','1693233','1693232','1693231']}
    
    l_nomi_pti=df_delta_long_trasv.index.unique().tolist()

    for nome_pila in d_pile.keys():
        
        for nome_pto in d_pile[nome_pila]:
            t=df_delta_long_trasv.loc[nome_pto]['Data Misura']
            delta_long=df_delta_long_trasv.loc[nome_pto]['Delta Long']
            delta_trasv=df_delta_long_trasv.loc[nome_pto]['Delta Trasv']
        
            color0 = 'tab:orange'
            color1 = 'tab:blue'
            fig, ax1 = plt.subplots()
            ax1.plot(t,delta_long,color=color0, label=nome_pto+"_long",linewidth=0.8)
            ax1.plot(t,delta_trasv, color=color1,label=nome_pto+"_trasv",linewidth=0.8)
            plt.grid(True, which='major', linestyle='--',dashes=[10, 10], linewidth=0.5)
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                       ncol=2, mode="expand", borderaxespad=0.)
            ax1.set_ylim([-0.05,0.05])
            ax1.set_xlabel('time')
            ax1.set_ylabel('delta [m]')
            fig.canvas.set_window_title(nome_pila+'_'+nome_pto)
            fig.tight_layout()
            fig.set_size_inches((16,9))
            fname='./grafici_spost_long_trasv/'+nome_pila+'_'+nome_pto+'_long_trasv'
            plt.savefig(fname, dpi=300, facecolor='w', edgecolor='w',
                    orientation='landscape', papertype=None, format=None,
                    transparent=False, bbox_inches='tight', pad_inches=None,
                    frameon=None, metadata=None)
            plt.show()

# grafico rotazione torsionale strutture in elevazione                 
def GraphRotTors(df_rot_tors):
    
    l_nomi_pile=df_rot_tors.index.unique().tolist()

    for nome_pila in l_nomi_pile:
        
        t=df_rot_tors.loc[nome_pila]['Data Misura']
        rot_tors=df_rot_tors.loc[nome_pila]['Rot Torsionale']
        
        color1 = 'tab:blue'
        fig, ax1 = plt.subplots()
        ax1.plot(t,rot_tors,color=color1, label=nome_pila+"_rot_tors",linewidth=0.8)
        plt.grid(True, which='major', linestyle='--',dashes=[10, 10], linewidth=0.5)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.,fontsize=10)
        ax1.set_ylim([-0.1,0.1])
        ax1.set_xlabel('time')
        ax1.set_ylabel('rotation [°]')
        fig.canvas.set_window_title(nome_pila)
        fig.tight_layout()
        fig.set_size_inches((16,9))
        fname='./grafici_rot_tors/'+nome_pila+'_rot_tors'
        plt.savefig(fname, dpi=300, facecolor='w', edgecolor='w',
                    orientation='landscape', papertype=None, format=None,
                    transparent=False, bbox_inches='tight', pad_inches=None,
                    frameon=None, metadata=None)
        plt.show()

# grafico spostamenti differenziali fra strutture di elevazione        
def GraphSpostDiff(df_spost_long_diff_base,df_spost_long_diff_testa):
    
    l_nomi_el=df_spost_long_diff_base.index.unique().tolist()

    for nome_el in l_nomi_el:
        
        t=df_spost_long_diff_base.loc[nome_el]['Data Misura']
        spost_diff_base=df_spost_long_diff_base.loc[nome_el]['Delta Diff']
        spost_diff_testa=df_spost_long_diff_testa.loc[nome_el]['Delta Diff']
        
        color1 = 'tab:blue'
        color2 = 'tab:orange'
        fig, ax1 = plt.subplots()
        ax1.plot(t,spost_diff_base,color=color1, label=nome_el+'base',linewidth=0.8)
        ax1.plot(t,spost_diff_testa,color=color2, label=nome_el+'testa',linewidth=0.8)
        plt.grid(True, which='major', linestyle='--',dashes=[10, 10], linewidth=0.5)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)
        ax1.set_ylim([-0.05,0.05])
        ax1.set_xlabel('time')
        ax1.set_ylabel('displacement [m]')
        fig.canvas.set_window_title(nome_el)
        fig.tight_layout()
        fig.set_size_inches((16,9))
        fname='./grafici_spost_diff/'+nome_el
        plt.savefig(fname, dpi=300, facecolor='w', edgecolor='w',
                    orientation='landscape', papertype=None, format=None,
                    transparent=False, bbox_inches='tight', pad_inches=None,
                    frameon=None, metadata=None)
        plt.show()

# grafico clinometri vs spostamenti testa topografici
def GraphClinSpostLong(df_delta_long_trasv,d_delta_av):
    # dizionario punti topografici testa pila
    d_pile={'PILA_1':['1693107'],'PILA_2':['1693113'],'PILA_3':['1693119'],
            'PILA_4':['1693125'],'PILA_5':['1693246'],'PILA_6':['1693238'],
            'PILA_7':['1693231']}
    
    for pila in d_pile.keys():
        for n_centralina in d_delta_av.keys():
            
            for n_column in d_delta_av[n_centralina].columns:
                if pila+'X' in n_column:
                    t_cl=d_delta_av[n_centralina][n_column].index
                    data_cl = d_delta_av[n_centralina][n_column].values
            
            df_data_topo=df_delta_long_trasv.loc[d_pile[pila][0]]
            t_topo=df_data_topo['Data Misura']
            data_topo=df_data_topo['Delta Long'].values
            
        fig, ax1 = plt.subplots()
        color0 = 'tab:blue'
        ax1.set_xlabel('time')
        ax1.set_ylabel('rotation [°]')
        ax1.plot(t_cl, data_cl, color=color0, label=pila,linewidth=0.5)
        ax1.tick_params(axis='y',labelsize=10,labelrotation=0)
        ax1.tick_params(axis='x',labelsize=10,labelrotation=-90)
        ax1.set_ylim([-0.1,0.1])
            
        plt.grid(True, which='major',axis='both', linestyle='--',dashes=[10, 10], linewidth=0.5)
        leg=plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                           ncol=2, mode="expand", borderaxespad=0.,fontsize=10)
            
        for line in leg.get_lines():
            line.set_linewidth(2)
        
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color2 = 'tab:orange'
        ax2.set_ylabel('delta_long [m]') 
        ax2.plot(t_topo, data_topo, color=color2, label="LONG DISPL", linewidth=0.5,alpha=0.4)
        ax2.tick_params(axis='y',labelsize=10,labelrotation=0)
        ax2.set_ylim([-0.1,0.1])
        
        fig.tight_layout()
        fig.set_size_inches((16,9))
        fname=pila
        fig.canvas.set_window_title(pila)

# calcolo spostamenti longitudinali dovuti alla temperatura nel periodo in cui 
# gli inclinometri non hanno fatto registrare spostamenti
def SpostLongTemp(df_delta_long_trasv,s_temp):
    # punti testa
    d_pti_testa={'PILA_1':['1693107'],
            'PILA_2':['1693113'],'PILA_3':['1693119'],
            'PILA_4':['1693125'],'PILA_5':['1693246'],
            'PILA_6':['1693238'],'PILA_7':['1693231']}
    # punti base
    d_pti_base={'PILA_1':['1693105','1693204'],'PILA_2':['1693111','1693210'],
            'PILA_3':['1693117','1693216'],'PILA_4':['1693123','1693222'],
            'PILA_5':['1693141','1693242'],'PILA_6':['1693137','1693239'],
            'PILA_7':['1693134','1693233']}
    
    # punti giunti
    d_pti_giunti={'GIUNTO_2':['1693208','1693209'],'GIUNTO_3':['1693214','1693215'],
            'GIUNTO_4':['1693220','1693221'],'GIUNTO_5':['1693226','1693227'],
            'GIUNTO_6':['1693245','1693244'],'GIUNTO_7':['1693236','1693235']}
   
    l_date_an=[datetime.date(2019, 3, 21),datetime.date(2019, 8, 21)]
    d_delta_long_temp_testa={}
    for elem in d_pti_testa.keys():
        delta=df_delta_long_trasv[df_delta_long_trasv['Data Misura']==l_date_an[1]].loc[d_pti_testa[elem]]['Delta Long']-df_delta_long_trasv[df_delta_long_trasv['Data Misura']==l_date_an[0]].loc[d_pti_testa[elem]]['Delta Long']
        delta_t=s_temp.loc[l_date_an[1]]-s_temp.loc[l_date_an[0]]
        d_delta_long_temp_testa[elem]=[delta.values[0],delta_t]
    
    d_delta_long_temp_base={}
    for elem in d_pti_base.keys():
        delta=df_delta_long_trasv[df_delta_long_trasv['Data Misura']==l_date_an[1]].loc[d_pti_base[elem]]['Delta Long'].mean()-df_delta_long_trasv[df_delta_long_trasv['Data Misura']==l_date_an[0]].loc[d_pti_base[elem]]['Delta Long'].mean()
        delta_t=s_temp.loc[l_date_an[1]]-s_temp.loc[l_date_an[0]]
        d_delta_long_temp_base[elem]=[delta,delta_t]

    d_delta_long_temp_giunti_top={}
    for elem in d_pti_giunti.keys():
        delta_t0=df_delta_long_trasv[df_delta_long_trasv['Data Misura']==l_date_an[0]].loc[d_pti_giunti[elem][1]]['Delta Long']-df_delta_long_trasv[df_delta_long_trasv['Data Misura']==l_date_an[0]].loc[d_pti_giunti[elem][0]]['Delta Long']
        delta_t1=df_delta_long_trasv[df_delta_long_trasv['Data Misura']==l_date_an[1]].loc[d_pti_giunti[elem][1]]['Delta Long']-df_delta_long_trasv[df_delta_long_trasv['Data Misura']==l_date_an[1]].loc[d_pti_giunti[elem][0]]['Delta Long']
        delta=delta_t1-delta_t0     # delta>0:allargamento giunto; delta<0 chiusura giunto
        delta_t=s_temp.loc[l_date_an[1]]-s_temp.loc[l_date_an[0]]
        d_delta_long_temp_giunti_top[elem]=[delta,delta_t]
        
    d_delta_long_temp_giunti_fess={}
    # trova nomi colonne giunti dir x
    l_col_giunti=[]
    l_col_giunti_x=[]
    for n_centralina in d_delta_av:
        for column_name in d_delta_av[n_centralina].columns:            
            if "GIUNTO" in column_name:
                l_col_giunti.append(column_name)        
    for column_name in l_col_giunti:            
        if "X" in column_name:
            l_col_giunti_x.append(column_name)
            
    for n_centralina in d_delta_av:   
        for column_name in d_delta_av[n_centralina].columns:
            if column_name in l_col_giunti_x:
               delta= d_delta_av[n_centralina][column_name].loc[l_date_an[1]]-d_delta_av[n_centralina][column_name].loc[l_date_an[0]]
               delta_t=s_temp.loc[l_date_an[1]]-s_temp.loc[l_date_an[0]]
               d_delta_long_temp_giunti_fess[column_name]=[delta,delta_t]
           
#------------------------------------------------------------------------------

def TrovaCentroRot(df_delta_long_trasv,d_delta_av_comp):    
    # dizionario punti topografici di cui vengono calcolati i delta longitudinali e trasversali
    d_pile={'PILA_1':['1693107'],'PILA_2':['1693113'],'PILA_3':['1693119'],
            'PILA_4':['1693125'],'PILA_5':['1693246'],'PILA_6':['1693238'],
            'PILA_7':['1693231']}
    
    d = []
    for pila in d_pile.keys():
        n_punto=d_pile[pila]
        delta_trasv=df_delta_long_trasv.loc[n_punto]["Delta Trasv"]
        delta_long=df_delta_long_trasv.loc[n_punto]["Delta Long"]
        data=df_delta_long_trasv.loc[n_punto]['Data Misura']

        for n_centralina in d_delta_av_comp.keys():
            for n_column in d_delta_av_comp[n_centralina].columns:
                if (pila + 'X') in n_column:
                    rot_long=d_delta_av_comp[n_centralina][n_column].loc[data]
                    h_long=delta_long/math.tan(math.radians(rot_long))
                    
                elif (pila + 'Y') in n_column:
                    rot_trasv=d_delta_av_comp[n_centralina][n_column].loc[data]
                    h_trasv=delta_trasv/math.tan(math.radians(rot_trasv))
                
        d.append({"Nome Pila": pila,"h Long":h_long.values[0],"h Trasv":h_trasv.values[0]})
    
    df_centri_rot=pd.DataFrame(d)
    
    return(df_centri_rot)
         
def calcVelSpostEl(df_delta_long_trasv):
    # dizionario punti topografici di cui vengono graficati i delta longitudinali e trasversali
    # punti alla base
    d_pile={'PILA_1':['1693105','1693204'],'PILA_2':['1693111','1693210'],
            'PILA_3':['1693117','1693216'],'PILA_4':['1693123','1693222'],
            'PILA_5':['1693141','1693242'],'PILA_6':['1693137','1693239'],
            'PILA_7':['1693134','1693233']}
    
    d=[]
    for nome_pila in d_pile.keys():
        # cerca date di misura in comune fra i punti della base di misura
        nome_pt_1=d_pile[nome_pila][0]
        nome_pt_2=d_pile[nome_pila][1]

        date_misu_pt_1=set(df_delta_long_trasv.loc[nome_pt_1]['Data Misura'].tolist())
        date_misu_pt_2=set(df_delta_long_trasv.loc[nome_pt_1]['Data Misura'].tolist())
        s_date_misu=date_misu_pt_1.intersection(date_misu_pt_2)
        l_date_misu=list(s_date_misu)
        l_date_misu.sort()
    
        for data_misu in l_date_misu:
            delta_long_pt_1=df_delta_long_trasv[df_delta_long_trasv['Data Misura']==data_misu].loc[nome_pt_1]['Delta Long']
            delta_long_pt_2=df_delta_long_trasv[df_delta_long_trasv['Data Misura']==data_misu].loc[nome_pt_2]['Delta Long']
            delta_long=delta_long_pt_2-delta_long_pt_1
            delta_long_medio=(delta_long_pt_1+delta_long_pt_2)/2
            d.append({"Nome Pila":nome_pila,"Data Misura":data_misu,"Delta Long Medio":delta_long_medio})
    
    df_spost_long_medio=pd.DataFrame(d)
    df_spost_long_medio=df_spost_long_medio.set_index('Nome Pila')
    
    vel_spost_pile={}
    for nome_pila in d_pile.keys():
        datetime_max=max(df_spost_long_medio.loc[nome_pila]['Data Misura'])
        datetime_min=min(df_spost_long_medio.loc[nome_pila]['Data Misura'])
        df_delta_long_trasv
        delta_t=datetime_max-datetime_min
        df_spost_long_medio[df_spost_long_medio['Data Misura']==datetime_max].loc[nome_pila]
        vel_spost=df_spost_long_medio[df_spost_long_medio['Data Misura']==datetime_max].loc[nome_pila]["Delta Long Medio"]*1000/delta_t.days*365
        vel_spost_pile[nome_pila]=vel_spost

def main():
    path_topo="./topografia"
    df_coord=CreateDataFrameCoord(path_topo)
    # scale factor for dxf rappresentation
    fs = 100
    l_nomi_pti=NomiPti(df_coord)
    l_dates=Dates(df_coord)
    s_dates_zero=DatesZero(df_coord,l_nomi_pti)
    df_coord_zero=ZeroCoord(df_coord,l_nomi_pti,s_dates_zero)
    [df_delta,df_coord_mean]=DeltaCoord(df_coord,df_coord_zero,l_dates)
    df_delta_long_trasv=CalcDeltaLongTrasv(df_delta)
    df_rot_tors=AnSpostElev(df_coord,df_delta_long_trasv,df_coord_zero)[0]
    df_spost_long_diff_base=AnSpostElev(df_coord,df_delta_long_trasv,df_coord_zero)[1]
    df_spost_long_diff_testa=AnSpostElev(df_coord,df_delta_long_trasv,df_coord_zero)[2]
    
    #GraphRotTors(df_rot_tors)
    #GraphSpostDiff(df_spost_long_diff_base,df_spost_long_diff_testa)
    #GraphDeltaLongTrasv(df_delta_long_trasv)
    #as3d(df_coord)

    #df_rot_topo=RotTopo(df_coord)
    #ConfrontoRot(d_delta_av_comp,df_rot_topo)
    #df_centri_rot=TrovaCentroRot(df_delta_long_trasv,d_delta_av_comp)

if __name__=="__main__":
    main()