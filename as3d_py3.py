#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 20:04:00 2018

@author: giacomo tancetti
"""
import pandas as pd
from datetime import datetime
import math
from scipy.optimize import fsolve
import ezdxf
    
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
    
# define dict of all measure dates for each point 
def DatesV2(df_coord,l_nomi_pti):
    d_dates={}
    for nome_pto in l_nomi_pti:
        l_dates=df_coord.loc[nome_pto]["Data Misura"].unique().tolist()
        l_dates.sort()
        d_dates[nome_pto]=l_dates
    return(d_dates)
    
# define list of all measure dates for each point
def DatesPerPoint(df_coord,l_nomi_pti):
    d_dates={}
    for point_name in l_nomi_pti:
        d_dates[point_name]=df_coord.loc[point_name]["Data Misura"].tolist()
    return(d_dates)

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

# calculate relative delta DataFrame
def DeltaCoordRel(l_dates,df_coord_mean):
    df_delta_rel=pd.DataFrame(columns=["Data Misura","N","E","H"])
    for i in range(1,len(l_dates)):
        l_deltaE=[]
        l_deltaN=[]
        l_deltaH=[]
        l_nomi_pti_misu=[]
        # find for each date the list of measured points names at time t-1
        l_nomi_pti_misu_p=df_coord_mean[df_coord_mean["Data Misura"]==l_dates[i-1]].index.tolist()
        # find for each date the list of measured points names at time t
        l_nomi_pti_misu=df_coord_mean[df_coord_mean["Data Misura"]==l_dates[i]].index.tolist()
        # find for each date the list of common points names in "l_nomi_pti_misu_p"
        # and "l_nomi_pti_misu"
        l_nomi_pti_misu_c=sorted(list(set(l_nomi_pti_misu).intersection(l_nomi_pti_misu_p)))
        
        for nome_pto in l_nomi_pti_misu_c:
            s_delta_rel_i=df_coord_mean[df_coord_mean["Data Misura"]==l_dates[i]][['E','N','H']].loc[nome_pto]-df_coord_mean[df_coord_mean["Data Misura"]==l_dates[i-1]][['E','N','H']].loc[nome_pto]
                 
            deltaE=s_delta_rel_i['E']
            deltaN=s_delta_rel_i['N']
            deltaH=s_delta_rel_i['H']
            l_deltaE.append(deltaE)
            l_deltaN.append(deltaN)
            l_deltaH.append(deltaH)
                    
        lenght=len(l_nomi_pti_misu_c)
        d_delta_rel={'Nome Punto':l_nomi_pti_misu_c,'Data Misura':[l_dates[i] for j in range(0,lenght)],'E':l_deltaE,'N':l_deltaN,'H':l_deltaH}
        df_delta_rel_i=pd.DataFrame(d_delta_rel,columns=['Data Misura',"E","N","H"],index =l_nomi_pti_misu_c)
        df_delta_rel=df_delta_rel.append(df_delta_rel_i,ignore_index=False)
    return(df_delta_rel)

def RelCoord(fs,l_dates,df_coord_mean,df_delta_rel,df_coord_zero):
    df_coord_rel=pd.DataFrame(columns=["Data Misura","N","E","H"])
    df_coord_rel=df_coord_zero    
    for i in range(1,len(l_dates)):              
        l_nomi_pti_misu=df_delta_rel[df_delta_rel["Data Misura"]==l_dates[i]].index.tolist()
        df_coord_rel_i=df_coord_rel[df_coord_rel['Data Misura']==l_dates[i-1]][['E','N','H']].loc[l_nomi_pti_misu]+fs*df_delta_rel[df_delta_rel['Data Misura']==l_dates[i]][['E','N','H']]
        lenght=len(df_coord_rel_i["E"])
        df_coord_rel_i.loc[:,'Data Misura'] = pd.Series([l_dates[i] for j in range(0,lenght)], index=l_nomi_pti_misu)
        df_coord_rel=df_coord_rel.append(df_coord_rel_i,ignore_index=False)
        
    df_coord_rel=df_coord_rel.fillna(value=0)      # soluzione tampone: modificare appena possibile
    return(df_coord_rel)
    
def RelCoordV2(fs,d_dates,df_coord_mean,l_nomi_pti,df_delta_rel,df_coord_zero):
    for i in range(1,len(l_dates)):
        l_deltaE=[]
        l_deltaN=[]
        l_deltaH=[]
        l_nomi_pti_misu=[]
        # find for each date the list of measured points names at time t-1
        l_nomi_pti_misu_p=df_delta_rel[df_delta_rel["Data Misura"]==l_dates[i-1]].index.tolist()
        # find for each date the list of measured points names at time t
        l_nomi_pti_misu=df_delta_rel[df_delta_rel["Data Misura"]==l_dates[i]].index.tolist()
        # find for each date the list of common points names in "l_nomi_pti_misu_p"
        # and "l_nomi_pti_misu"
        l_nomi_pti_misu_c=sorted(list(set(l_nomi_pti_misu).intersection(l_nomi_pti_misu_p)))
        for nome_pto in l_nomi_pti_misu_c:
            s_coord_rel_i=df_coord_rel[df_coord_rel['Data Misura']==l_dates[i-1]][['E','N','H']].loc[nome_pto]+fs*df_delta_rel[df_delta_rel['Data Misura']==l_dates[i]][['E','N','H']]
                   
    return(df_coord_rel)

# creation of displacement vector   
def ArrowPointsCreation(x0,y0,z0,x1,y1,z1, layer_name,modelspace):
    # vector magnitude calculation
    mod=math.sqrt((x1-x0)**2+(y1-y0)**2+(z1-z0)**2)
    # Calculation of a,d,c coefficient of the plane equation (base of arrow).
    # ax+by+cz+d=0
    # Coefficients are calculated by orthogonality condition between plane and
    # displacement vector.
    # If vector magnitude equal zero then a=b=c=0
    if mod==0.:
        a=b=c=0.
    else:
        a=(x1-x0)/mod  # coseno direttore x
        b=(y1-y0)/mod  # coseno direttore y
        c=(z1-z0)/mod  # coseno direttore z
    # Calculation of d coefficient of the plane equation.
    # Coefficient is calculated by imposing that point P belongs to plane 
    xP=x0+0.8*(x1-x0)
    yP=y0+0.8*(y1-y0)
    zP=z0+0.8*(z1-z0)
    d=-(a*xP+b*yP+c*zP)
    # Widht of arrow base.
    r=0.1*mod
    # Calculation of parameter of circumference equation to wich belong the 4
    # points of the base of the arrow.
    # x=xP+r*ex*cos(alpha)+r*fx*sin(alpha)
    # y=yP+r*ey*cos(alpha)+r*fy*sin(alpha)
    # z=zP+r*ez*cos(alpha)+r*fz*sin(alpha)
    if a==0.:
        ex=1.
    else:
        ex=-math.sqrt(a**2/(a**2+c**2))*(c/a)
    
    ey=0.
    
    if c==0.:
        ez=1.
    else:
        ez=-ex*a/c
    # Calculation of fx, fy, fz parameters imposing the condition of
    # orthogonality between e=(ex,ey,ez) and f=(fx,fy,fz)
    def Equations(p):
        fx, fy, fz = p
        return (ex*fx+ey*fy+ez*fz, fx*a+fy*b+fz*c, fx**2+fy**2+fz**2-1)   
    
    fx,fy,fz = fsolve(Equations, (1,1,1))
    # Alpha angles of the 4 base point of the arrow
    alpha3=0
    alpha4=math.pi/2.
    alpha5=math.pi
    alpha6=(3./2.)*math.pi
           
    # drawing of arrow points         
    P1=(x0,y0,z0)
    P2=(x1,y1,z1)
    P3=(xP+r*ex*math.cos(alpha3)+r*fx*math.sin(alpha3),yP+r*ey*math.cos(alpha3)+r*fy*math.sin(alpha3),zP+r*ez*math.cos(alpha3)+r*fz*math.sin(alpha3))
    P4=(xP+r*ex*math.cos(alpha4)+r*fx*math.sin(alpha4),yP+r*ey*math.cos(alpha4)+r*fy*math.sin(alpha4),zP+r*ez*math.cos(alpha4)+r*fz*math.sin(alpha4))
    P5=(xP+r*ex*math.cos(alpha5)+r*fx*math.sin(alpha5),yP+r*ey*math.cos(alpha5)+r*fy*math.sin(alpha5),zP+r*ez*math.cos(alpha5)+r*fz*math.sin(alpha5))
    P6=(xP+r*ex*math.cos(alpha6)+r*fx*math.sin(alpha6),yP+r*ey*math.cos(alpha6)+r*fy*math.sin(alpha6),zP+r*ez*math.cos(alpha6)+r*fz*math.sin(alpha6))
    
    modelspace.add_line(P1,P2,dxfattribs={'layer': layer_name})
    modelspace.add_line(P2,P3,dxfattribs={'layer': layer_name})
    modelspace.add_line(P2,P4,dxfattribs={'layer': layer_name})
    modelspace.add_line(P2,P5,dxfattribs={'layer': layer_name})
    modelspace.add_line(P2,P6,dxfattribs={'layer': layer_name})
    modelspace.add_line(P3,P4,dxfattribs={'layer': layer_name})
    modelspace.add_line(P4,P5,dxfattribs={'layer': layer_name})
    modelspace.add_line(P5,P6,dxfattribs={'layer': layer_name})   
    modelspace.add_line(P3,P6,dxfattribs={'layer': layer_name})

# creation of labels  
def DatesLabelsCreation(layer_name,i,text_h,drawing,modelspace):
    # insert point of dates labels
    x0=10
    y0=-20
    z0=0
    # vertical spacing of dates labels
    delta_y=text_h+(text_h/2)
    # insert point
    P_ins=(x0,y0-(delta_y*i),z0)
    modelspace.add_text(layer_name, dxfattribs={'layer': layer_name,'height': text_h}).set_pos(P_ins, align='CENTER')
  
# creation of scale bar  
def ScaleBarCreation(fs,text_h,drawing,modelspace):
    # lenght of 1 cm scalebar
    fs1=fs/100
    drawing.layers.new('scale_bar', dxfattribs={'color': 7})
    # insertion point of scale bar
    x0=10.
    y0=-20.
    z0=0.
    P0=(x0,y0,z0)
    P1=(x0,y0+(fs1/20.),z0)
    P2=((x0+fs1),y0,z0)
    P3=((x0+fs1),y0+(fs1/20.),z0)
       
    modelspace.add_line(P0,P1,dxfattribs={'layer': 'scale_bar'})   
    modelspace.add_line(P0,P2,dxfattribs={'layer': 'scale_bar'})    
    modelspace.add_line(P2,P3,dxfattribs={'layer': 'scale_bar'})    
    modelspace.add_text('0', dxfattribs={'layer': 'scale_bar','height': text_h}).set_pos(P1, align='CENTER')   
    modelspace.add_text('1 cm', dxfattribs={'layer': 'scale_bar','height': text_h}).set_pos(P3, align='BOTTOM_LEFT')

# creation of points names labels
def PointsLabels(drawing,text_h,df_coord_zero,modelspace):
    
    drawing.layers.new('points_names', dxfattribs={'color': 7})    
    for nome_pto in df_coord_zero.index:
        x=df_coord_zero['E'].loc[nome_pto]
        y=df_coord_zero['N'].loc[nome_pto]
        z=df_coord_zero['H'].loc[nome_pto]
        
        modelspace.add_text(nome_pto, dxfattribs={'layer': 'points_names','height': text_h}).set_pos((x,y,z), align='BOTTOM_LEFT')
    
          
# creation of displacements arrows
def DrawDisp(drawing,l_dates,df_coord,df_coord_zero,df_coord_rel,text_h,l_nomi_pti,modelspace):
    
    j=1 # layer color index
    
    for i in range(1,len(l_dates)):
        l_dates_i_dt= pd.to_datetime(str(l_dates[i]))
        layer_name = l_dates_i_dt.strftime('%Y%m%d')
        drawing.layers.new(layer_name, dxfattribs={'color': j})
        
        DatesLabelsCreation(layer_name,i,text_h,drawing,modelspace)

        #l_nomi_pti_misu=df_coord_rel[df_coord_rel['Data Misura']==l_dates[i]].index.tolist()
        
        # find for each date the list of measured points names at time t-1
        l_nomi_pti_misu_p=df_coord_rel[df_coord_rel["Data Misura"]==l_dates[i-1]].index.tolist()
        # find for each date the list of measured points names at time t
        l_nomi_pti_misu=df_coord_rel[df_coord_rel["Data Misura"]==l_dates[i]].index.tolist()
        
        for nome_pto in l_nomi_pti_misu:
            
            if nome_pto in l_nomi_pti_misu_p:

                x0=df_coord_rel[df_coord_rel['Data Misura']==l_dates[i-1]].loc[nome_pto]['E']
                y0=df_coord_rel[df_coord_rel['Data Misura']==l_dates[i-1]].loc[nome_pto]['N']
                z0=df_coord_rel[df_coord_rel['Data Misura']==l_dates[i-1]].loc[nome_pto]['H']
         
                x1=df_coord_rel[df_coord_rel['Data Misura']==l_dates[i]].loc[nome_pto]['E']
                y1=df_coord_rel[df_coord_rel['Data Misura']==l_dates[i]].loc[nome_pto]['N']
                z1=df_coord_rel[df_coord_rel['Data Misura']==l_dates[i]].loc[nome_pto]['H']
            
                ArrowPointsCreation(x0,y0,z0,x1,y1,z1,layer_name,modelspace)
                
        j=j+1
        
        # draw the displacement's resultant for each point        
        layer_name='risultanti'
        
        # find first and last measure date
        l_last_measure_dates=[]
        l_first_measure_dates=[]
        
        for nome_pto in l_nomi_pti:           
            l_first_measure_dates.append(df_coord.loc[nome_pto]['Data Misura'].iloc[0])
            l_last_measure_dates.append(df_coord.loc[nome_pto]['Data Misura'].iloc[-1])
           
        first_measure_dates=pd.Series(data=l_first_measure_dates,index=l_nomi_pti)
        last_measure_dates=pd.Series(data=l_last_measure_dates,index=l_nomi_pti)
        
        for nome_pto in l_nomi_pti:
                x0=df_coord_zero[df_coord_zero['Data Misura']==first_measure_dates.loc[nome_pto]].loc[nome_pto]['E']
                y0=df_coord_zero[df_coord_zero['Data Misura']==first_measure_dates.loc[nome_pto]].loc[nome_pto]['N']
                z0=df_coord_zero[df_coord_zero['Data Misura']==first_measure_dates.loc[nome_pto]].loc[nome_pto]['H']
                x1=df_coord_rel[df_coord_rel['Data Misura']==last_measure_dates.loc[nome_pto]].loc[nome_pto]['E']
                y1=df_coord_rel[df_coord_rel['Data Misura']==last_measure_dates.loc[nome_pto]].loc[nome_pto]['N']
                z1=df_coord_rel[df_coord_rel['Data Misura']==last_measure_dates.loc[nome_pto]].loc[nome_pto]['H']
                ArrowPointsCreation(x0,y0,z0,x1,y1,z1,layer_name,modelspace)
 
def as3d(df_coord):
    # height of text labels
    text_h=0.8
    # scale factor for dxf rappresentation
    fs = 300
    l_nomi_pti=NomiPti(df_coord)
    l_dates=Dates(df_coord)
    s_dates_zero=DatesZero(df_coord,l_nomi_pti)
    df_coord_zero=ZeroCoord(df_coord,l_nomi_pti,s_dates_zero)
    [df_delta,df_coord_mean]=DeltaCoord(df_coord,df_coord_zero,l_dates)
    df_delta_rel=DeltaCoordRel(l_dates,df_coord_mean)
    df_coord_rel=RelCoord(fs,l_dates,df_coord_mean,df_delta_rel,df_coord_zero)

    # creation of dxf file of all dispacements
    drawing = ezdxf.new(dxfversion='R2010')
    modelspace = drawing.modelspace()
    PointsLabels(drawing,text_h,df_coord_zero,modelspace)
    ScaleBarCreation(fs,text_h,drawing,modelspace)
    DrawDisp(drawing,l_dates,df_coord,df_coord_zero,df_coord_rel,text_h,l_nomi_pti,modelspace)
    drawing.saveas('cerrano.dxf')