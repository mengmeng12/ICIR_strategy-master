# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 18:38:12 2019

@author: Jiang Yi
"""
import numpy as np
import pandas as pd
from numpy import sign
from numpy import log
from numpy import sqrt
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
# IMPORT DATA
dataIndicator = pd.read_csv('F:\\研一上\\资产定价\\研报系列\\数据\\AShareEODDerivativeIndicator.csv').drop(columns='Unnamed: 0')
dataPrice= pd.read_csv("F:\\研一上\\资产定价\\研报系列\\数据\\AShareEODPrices0927.csv").drop(columns='Unnamed: 0')
Indicator = dataIndicator[["S_INFO_WINDCODE","TRADE_DT","S_VAL_MV","S_VAL_PE_TTM","S_VAL_PE","S_VAL_PB_NEW","TOT_SHR_TODAY","OPER_REV_TTM"]]
Price= dataPrice[["S_INFO_WINDCODE","TRADE_DT","S_DQ_ADJCLOSE"]]
data = pd.merge(Indicator,Price,on=["S_INFO_WINDCODE","TRADE_DT"])

data.rename(columns = {"S_INFO_WINDCODE":"stkcd","TRADE_DT":"date"},inplace=True)
data.sort_values(["stkcd","date"],ascending = [True,True],inplace = True)
data.columns

# Construct fundemental Factors
data["net_asset"] = data["S_VAL_MV"]/data["S_VAL_PB_NEW"]

data["net_profit_TTM"] =  data["S_VAL_MV"]/data["S_VAL_PE_TTM"]
data["net_profit"] =  data["S_VAL_MV"]/data["S_VAL_PE"]

data["ROE_TTM"] = data["net_profit_TTM"]/data["TOT_SHR_TODAY"]
data["ROE"] = data["net_profit"]/data["TOT_SHR_TODAY"]

data["net_profit_TTM_change_rate"] = data.groupby("stkcd")["net_profit_TTM"].pct_change()
data["ROE_change_rate"] = data.groupby("stkcd")["ROE"].pct_change()

data["revenues_change_rate"] = data.groupby("stkcd")["OPER_REV_TTM"].pct_change()

data["ret"] = data.groupby("stkcd")["S_DQ_ADJCLOSE"].pct_change()

def rank(x):
    ra_ = pd.DataFrame(index=data.index)
    ra_['r'] = x.reset_index(drop=True).dropna().groupby(data['date']).rank(pct = True)
    return ra_.r

def delay(x, d):
    return x.groupby(data['stkcd']).shift(d).reset_index(drop=True)

data["ret(T+1)"] =data.groupby(data['stkcd'])['ret'].shift(-1)
data.head()

#Calculate IC & IR
calc_ic = data.drop(columns = ['stkcd','date']).groupby(data['date']).rank()
ic = calc_ic.groupby(data['date']).corr().loc[(slice(None),'ret(T+1)'),:]
ic.index = ic.index.droplevel(1)
ic = ic.drop(columns =[ 'ret(T+1)','ret'])
#IC
ic

ICIR = pd.DataFrame({'ic':ic.mean(),'ir':ic.mean()/ic.std()})
ICIR 
ic.loc["20160101":]
ICIR['2016ic'] = ic.loc["20160101":].mean()
ICIR['2016ir'] = ic.loc["20160101":].mean()/ic.loc["20160101":].std()
#ICIR
ICIR

#Daily Portfolio Returns
def dailylongret(x):
    tmp = data[['stkcd','date','ret(T+1)']].copy()
    tmp['factor_r'] = rank(data[x])
    return tmp.loc[tmp['factor_r']>=0.9]['ret(T+1)'].groupby(data['date']).mean()

def dailyshortret(x):
    tmp = data[['stkcd','date','ret(T+1)']].copy()
    tmp['factor_r'] = rank(data[x])
    return tmp.loc[tmp['factor_r']<=0.1]['ret(T+1)'].groupby(data['date']).mean()

def daily_fac_cost_l(x) :  
    tempp=data[['stkcd','date']].copy()
    tempp['factor_r'] = rank(data[x]).fillna(0)
    tempp['hold'] = (tempp['factor_r'] >=0.9).astype(int)  
    temppindexed = tempp.set_index(['stkcd'])  
    temppindexed['Hold_shift'] = temppindexed.groupby('stkcd')['hold'].shift(1)
    temppindexed['Hold_change'] = abs(temppindexed['hold']-temppindexed['Hold_shift'])
    turnover_rate=(temppindexed.groupby('date')['Hold_change'].sum()/temppindexed.groupby('date')['hold'].sum()).fillna(0)
    where_are_inf = np.isinf(turnover_rate)
    turnover_rate[where_are_inf] = 1
    tcost=turnover_rate*0.0015
    dict_data = {'date':tcost.index, 't-cost':tcost}
    t_cost=pd.DataFrame(dict_data).drop('date', 1)
    t_cost.columns=[x]
    return t_cost
    
dlcost= pd.DataFrame(dailylongret('ret(T+1)')).drop(columns='ret(T+1)') # get all trading days as index
    for i in ICIR.index:
        dlcost[i] = daily_fac_cost_l(i)
        
   
dlr = pd.DataFrame(dailylongret('ret(T+1)')).drop(columns='ret(T+1)') # get all trading days as index
dsr = dlr.copy()

for i in ICIR.index:
    dlr[i] = dailylongret(i)
    dsr[i] = dailyshortret(i)

# 多头收益及对应t值(扣费) 
dlcr = dlr.sub(dlcost)  # ret minus trasaction cost
dlcr['year'] = dlcr.index
dlcr['year'] = dlcr['year'].apply(lambda x:int(int(x)/10000))
dlcr['year'] = dlcr.year
dlcr['year'] = dlcr.year
dlcr
#多空收益及对应t值(不必扣费) 
dlsr = dlr.sub(dsr)
dlr['year'] = dlsr.index
dlr['year'] = dlr['year'].apply(lambda x:int(int(x)/10000))
dsr['year'] = dlr.year
dlsr['year'] = dlr.year
dlsr
#多头相对中证500收益(扣费)
# Compare Long Portfolio with ZZ500 Index, cost considered
ZZ500 = pd.read_csv('F:\\研一上\\资产定价\\研报系列\\数据\\zz500.csv')
ZZ500.sort_values(["Date"],ascending = [True],inplace = True)
ZZ500['retSX'] = (ZZ500.opn.shift(-2)/ZZ500.opn.shift(-1)-1)
ZZ500 = ZZ500[:-737].set_index('Date')
ZZ500.tail()

comparezz=dlcr.sub(ZZ500.retSX,axis=0)
comparezz
comparezz['year'] = dlr.year
reportlczz = CumRet_T(comparezz)
reportlczz

def CumRet_T(df):
    CumRet = df.groupby('year').sum()
    tvalue = sqrt(df.groupby('year').count()).mul(df.groupby('year').mean()).div(df.groupby('year').std())  
    Rep = CumRet.append(tvalue).T   
    Rep.columns=['2015','2016','t2015','t2016']
    return Rep

reportdlcr = CumRet_T(dlcr)
reportdlcr  

reportdlsr = CumRet_T(dlsr)
reportdlsr



