import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Hs = [1, 5, 15] #, 30]
gamma_trade = [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1]
file = [f'H{H}_trade{g_trade}' for H in Hs for g_trade in gamma_trade]

# weight figure
weights = {}
for f in range(len(file)):
    weights[f] = pd.read_csv(f'pi_{file[f]}.csv', index_col='Dates', parse_dates=['Dates'])
    weights[f][weights[f]<0] = 0
    a = weights[f].sum(axis=1)
    weights[f] = weights[f].apply(lambda x: x/sum(x), axis=1)
    #weights[f].plot()
    #plt.title(file[f])
    #plt.savefig(file[f]+'.jpg')

data = pd.read_csv('index_data.csv', index_col='Dates', parse_dates=['Dates'])
data = data.sort_index()
data = data.ffill()
ret = data.pct_change().dropna()



# value line figure
values = np.zeros((weights[0].shape[0], len(file)+1))
for f in range(len(file)+1):
    if f == len(file):
        for i,d in enumerate(weights[0].index):
            a = ret.loc[d, :].T
            if i == 0:
                values[i, f] = 1 + np.mean(a)
            else:
                values[i, f] = values[i-1,f]*(1+np.mean(a))
    else:
        for i,d in enumerate(weights[0].index):
            a = ret.loc[d, :].T
            b = weights[f].loc[d, :]
            if i == 0:
                values[i,f] = 1+a@b
            else:
                values[i,f] = values[i-1,f]*(1+a@b)
values = pd.DataFrame(values, index=weights[0].index,columns=file+['fix_mix'])
values.plot()
plt.show()

# calculate ratios
def calculate_ratios(ret, value):
    # ret(t*s): returns of strategies(s) over time period(t)
    # value(t*s): value lines of strategies(s) over time period(t)
    def maxdd(x):
        hist_max = 0
        max_dd = 0
        for i in x:
            if i>hist_max:
                hist_max = i
            else:
                if hist_max-i>max_dd:
                    max_dd = hist_max - i
        return max_dd


    cols = ['Ret', 'Vol', 'Sharpe', 'MaxDD', 'Calmar', 'Turnover', 'CumRet']
    ratios = np.zeros((len(file)+1, len(cols)))

    ratios[:, 0] = np.mean(ret,axis=0) * 252 # Ret
    ratios[:, 1] = np.std(ret, axis=0) * np.sqrt(252)# Vol
    ratios[:, 2] = ratios[:, 0]/ratios[:, 1] # Sharpe, assume rf=0?
    ratios[:, 3] = value.apply(maxdd, axis=0) # Max DD
    ratios[:, 4] = ratios[:,0]/ratios[:, 3] # Calmar
    # need weights to calculate turnover
    ratios[:, 6] = value.ioc[-1,:] - 1 # CumRet

    return pd.DataFrame(ratios, index=ret.columns, columns=cols)

RP_ratios = calculate_ratios(ret.iloc[-values.shape[0]:, :], values)
print(RP_ratios)







