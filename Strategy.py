from RiskParity import *
from GHMM2 import *
import warnings
warnings.filterwarnings('ignore')

# load and process index data
data = pd.read_csv('index_data.csv', index_col='Dates', parse_dates=['Dates'])
data = data.sort_index()
data = data.ffill()
ret = data.pct_change().dropna()

# train HMM model
hmm_model = GHMM(data)

# parameters
Hs = [1, 5, 15, 30]
gamma_trade = [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1]
for H in Hs:
    for g_trade in gamma_trade:
        pi_rp = np.zeros((data.shape[0]-2001, data.shape[1]))
        pi_0 = np.ones((data.shape[1],1))/data.shape[1]
        for d in range(2001,data.shape[0]):
            start = time.time()
            try:
                if d == 2001:
                    pi_pre = pi_0
                else:
                    pi_pre = pi_rp[d-2002,:].reshape(data.shape[1],1)

                today = data.index[d]
                hmm_model.trainingModel(today)
                Sigma = np.array(hmm_model.sigma_t(H=H))
                pi_d = Algorithm1(pi_pre,Sigma,g_trade,H)
                pi_rp[d-2001,:] = list(pi_d[:,0])
            except:
                pi_rp[d-2001,:] = pi_rp[d-2002,:]
                print(f'{data.index[d]} failed.')
                continue
            end = time.time()
            print(f"{data.index[d]} finished, time usage {end-start} s")

        print(pi_rp)
        pd.DataFrame(pi_rp, index=data.index[2001:],columns=data.columns).to_csv(f'pi_H{H}_trade{g_trade}.csv')
        print(f'pi_H{H}_trade{g_trade}.csv saved.')






print()