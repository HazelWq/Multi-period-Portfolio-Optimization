import time
import numpy as np
import pandas as pd
from hmmlearn import hmm
# from datetime import datetime as dt
# from pandas_market_calendars import get_calendar
import cvxpy as cp


class GHMM:
    # Constructor: initialize the dataframe and the GHMM
    def __init__(self, df):
        # Set a seed to ensure a consistent outcome.
        np.random.seed(int(time.time()))
        # Fill NaN values by the previous valid value: we assume the index do not change
        # Calculate returns
        self.__ret = df.sort_index().ffill().pct_change().dropna() * 100
        self.__ghmm = hmm.GaussianHMM(n_components=2, n_iter=2_000, covariance_type='full')
        # Valid trading dates
        self.__tradingDates = self.__ret.index  # 暂时先使用这个，之后使用 'pandas_market_calendars' 生成将来的交易日
        # Training
        self.__train = None
        # Test
        self.__test = None
        # Regimes
        self.__n = None
        self.__c = None
        # The last date of the data frame that used to train the model
        self.__date = None
        # Expected returns
        self.__mu_n = None
        self.__mu_c = None
        # Covariances
        self.__sigma_n = None
        self.__sigma_c = None
        # Transaction Matrix
        self.__P = None
        # Stationary probabilities pi
        self.__pi = None
        # Hidden states
        self.__X = None

    # Return the earliest date which has enough number (=2000) of data for training
    # Note that: for the dates after that will have > 2000 data
    def earliestDate(self):
        return self.__ret.iloc[2000, :].name.strftime('%Y-%m-%d')

    # Set the date: time t 
    def setDate(self, str):
        date = pd.to_datetime(str)
        # Check if the given date is valid
        if date not in self.__tradingDates:
            print("The provided date is not a valid trading date; kindly choose an alternative date.")
            return None
        if date < pd.to_datetime(self.earliestDate()):
            print("Insufficient data is available for this date; please select a later date.")
            return None
        # Convert str to datetime object
        self.__date = date

    # Training the GHMM by the data before the date
    def trainingModel(self):
        # Check if the date (time t) is set
        if self.__date is None:
            print("Please set the date before initiating the model training.")
            return None
        # Training
        self.__train = self.__ret[self.__ret.index <= self.__date].tail(2000)
        # Test
        self.__test = self.__ret[self.__ret.index > self.__date]
        # Train the model
        self.__ghmm.fit(self.__train.values)
        # Determine the states that exhibit normal or constriction regimes
        if sum(self.__ghmm.means_[0]) > sum(self.__ghmm.means_[1]):
            self.__n = 0
            self.__c = 1
        else:
            self.__n = 1
            self.__c = 0
        # Expected returns
        self.__mu_n = self.__ghmm.means_[self.__n]
        self.__mu_c = self.__ghmm.means_[self.__c]
        # Covariances
        self.__sigma_n = self.__ghmm.covars_[self.__n]
        self.__sigma_c = self.__ghmm.covars_[self.__c]
        # Transaction Matrix
        self.__P = self.__ghmm.transmat_
        # Stationary probabilities pi
        self.__pi = self.__ghmm.get_stationary_distribution()
        # Hidden states
        self.__X = self.__ghmm.decode(self.__train)[1]

    # Proceed to the following date
    def updateModel(self):
        self.__ghmm = hmm.GaussianHMM(n_components=2, n_iter=2_000, covariance_type='full')
        self.__date = self.__tradingDates[np.where(self.__tradingDates == self.__date)[0][0] + 1]
        self.trainingModel()  # 这里假设数据集是给定的，现实生活中的数据集需要收集当天的数据
        # 之后更改为给定数据加入到training中

    # Return the date: time t
    def getDate(self):
        return self.__date.strftime('%Y-%m-%d')

    # Return the state of normal regimes
    def getNormState(self):
        return self.__n

    # Return the state of contraction regimes
    def getContrState(self):
        return self.__c

    # Return the expected returns in normal regimes
    def getMu_n(self):
        return self.__mu_n

    # Return the expected returns in contraction regimes
    def getMu_c(self):
        return self.__mu_c

    # Return the covariance matrices in normal regimes
    def getSigma_n(self):
        return self.__sigma_n

    # Return the covariance matrices in contraction regimes
    def getSigma_c(self):
        return self.__sigma_c

    # Return the transaction matrix
    def getTransMat(self):
        return self.__P

    # Return the stationary probabilities pi
    def getStatProb(self):
        return self.__pi

    # Return the predicted hidden states
    def getHiddenStates(self):
        return self.__X

    # Return the test set
    def getTest(self):
        return self.__test

    # Calculate the probability that the market is under normal regime at time t+1, ..., t+H
    def estimateQ(self, H):
        p_nn = self.__P[self.__n][self.__n]
        p_cc = self.__P[self.__c][self.__c]
        q = [self.__ghmm.predict_proba(self.__train.loc[self.__date].values.reshape(1, -1))[0][self.__n]]
        for _ in range(H):
            q.append(q[-1] * p_nn + (1 - q[-1]) * (1 - p_cc))
        return np.array(q)

    # Calculate the forecasting of expected return vector at time: t+1, t+2, t+3,..., t+H
    def estimateMu(self, H):
        q = self.estimateQ(H)
        mu = []
        for i in range(len(q)):
            mu.append(q[i] * self.__mu_n + (1 - q[i]) * self.__mu_c)
        return mu

    # Calculate the forecasting of covariance matrix of returns at time: t+1, t+2, t+3,..., t+H
    def estimateSigma(self, H):
        q = self.estimateQ(H)
        mu = self.estimateMu(H)
        sigma = []
        for i in range(len(q)):
            sigma.append((q[i] * self.__sigma_n
                          + (1 - q[i]) * self.__sigma_c)
                         + q[i] * np.outer((self.__mu_n - mu[i]), (self.__mu_n - mu[i]))
                         + (1 - q[i]) * np.outer((self.__mu_c - mu[i]), (self.__mu_c - mu[i])))
        return sigma


class MPC:
    def __init__(self, model):
        self.__model = model
        self.__H = None
        self.__mu = None
        self.__sigma = None
        self.__gammaRisk = 0
        self.__gammaTrade = 0

    # Set the number of periods
    # Note that, it automatically invoke the model functions to estimate expected returns and covariances
    def setH(self, H):
        self.__H = H
        self.__mu = self.__model.estimateMu(self.__H)
        self.__sigma = self.__model.estimateSigma(self.__H)

    # Set the risk-aversion parameter
    def setGammaRisk(self, gammaRisk):
        self.__gammaRisk = gammaRisk

    # Set the penalty for transactions
    def setGammaTrade(self, gammaTrade):
        self.__gammaTrade = gammaTrade

    # Return the number of periods
    def getH(self):
        return self.__H

    # Return the estimated expected returns
    def getMu_c(self):
        return self.__mu

    # Return the estimated covariance matrices
    def getSigma_n(self):
        return self.__sigma

    # Return the risk-aversion parameter
    def getGammaRisk(self):
        return self.__gammaRisk

    # Return the penalty for transactions
    def getGammaTrade(self):
        return self.__gammaTrade

    # Model Predictive Control with Mean-Variance
    def withMeanVariance(self, pi_pre):
        n_assets = pi_pre.shape[0]  # Number of assets
        pi = cp.Variable((n_assets, self.__H))  # Portfolio weights for each period
        # Objective function components
        returns = sum(self.__mu[t].T @ pi[:, t] for t in range(self.__H))
        risk = sum(cp.quad_form(pi[:, t], self.__sigma[t]) for t in range(self.__H))
        epsilon = 1e-8
        transaction_cost = sum(cp.norm1(pi[:, t] - pi[:, t - 1]) for t in range(1, self.__H))
        transaction_cost += cp.norm1(pi[:, 0] - pi_pre)  # Initial transaction cost
        # Objective function
        objective = cp.Maximize(returns - self.__gammaRisk * risk - self.__gammaTrade * transaction_cost)
        # Constraints
        constraints = [epsilon <= pi] + [pi <= 0.4] + [cp.sum(pi[:, t]) == 1 for t in range(self.__H)]
        # Solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve()
        # Check if the problem has converged
        if problem.status not in ["infeasible", "unbounded"]:
            # The problem has an optimal solution
            return pi.T.value[0]  # 我们只需要t+1的结果，对于t+2需要更新模型
        else:
            # The problem does not have an optimal solution
            print(f"Problem did not converge, status: {problem.status}")
            return None

    # MPC 如何更新
    def nextPeriod(self):
        self.__model.updateModel()
        self.setH(self.__H)


class PortfolioOptimization:
    def __init__(self, method, H, T, pi_pre):
        self.__method = method
        # The investment horizon T
        self.__T = T
        # The number of periods
        self.__H = H
        # The pre allocation
        self.__pi_pre = pi_pre
        # The allocation at the beginning of each period
        self.__allocation = [pi_pre]

    # Execute the portfolio optimization
    # Note that, 0 means MPC with Mean-Variance; 1 means MPC with Risk Parity
    def excutePortOpt(self, opt=0, gamma_risk=0.0, gamma_trade=0.0):
        self.__method.setGammaRisk(gamma_risk)
        self.__method.setGammaTrade(gamma_trade)
        self.__method.setH(self.__H)
        if opt == 0:
            for i in range(self.__T):
                self.__allocation.append(self.__method.withMeanVariance(self.__allocation[-1]))
                self.__method.nextPeriod()
        else:
            return None

    # Return the allocations from time t
    def getAllocation(self):
        return self.__allocation


def main():
    # Read data
    df = pd.read_csv("index_data.csv", index_col="Dates")
    df.index = pd.to_datetime(df.index)
    # Sort by date
    df = df.sort_index()

    model = GHMM(df)
    model.setDate(model.earliestDate())
    model.trainingModel()
    print(model.getDate())

    mpc = MPC(model)
    mpc.setH(20)
    mpc.setGammaRisk(0.1)
    mpc.setGammaTrade(0.001)
    pi = mpc.withMeanVariance(np.array([1/10] * 10))
    print(pi)

    opt = PortfolioOptimization(mpc, 10, 50, np.array([1/10] * 10))
    opt.excutePortOpt(gamma_risk=0.1, gamma_trade=0.0001)
    for i in opt.getAllocation():
        print(i)


if __name__ == "__main__":
    main()
