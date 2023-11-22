import time
import numpy as np
import pandas as pd
from hmmlearn import hmm
from datetime import datetime as dt

class GHMM:
    # Constructor: initialize the dataframe and the GHMM
    def __init__(self, df):
        # Set a seed to ensure a consistent outcome.
        np.random.seed(int(time.time()))
        # Fill NaN values by the previous valid value: we assume the index do not change
        # Calculate returns
        self.__ret = df.fillna(method='ffill').pct_change().dropna() * 100
        self.__ghmm = hmm.GaussianHMM(n_components=2, n_iter=2_000, covariance_type='full')
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
        # Probability of normal regime for the state at time t
        self.__q_t = None


    # Return the earliest date which has enough number (=2000) of data for training
    # Note that: for the dates after that will have > 2000 data
    def earliestDate(self):
        return self.__ret.iloc[2000, :].name.strftime('%Y-%m-%d')

    # Return the date: time t 
    def date(self):
        return self.__date.strftime('%Y-%m-%d')

    # Training the GHMM by the data before the date
    def trainingModel(self, str):
        # Convert str to datetime object
        self.__date = pd.to_datetime(str)
        # Check if the given date is valid
        if self.__date < pd.to_datetime(self.earliestDate()):
            print("Insufficient data is available for this date; please select a later date.")
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

    # Calculate the probability that the market is under normal regime at time t+1, ..., t+H
    def estimateQ(self, H):
        p_nn = self.__P[self.__n][self.__n]
        p_cc = self.__P[self.__c][self.__c]
        q = [self.__ghmm.predict_proba(self.__train.loc[self.__date].values.reshape(1, -1))[0][self.__n]]
        for _ in range(H):
            q.append(q[-1]*p_nn + (1-q[-1])*(1-p_cc))
        return np.array(q)

    # Calculate the forecasting of expected return vector at time: t+1, t+2, t+3,..., t+H
    def estimateMu(self, H):
        q = self.estimateQ(H)
        mu = []
        for i in range(len(q)):
            mu.append(q[i]*self.__mu_n + (1-q[i])*self.__mu_c)
        return mu

    # Calculate the forecasting of covariance matrix of returns at time: t+1, t+2, t+3,..., t+H
    def sigma_t(self, H):
        q = self.estimateQ(H)
        mu = self.estimateMu(H)
        sigma = []
        for i in range(len(q)):
            sigma.append((q[i]*self.__sigma_n 
                          + (1-q[i])*self.__sigma_c) 
                          + q[i]*np.outer((self.__mu_n - mu[i]), (self.__mu_n - mu[i])) 
                          + (1-q[i])*np.outer((self.__mu_c - mu[i]), (self.__mu_c - mu[i])))
        return sigma

    # Get the state of normal regimes
    def getNormState(self):
        return self.__n
    
    # Get the state of contraction regimes
    def getContrState(self):
        return self.__c

    # Get the expected returns in normal regimes
    def getMu_n(self):
        return self.__mu_n
    
    # Get the expected returns in contraction regimes
    def getMu_c(self):
        return self.__mu_c
    
    # Get the covariance matrices in normal regimes
    def getSigma_n(self):
        return self.__sigma_n
    
    # Get the covariance matrices in contraction regimes
    def getSigma_c(self):
        return self.__sigma_c
    
    # Get the transaction matrix
    def getTransMat(self):
        return self.__P
    
    # Get the stationary probabilities pi
    def getStatProb(self):
        return self.__pi

    # Get the predicted hidden states
    def getHiddenStates(self):
        return self.__X
    
    # Get the test set
    def getTest(self):
        return self.__test

