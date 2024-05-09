import numpy as np
import pandas as pd
from math import *
from statistics import NormalDist

class Stock:

    def __init__(self, S_0, r, sig):
        """
        Initialize current stock parameters
        :param S_0: Stock price at time 0
        :param r: Constant risk-free rate
        :param sig: Constant volatility
        """
        self.S_0 = S_0
        self.r = r
        self.sig = sig

    def data(self):
        """
        Outputs a stock's data
        :return: String of parameters
        """
        r = self.r * 100
        vol = self.sig * 100
        print("Trading at $"f"{self.S_0} with a risk free rate of " f"{r}% and volatility " f'{vol}%')

    def price_euro(self, K, T, call, mc_methods):
        """
        Calculates price of european option for underlying asset
        :param K: Strike price
        :param T: Time to maturity
        :param call: Boolean representing a call (True) or put (False) option
        :param mc_methods: List of specified MC calculation methods ('mc' [Crude], 'av' [Antithetic],'cv' [Control])
        :return: 2D list of option MC estimates and variance
        """
        option = Price(self.S_0, K, self.r, self.sig, T, call)
        return option.Euro_MC(mc_methods)

    def price_asian(self, K, T, call, mc_methods):
        """
        Calculates price of asian option for underlying asset
        :param K: Strike price
        :param T: Time to maturity
        :param call: Boolean representing a call (True) or put (False) option
        :param mc_methods: List of specified MC calculation methods ('mc' [Crude], 'av' [Antithetic],'cv' [Control])
        :return: 2D list of option MC estimates and variance
        """
        option = Price(self.S_0, K, self.r, self.sig, T, call)
        return option.Asian_MC(mc_methods)

    def price_barrier(self, K, T, call, tau, barr, bb,  mc_methods):
        """
        Calculates price of barrier option for underlying asset
        :param K: Strike price
        :param T: Time to maturity
        :param call: Boolean representing a call (True) or put (False) option
        :param tau: Barrier price
        :param barr: Boolean calculating knock-in (1) or knock-out barrier (0)
        :param bb: Boolean sampling from Brownian Bridge (1) or GBM (0)
        :param mc_methods: List of specified MC calculation methods ('mc' [Crude], 'cond' [Conditional],'cv' [Control]
            {for bb}) ('mc', 'cv' {for gbm})
        :return: 2D list of option MC estimates and variance
        """
        option = Price(self.S_0, K, self.r, self.sig, T, call)
        return option.Barrier_MC(tau, barr, bb, mc_methods)

class Price:

    def __init__(self, S_0, K, r, sig, T, call):
        """
        Initialize current stock and option parameters
        :param S_0: Underlying price at time 0
        :param K: Strike price
        :param r: Constant risk-free rate
        :param sig: Constant volatility
        :param T: Time to maturity
        :param call: Boolean representing a call (True) or put (False) option
        """
        self.S_0 = S_0
        self.K = K
        self.r = r
        self.sig = sig
        self.T = T
        self.call = call
        # initialize additional parameters needed for MC estimation
        self.N = 260 * self.T       # number of time steps
        self.t = 0                  # default current time
        self.n = 10000              # number of simulation runs
        self.n_pilot = 100          # number of simulation runs in pilot studies (for CV estimates)
        self.seed = 100             # default seed for reproducibility
        np.random.seed(self.seed)   # set seed

    # helper functions for pricing

    def payout(self, S_t):
        """
        Compute payout for option or put given price
        :return: option payout formula
        """
        return S_t - self.K if self.call else self.K - S_t

    def knock(self, tau, barr, S):
        '''
        Checks if barrier was hit given price path
        :param tau: Barrier price
        :param barr: knock-in (1) or knock-out (0) barrier
        :param S: Underlying price path
        :return: if condition was hit or not
        '''
        if barr:  # knock-in barrier
            return np.any(S >= tau)
        else:  # knock-out barrier
            return np.any(S <= tau)

    def S_T(self, Z, t):
        """
        Computes stock price at time t given an array of standard normal variables
        :param Z: Array of standard random normal variables
        :param t: time
        :return: Underlying stock price at time t
        """
        res = self.S_0 * np.exp(((self.r - ((self.sig ** 2) / 2)) * t) + (self.sig * Z))
        return res

    def gbm_path(self, Z):
        """
        Applies GBM transformation to series of iid. std. normal rv's
        :param Z: iid. series of Z_i ~ N(0,1)
        :return: series of elements from a GBM
        """
        W = Z.cumsum() * sqrt(self.T / self.N)
        W = self.S_0 * np.exp(((self.r - ((self.sig ** 2) / 2)) * self.T) + (self.sig * W))
        return W

    def bridge(self, B_t):
        """
        Sample from a brownian bridge
        :param B_t: dataframe of iid standard normal rv's
        :return: dataframe of brownian bridge paths
        """
        B_T = B_t.iloc[:, self.N]
        for i in range(self.N - 1):
            t_ = i / self.N
            t = (i + 1) / self.N
            den = self.T - t_
            B_t.iloc[:, i + 1] = ((((self.T - t) / den) * B_t[i - 1]) +
                                 (((t - t_) / den) * B_T) +
                                 (sqrt(((t - t_)*(self.T-t)) / den) * B_t[i]))
        return B_t

    def Euro_True(self, S_t, t):
        """
        Compute the true price of a european option using Black-Scholes model
        :param S_t: Underlying price at time t
        :param t: Current time
        :return: True price
        """
        _t = self.T - t
        disc = exp(-self.r * _t)
        d1 = (log(S_t / self.K) + (self.r + (self.sig ** 2) / 2) * _t) / (self.sig * sqrt(_t))
        d2 = d1 - self.sig * sqrt(_t)
        C_t = (S_t * NormalDist().cdf(d1)) - (self.K * disc * NormalDist().cdf(d2))
        return C_t if self.call else (disc * self.K) + C_t - S_t

    # Monte Carlo Methods to estimate option prices

    def Euro_MC(self, mc_methods):
        """
        Computes european option price estimate
        :param mc_methods: List of specified MC calculation methods ('mc' [Crude], 'av' [Antithetic],'cv' [Control])
        :return: 2D list of option MC estimates and variance
        """
        ests = []   # resultant list
        # Generate standard normals
        Z = np.random.normal(size=self.n)
        # Compute discounted price at time T
        S = self.S_T(Z, self.T)
        X = exp(-self.r * self.T) * np.maximum(self.payout(S), 0)
        for meth in mc_methods:
            # Crude estimates
            if meth == 'mc':
                mc_est = np.mean(X)
                mc_var = np.var(X) / self.n
                ests.append([mc_est, mc_var])

            # Antithetic Variates
            elif meth == 'av':
                AV_Z = -Z[0: self.n // 2]
                AV_S = self.S_T(AV_Z, self.T)
                AV_X = (X[0: self.n // 2] + (exp(-self.r * self.T) * np.maximum(self.payout(AV_S), 0))) / 2
                # Compute AV realizations
                av_est = np.mean(AV_X)
                av_var = np.var(AV_X) / self.n
                ests.append([av_est, av_var])

            # Control Variables with Pilot Study
            elif meth == 'cv':
                # Compute control variates
                Z_pilot = np.random.normal(size=self.n_pilot)
                S_pilot = self.S_T(Z_pilot, self.T)
                X_pilot = exp(-self.r * self.T) * np.maximum(self.payout(S_pilot), 0)
                C_pilot = S_pilot > self.K

                mean = log(self.S_0) + (self.r - self.sig ** 2 / 2) * self.T
                std_dev = sqrt(self.sig ** 2 * self.T)
                mu_C = 1 - NormalDist(mu=mean, sigma=std_dev).cdf(log(self.K))
                # estimate beta
                beta = np.cov(X_pilot, C_pilot)[0][1] / np.var(C_pilot)
                # Large study
                C = S > self.K
                CV_X = X + beta * (mu_C - C)
                # Compute estimates
                cv_est = np.mean(CV_X)
                cv_var = np.var(CV_X) / self.n
                ests.append([cv_est, cv_var])

        return ests

    def Asian_MC(self, mc_methods):
        """
        Computes european option price estimate
        :param mc_methods: List of specified MC calculation methods ('mc' [Crude], 'av' [Antithetic],'cv' [Control])
        :return: 2D list of option MC estimates and variance
        """
        ests = []  # resultant list
        # Generate paths
        Z = pd.DataFrame(np.random.normal(size=(self.n, self.N)))   ## n x N matriz of standard normal rv's
        # Crude MC
        # apply gbm transformation
        S_t = Z.apply(self.gbm_path, axis=1)
        # compute crude MC realizations
        MC_X = exp(-self.r * self.T) * S_t.apply(lambda x: np.maximum(self.payout(x.mean()), 0), axis=1)

        for meth in mc_methods:
            # Crude MC estimates
            if meth == 'mc':
                mc_est = MC_X.mean()
                mc_var = MC_X.var() / self.n
                ests.append([mc_est, mc_var])

            # Antithetic Variates
            elif meth == 'av':
                # Compute antithetic variables and gmb transformation
                Z_AV = -Z.iloc[0:(self.n // 2), :]
                AV_S_t = Z_AV.apply(self.gbm_path, axis=1)
                # Compute AV realizations
                AV_X = (exp(-self.r * self.T)
                        * (S_t.iloc[0:(self.n // 2), :].apply(lambda x: np.maximum(self.payout(x.mean()), 0), axis=1)
                           + AV_S_t.apply(lambda x: np.maximum(self.payout(x.mean()), 0), axis=1)) / 2)
                ## Compute antithetic MC estimates
                av_est = AV_X.mean()
                av_var = AV_X.var() / self.n
                ests.append([av_est, av_var])

            # Control Variables with Pilot Study
            elif meth == 'cv':
                # Compute control variates using geometric mean
                C = S_t.apply(lambda x: np.maximum(self.payout(exp(np.log(x).mean())), 0), axis=1)
                # Compute control mean
                a = log(self.S_0) + ((self.r - ((self.sig ** 2) / 2)) * self.T * (self.N + 1) / (2 * self.N))
                b = (self.sig ** 2) * self.T * (self.N + 1) * ((2 * self.N) + 1) / (6 * (self.N ** 2))
                d1 = (-log(self.K) + a + b) / sqrt(b)
                d2 = d1 - sqrt(b)
                N_d1 = NormalDist().cdf(d1)
                N_d2 = NormalDist().cdf(d2)
                mu_C = ((exp(a + (b / 2)) * N_d1) - (self.K * N_d2))
                if not self.call: mu_C = mu_C + (exp(-self.r * self.T) * self.K) - self.S_0
                # Pilot estimate
                Z_pilot = pd.DataFrame(np.random.normal(size=(self.n_pilot, self.N)))
                S_pilot = Z_pilot.apply(self.gbm_path, axis=1)
                X_pilot = exp(-self.r * self.T) * S_pilot.apply(lambda x: np.maximum(self.payout(x.mean()), 0), axis=1)
                C_pilot = S_pilot.apply(lambda x: np.maximum(self.payout(exp(np.log(x).mean())), 0), axis=1)
                beta_hat = np.cov(X_pilot, C_pilot)[0][1] / X_pilot.var()
                # Compute CV realizations
                CV_X = MC_X + (beta_hat * (mu_C - C))
                # Compute CV estimates
                cv_est = CV_X.mean()
                cv_var = CV_X.var() / self.n
                ests.append([cv_est, cv_var])

        return ests

    def Barrier_MC(self, tau, barr, bb, mc_methods):
        """
        Computes barrier option price estimate
        :param tau: Barrier price
        :param barr: Boolean calculating knock-in (1) or knock-out barrier (0)
        :param bb: Boolean sampling from Brownian Bridge (1) or GBM (0)
        :param mc_methods: List of specified MC calculation methods ('mc' [Crude], 'cond' [Conditional],'cv' [Control]
            {for bb}) ('mc', 'cv' {for gbm})
        :return: 2D list of option MC estimates and variance
        """
        ests = []  # resultant list
        # Generate paths
        Z = pd.DataFrame(np.random.normal(size=(self.n, self.N)))  # n x N matriz of standard normal rv's

        if bb:  # sample from brownian bridge

            Z.insert(0, -1, 0)
            # Brownian Bridge transformation
            Z = self.bridge(Z)
            S_t = Z.apply(lambda x: self.S_T(x, (x.name + 1) / self.N))
            # check if barrier was hit
            I = S_t.apply(lambda x: self.knock(tau, barr, x), axis=1)
            # Compute MC realizations
            bb_X = exp(-self.r * self.T) * (np.maximum(self.payout(S_t[self.N - 1]), 0))
            bb_MC_X = bb_X * I

            for meth in mc_methods:
                # Crude MC estimates
                if meth == 'mc':
                    # Compute MC estimate
                    bb_mc_est = bb_MC_X.mean()
                    bb_mc_var = bb_MC_X.var() / self.n
                    ests.append([bb_mc_est, bb_mc_var])

                # Sample from BB conditioned on being in the money
                elif meth == 'cond':
                    # Sample from BB conditioned on being in the money
                    # Compute P(B_T > K') [P(S_T > K)]
                    K_ = (log(self.K / self.S_0) - (self.r - ((self.sig ** 2) / 2)) * self.T) / self.sig
                    p_min = 1 - NormalDist().cdf(K_ / sqrt(self.T))
                    # Sample B_T > K' (S_T > K) for call, B_T < K' (S_T < K) for put
                    bound = 1 - p_min if self.call else p_min
                    B_T = pd.DataFrame({
                        self.N - 1: pd.DataFrame(
                            np.random.uniform(size=self.n, low=bound, high=1) * sqrt(self.T)
                        ).apply(lambda x: NormalDist().inv_cdf(x.iloc[0]), axis=1)
                    })
                    B_T = pd.concat([pd.DataFrame(np.random.normal(size=(self.n, self.N - 1))), B_T], axis=1)
                    B_T.insert(0, -1, 0)
                    B_T = self.bridge(B_T)
                    # check if barrier was hit
                    S_t = B_T.apply(lambda x: self.S_T(x, (x.name + 1) / self.N))
                    I = S_t.apply(lambda x: self.knock(tau, barr, x), axis=1)
                    # Compute MC realizations
                    MC_cond = (p_min * exp(-self.r * self.T) * I *
                               S_t.apply(
                                   lambda x: x[self.N - 1] - self.K if self.call else self.K - x[self.N - 1], axis=1))
                    # Compute MC estimate
                    cond_est = MC_cond.mean()
                    cond_var = MC_cond.var() / self.n
                    ests.append([cond_est, cond_var])

                # Brownian bridge control variables with Pilot Study
                elif meth == 'cv':
                    # compute estimate from pilot study
                    bb_Z_pilot = pd.DataFrame(np.random.normal(size=(self.n_pilot, self.N)))
                    bb_Z_pilot.iloc[:, self.N - 1] = bb_Z_pilot.iloc[:, self.N - 1] * sqrt(self.T)
                    bb_B_t_pilot = bb_Z_pilot.copy()
                    bb_B_t_pilot.insert(0, -1, 0)
                    bb_B_t_pilot = self.bridge(bb_B_t_pilot)
                    bb_S_t_pilot = bb_B_t_pilot.apply(lambda x: self.S_T(x, (x.name + 1) / self.N))
                    bb_I_pilot = bb_S_t_pilot.apply(lambda x: self.knock(tau, barr, x), axis=1)
                    bb_C_pilot = (exp(-self.r * self.T) *
                                  bb_S_t_pilot.apply(lambda x: np.maximum(self.payout(x[self.N - 1]), 0), axis=1))
                    bb_X_pilot = bb_C_pilot * bb_I_pilot
                    # estimate beta
                    bb_beta_hat = np.cov(bb_X_pilot, bb_C_pilot)[0][1] / np.var(bb_C_pilot)
                    mu_C = self.Euro_True(self.S_0, self.t)
                    bb_CV_X = bb_MC_X + (bb_beta_hat * (mu_C - bb_X))
                    bb_cv_est = bb_CV_X.mean()
                    bb_cv_var = bb_CV_X.var() / self.n
                    ests.append([bb_cv_est, bb_cv_var])

            return ests

        else:   # sample from GBM
            # apply gbm transformation
            S_t = Z.apply(self.gbm_path, axis=1)
            # check if barrier was hit
            I = S_t.apply(lambda x: self.knock(tau, barr, x), axis=1)
            # compute crude realizations
            X = exp(-self.r * self.T) * np.maximum(self.payout(S_t[self.N - 1]), 0)
            MC_X = X * I

            for meth in mc_methods:
                # Crude MC estimates
                if meth == 'mc':
                    # compute crude estimate
                    mc_est = MC_X.mean()
                    mc_var = MC_X.var() / self.n
                    ests.append([mc_est, mc_var])

                # Control Variables with Pilot Study
                elif meth == 'cv':
                    # compute control variates
                    # Pilot estimate
                    Z_pilot = pd.DataFrame(np.random.normal(size=(self.n_pilot, self.N)))
                    S_pilot = Z_pilot.apply(self.gbm_path, axis=1)
                    I_pilot = S_pilot.apply(lambda x: self.knock(tau, barr, x), axis=1)
                    C_pilot = exp(-self.r * self.T) * np.maximum(self.payout(S_pilot[self.N - 1]), 0)
                    X_pilot = C_pilot * I_pilot
                    # estimate beta
                    beta_hat = np.cov(X_pilot, C_pilot)[0][1] / np.var(C_pilot)
                    # Large study
                    mu_C = self.Euro_True(self.S_0, self.t)
                    if not self.call: mu_C = mu_C + (exp(-self.r * self.T) * self.K) - self.S_0
                    # Compute CV realizations
                    CV_X = MC_X + (beta_hat * (mu_C - X))
                    # Compute CV estimates
                    cv_est = CV_X.mean()
                    cv_var = CV_X.var() / self.n
                    ests.append([cv_est, cv_var])

            return ests