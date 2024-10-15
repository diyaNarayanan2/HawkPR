# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
import torch.nn as nn
import math

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from itertools import chain, combinations
from scipy.stats import f as fdist
from scipy.stats import ttest_ind

from torch.autograd import grad

import scipy.optimize
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

import matplotlib
import matplotlib.pyplot as plt


def pretty(vector):
    vlist = vector.view(-1).tolist()
    return "[" + ", ".join("{:+.4f}".format(vi) for vi in vlist) + "]"


# For each method, training and compilation of results is done upon initilization, only solution is returned on solution fucntion call


class InvariantRiskMinimization(object):
    def __init__(self, environments, args):
        best_reg = 0
        best_err = 1e6

        x_val = environments[-1][0]
        y_val = environments[-1][1]

        for reg in [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            self.train(environments[:-1], args, reg=reg)
            err = (x_val @ self.solution() - y_val).pow(2).mean().item()

            if args["verbose"]:
                print("IRM (reg={:.3f}) has {:.3f} validation error.".format(reg, err))

            if err < best_err:
                best_err = err
                best_reg = reg
                best_phi = self.phi.clone()
        # self.phi = best_phi

    def train(self, environments, args, reg=0):
        dim_x = environments[0][0].size(1)

        # Initialize phi as a parameter with Xavier initialization
        self.phi = nn.Parameter(torch.empty(dim_x, dim_x))
        nn.init.xavier_uniform_(self.phi)

        # Initialize weights with Xavier initialization
        self.w = torch.empty(dim_x, 1)
        nn.init.xavier_uniform_(self.w)
        self.w.requires_grad = True

        opt = torch.optim.Adam([self.phi], lr=args["lr"], weight_decay=1e-5)
        loss = torch.nn.PoissonNLLLoss(log_input=True)
        # loss = torch.nn.MSELoss()
        # change this loss function and optimization function to be for poisson regression instead of linear

        for iteration in range(args["n_iterations"]):
            penalty = 0
            error = 0
            for x_e, y_e in environments:
                error_e = loss((x_e @ self.phi @ self.w), y_e)
                penalty += grad(error_e, self.w, create_graph=True)[0].pow(2).mean()
                error += error_e

            opt.zero_grad()
            # maybe optimizer should have grad
            (reg * error + (1 - reg) * penalty).backward()
            opt.step()

            if args["verbose"] and iteration % 100 == 0:
                w_str = pretty(self.solution())
                print(
                    "{:05d} | {:.5f} | {:.5f} | {:.5f} | {}".format(
                        iteration, reg, error, penalty, w_str
                    )
                )

    def solution(self):
        return (self.phi @ self.w).view(-1, 1)


class InvariantCausalPrediction(object):
    def __init__(self, environments, args):
        self.coefficients = None
        self.alpha = args["alpha"]

        x_all = []
        y_all = []
        e_all = []

        for e, (x, y) in enumerate(environments):
            x_all.append(x.numpy())
            y_all.append(y.numpy())
            e_all.append(np.full(x.shape[0], e))

        x_all = np.vstack(x_all)
        y_all = np.vstack(y_all)
        e_all = np.hstack(e_all)

        dim = x_all.shape[1]

        accepted_subsets = []
        for subset in self.powerset(range(dim)):
            if len(subset) == 0:
                continue

            x_s = x_all[:, subset]
            reg = LinearRegression(fit_intercept=False).fit(x_s, y_all)

            p_values = []
            for e in range(len(environments)):
                e_in = np.where(e_all == e)[0]
                e_out = np.where(e_all != e)[0]

                res_in = (y_all[e_in] - reg.predict(x_s[e_in, :])).ravel()
                res_out = (y_all[e_out] - reg.predict(x_s[e_out, :])).ravel()

                p_values.append(self.mean_var_test(res_in, res_out))

            # TODO: Jonas uses "min(p_values) * len(environments) - 1"
            p_value = min(p_values) * len(environments)

            if p_value > self.alpha:
                accepted_subsets.append(set(subset))
                if args["verbose"]:
                    print("Accepted subset:", subset)

        if len(accepted_subsets):
            accepted_features = list(set.intersection(*accepted_subsets))
            if args["verbose"]:
                print("Intersection:", accepted_features)
            self.coefficients = np.zeros(dim)

            if len(accepted_features):
                x_s = x_all[:, list(accepted_features)]
                reg = LinearRegression(fit_intercept=False).fit(x_s, y_all)
                self.coefficients[list(accepted_features)] = reg.coef_

            self.coefficients = torch.Tensor(self.coefficients)
        else:
            self.coefficients = torch.zeros(dim)

    def mean_var_test(self, x, y):
        pvalue_mean = ttest_ind(x, y, equal_var=False).pvalue
        pvalue_var1 = 1 - fdist.cdf(
            np.var(x, ddof=1) / np.var(y, ddof=1), x.shape[0] - 1, y.shape[0] - 1
        )

        pvalue_var2 = 2 * min(pvalue_var1, 1 - pvalue_var1)

        return 2 * min(pvalue_mean, pvalue_var2)

    def powerset(self, s):
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    def solution(self):
        return self.coefficients.view(-1, 1)


class EmpiricalRiskMinimizer(object):
    def __init__(self, environments, args):
        # x is the covariates matrix
        # y is the depedent variable
        x_all = torch.cat([x for (x, y) in environments]).numpy()
        y_all = torch.cat([y for (x, y) in environments]).numpy()
        print(np.isnan(x_all).sum())
        print(np.isinf(x_all).sum())
        print(np.isnan(y_all).sum())
        print(np.isinf(y_all).sum())
        print(np.min(y_all))
        print(np.min(x_all))

        # w = LinearRegression(fit_intercept=False).fit(x_all, y_all).coef_
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        x_all_scaled = scaler_x.fit_transform(x_all)
        y_all_scaled = scaler_y.fit_transform(y_all.reshape(-1, 1))
        model = PoissonRegressor(max_iter=100, verbose=0)
        model.fit(x_all, y_all.ravel())
        w_scaled = model.coef_
        w_original = w_scaled / scaler_x.scale_
        self.w = torch.Tensor(w_original).view(-1, 1)

    def solution(self):
        return self.w
