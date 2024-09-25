import numpy as np
import scipy.sparse as sparse
from scipy.stats import weibull_min
from scipy.optimize import minimize
from plot import covid_tr_ext_i, covid_tr_ext_j


def fitwbl(covid_tr, day_for_tr, n_cty, n_day_tr, q):
    """Fit weibull distribution for the event values given
    Output Alpha and Beta"""
    # Create the observation matrix
    obs = np.tril(
        np.arange(1, day_for_tr + 1).reshape(-1, 1) - np.arange(1, day_for_tr + 1),
        -1,
    )

    # Calculate the frequency
    freq = (
        covid_tr_ext_j(covid_tr, n_day_tr)
        * covid_tr_ext_i(covid_tr, n_day_tr, n_cty)
        * q
    )
    freq = freq.toarray() if isinstance(freq, sparse.csr_matrix) else freq
    freq = np.sum(
        np.transpose(freq.reshape(n_day_tr, n_cty, n_day_tr), (0, 2, 1)), axis=2
    )
    freq = freq[:day_for_tr, :day_for_tr]

    # Find indices where both obs and freq are positive
    Ind_ret = np.where((obs > 0) & (freq > 0))
    obs = obs[Ind_ret]
    freq = freq[Ind_ret]

    # Equivalent to: [coef,~] = wblfit(obs,[],[],freq); there is no py function to fit a weibull distribution need to MLE manually

    def neg_log_likelihood(params, obs, freq):
        alpha, beta = params
        # Weibull PDF for the observed values
        pdf_vals = weibull_min.pdf(obs, c=beta, scale=alpha)
        # Log-likelihood weighted by freq
        log_likelihood = np.sum(freq * np.log(pdf_vals))
        # Return negative log-likelihood
        return -log_likelihood

    # Initial guess for alpha and beta
    initial_guess = [1.0, 1.0]
    # Minimize the negative log-likelihood
    optimum = minimize(
        neg_log_likelihood,
        initial_guess,
        args=(obs, freq),
        method="L-BFGS-B",
        bounds=[(1e-5, None), (1e-5, None)],
    )

    # Extracting the scale (alpha) and shape (beta) parameters
    alpha, beta = optimum.x

    # Now you can use these parameters as needed
    print(f"Fitted alpha (scale): {alpha}")
    print(f"Fitted beta (shape): {beta}")

    return alpha, beta
