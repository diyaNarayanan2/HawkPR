import numpy as np
from scipy.stats import weibull_min, poisson


def HawkSim(SimTimes, n_per_batch, n_cty, n_day, DaysPred, alpha, beta, fK0, mus, covid_tr):
    '''Code to run Hawkes process simulation'''
    T_sim = n_day
    Tlow = T_sim - DaysPred
    # Simulation results
    sim = np.zeros((n_cty, T_sim, SimTimes))
    K0_sim = fK0[:, Tlow:]
    
    for itr in range(SimTimes):
        
        np.random.seed(itr)

        # Calculate base rate
        base = np.zeros((n_cty, DaysPred))
        n_exh = np.zeros((n_cty, DaysPred))

        t_stamps = np.arange(Tlow + 1, T_sim + 1)[:, None] - np.arange(1, Tlow + 1)  
        intense = (np.tile(weibull_min.pdf(t_stamps, beta, scale=alpha), (n_cty, 1, 1)) * np.tile(fK0[:, :Tlow].reshape(n_cty, 1, Tlow), (1, DaysPred, 1)) *
        np.tile(covid_tr[:, :Tlow].reshape(n_cty, 1, Tlow), (1, DaysPred, 1)))
        
        base = np.sum(intense, axis=2) + mus #mus_sim
        n_exh = np.random.poisson(base)

        for itr_cty in range(int(np.ceil(n_cty * 0.5))):
            for itr_d in range(DaysPred):
                max_d = DaysPred - itr_d

                # Sample first
                if n_exh[itr_cty, itr_d] > n_per_batch:
                    n_batch = n_exh[itr_cty, itr_d] // n_per_batch
                    cand = np.random.poisson(K0_sim[itr_cty, itr_d], size=n_per_batch)
                    n_mod = n_exh[itr_cty, itr_d] % n_per_batch
                    n_offs = np.sum(cand) * n_batch + np.sum(np.random.poisson(K0_sim[itr_cty, itr_d], size=n_mod))
                else:
                    n_offs = np.sum(np.random.poisson(K0_sim[itr_cty, itr_d], size=n_exh[itr_cty, itr_d]))

                if n_offs > n_per_batch:
                    n_batch = n_offs // n_per_batch
                    n_mod = n_offs % n_per_batch

                    sim_cand_wbl = np.ceil(weibull_min.rvs(alpha, scale=beta, size=n_per_batch))
                    sim_cand_wbl = sim_cand_wbl[sim_cand_wbl <= max_d]
                    sim_cand_wbl = np.histogram(sim_cand_wbl, bins=np.arange(1, max_d + 2))[0]

                    t_delta = np.ceil(weibull_min.rvs(alpha, scale=beta, size=n_mod))
                    t_delta = t_delta[t_delta <= max_d]
                    nt = np.histogram(t_delta, bins=np.arange(1, max_d + 2))[0] + sim_cand_wbl * n_batch
                else:
                    t_delta = np.ceil(weibull_min.rvs(alpha, scale=beta, size=n_offs))
                    t_delta = t_delta[t_delta <= max_d]
                    nt = np.histogram(t_delta, bins=np.arange(1, max_d + 2))[0]

                n_exh[itr_cty, itr_d:] = n_exh[itr_cty, itr_d:]

        sim[:, :, itr] = np.hstack([covid_tr, n_exh])
    
    sim_out = np.mean(sim, axis=2)
    return sim_out
    
