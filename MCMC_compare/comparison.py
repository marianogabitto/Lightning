import pystan
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats

import sys
sys.path.append('../')

from LightningF.Datasets.data import create_twofluo, data_import
from LightningF.Models.pdp_simple import TimeIndepentModelPython as ModelSpacePy


def plot_trace(param, param_name='parameter'):
    """Plot the trace and posterior of a parameter."""

    # Summary statistics
    mean = np.mean(param)
    median = np.median(param)
    cred_min, cred_max = np.percentile(param, 2.5), np.percentile(param, 97.5)

    # Plotting
    plt.subplot(2, 1, 1)
    plt.plot(param)
    plt.xlabel('samples')
    plt.ylabel(param_name)
    plt.axhline(mean, color='r', lw=2, linestyle='--')
    plt.axhline(median, color='c', lw=2, linestyle='--')
    plt.axhline(cred_min, linestyle=':', color='k', alpha=0.2)
    plt.axhline(cred_max, linestyle=':', color='k', alpha=0.2)
    plt.title('Trace and Posterior Distribution for {}'.format(param_name))

    plt.subplot(2, 1, 2)
    plt.hist(param, 30, density=True)
    if param.shape[1] > 1:
        sns.kdeplot(param[:, 1], shade=True)
        sns.kdeplot(param[:, 0], shade=True)
    else:
        sns.kdeplot(param, shade=True)
    plt.xlabel(param_name)
    plt.ylabel('density')
    # plt.axvline(mean, color='r', lw=2, linestyle='--', label='mean')
    # plt.axvline(median, color='c', lw=2, linestyle='--', label='median')
    plt.axvline(cred_min, linestyle=':', color='k', alpha=0.2, label='95% CI')
    plt.axvline(cred_max, linestyle=':', color='k', alpha=0.2)

    plt.gcf().tight_layout()
    plt.legend()


model1 = """
data {
    int<lower=0> N;
    real x[N];
    real sigma[N];
}
parameters {
    real mu;
}
model {
    mu ~ normal(-100, 100);
    x ~ normal(mu, sigma);
}
"""

model2 = """
data {
 int<lower=0> N;
 real x[N];
 real sigma[N];
}

parameters {
  vector[2] mu;
  real<lower=0, upper=1> theta;
}

model {
 mu ~ normal(0, 2);
 theta ~ beta(5, 5);
 for (n in 1:N)
   target += log_mix(theta,
                     normal_lpdf(x[n] | mu[1], sigma[n]),
                     normal_lpdf(x[n] | mu[2], sigma[n]));
}
"""

model3 = """
data {
 int<lower=0> N;
 real x[N];
 real sigma[N];
}

parameters {
  real mu[2];
  real<lower=0, upper=1> theta;
}

model {
 mu[1] ~ normal(120, 2);
 mu[2] ~ normal(0, 2);

 theta ~ beta(5, 5);
 for (n in 1:N)
   target += log_mix(theta,
                     normal_lpdf(x[n] | mu[1], sigma[n]),
                     normal_lpdf(x[n] | mu[2], sigma[n]));
}
"""

model4 = """
data {
 int<lower=0> N;
 real x[N, 2];
 real sigma[N, 2];
}

parameters {
  real mu[2, 2];
  real<lower=0, upper=1> theta;
}

model {
 row_vector[2] sgm = [1000, 1000];
 row_vector[2] pmu1 = [-10, 0];
 row_vector[2] pmu2 = [10, 0];
 mu[1] ~ normal(pmu1, sgm);
 mu[2] ~ normal(pmu2, sgm);

 theta ~ beta(5, 5);
 for (n in 1:N)
   target += log_mix(theta,
                     normal_lpdf(x[n] | mu[1], sigma[n]),
                     normal_lpdf(x[n] | mu[2], sigma[n]));
}
"""

# Parameters
sns.set()  # Nice plot aesthetic

rango = np.arange(50, 15, -10)
results = []
print("Begin Script")
for sep_ in rango:
    print("Running Distance: {}".format(sep_))

    # Data
    new = data_import(pl=0, dataset='Origami-AF647')
    data1, data2 = create_twofluo(fluos=new, dist=sep_, noise=0, pl=0, plcircles=0, seed=0)
    x = data1[:, 1:3]
    data1[:, 3:5] = data1[:, 3:5]/10
    sigma = data1[:, 3:5]**0.5
    data = {'N': len(x), 'x': x, 'sigma': sigma}

    # LIGHTNING MODEL
    spacePy = ModelSpacePy(data=data1, init_type='rl_cluster', infer_pi1=True, infer_alpha0=True, prt=0)
    spacePy.fit(iterations=200, pl=0, prt=True)

    # STAN MODEL
    sm = pystan.StanModel(model_code=model4)
    fit = sm.sampling(data=data, iter=1500, chains=1, warmup=500, thin=1, seed=101)

    # Extracting Stan Values
    mu_stan = fit['mu']
    mean_mu_stan = np.sort(np.mean(mu_stan, axis=0), axis=0)
    std_mu_stan = np.std(mu_stan, axis=0)

    # Extracting Lightning Values
    mu_l = np.sort(spacePy.Post.mu, axis=0)
    std_l = spacePy.Post.sigma2**0.5

    # Make one Nice Plot
    if sep_ == 50:
        f, ax = plt.subplots(3, figsize=(8, 8))
        f.suptitle('Inferred  Position of Fluorophores\' Centers with Different Approximate Bayesian Inference Methods',
                   fontsize=12)

        ax[0].set_aspect("equal")
        ax[0] = sns.scatterplot(data1[:, 1], data1[:, 2], ax=ax[0], color='b', marker='P')
        for n_ in np.arange(data1.shape[0]):
            ax[0].add_artist(plt.Circle((data1[n_, 1], data1[n_, 2]), np.sqrt(data1[n_, 3]), fill=False))

        ax[1].set_aspect("equal")
        ax[1] = sns.kdeplot(mu_stan[:, 0, 0], mu_stan[:, 0, 1], cmap="Reds", shade=True, thresh=0.05, ax=ax[1])
        ax[1] = sns.kdeplot(mu_stan[:, 1, 0], mu_stan[:, 1, 1], cmap="Reds", shade=True, thresh=0.05, ax=ax[1])

        ax[2].set_aspect("equal")
        x = stats.multivariate_normal(mean=mu_l[0, :], cov=[[std_l[0, 0], 0], [0, std_l[0, 1]]]).rvs(size=10000)
        ax[2] = sns.kdeplot(x[:, 0], x[:, 1], cmap="Blues", shade=True, thresh=0.05, ax=ax[2])
        x = stats.multivariate_normal(mean=mu_l[1, :], cov=[[std_l[1, 0], 0], [0, std_l[1, 1]]]).rvs(size=10000)
        ax[2] = sns.kdeplot(x[:, 0], x[:, 1], cmap="Blues", shade=True, thresh=0.05, ax=ax[2])

        xl = [np.min([ax[0].get_xlim(), ax[1].get_xlim(), ax[2].get_xlim()]),
              np.max([ax[0].get_xlim(), ax[1].get_xlim(), ax[2].get_xlim()])]
        yl = [np.min([ax[0].get_ylim(), ax[1].get_ylim(), ax[2].get_ylim()]),
              np.max([ax[0].get_ylim(), ax[1].get_ylim(), ax[2].get_ylim()])]
        for i_ in np.arange(3):
            ax[i_].set_xlim(xl)
            ax[i_].set_ylim(yl)

        ax[0].title.set_text('Observations.')
        ax[1].title.set_text('Inferred Fluorophores\' Centers [MCMC]')
        ax[2].title.set_text('Inferred Fluorophores\' Centers [VI]')
        f.tight_layout()

        pp = PdfPages("inference_50nm.pdf")
        pp.savefig(f)
        pp.close()

    temp_res = [sep_, mu_stan, mean_mu_stan, std_mu_stan, mu_l, std_l]

    results.append(temp_res)

# Postprocessing Results - Building Result Table
print("Finish Fitting")


def savagedickey(samples1, post_mean, post_std, prior1_mean=0.0, prior1_std=2.0, prior2_mean=0.0, prior2_std=2.0):
    samples2 = stats.norm.rvs(loc=post_mean, scale=post_std, size=samples1.shape[0])
    delta_theta = (np.array([samples1]).T - samples2).flatten()
    density = stats.kde.gaussian_kde(delta_theta, bw_method='scott')

    numerator = stats.norm.pdf(0, loc=prior1_mean - prior2_mean,
                               scale=np.sqrt(prior1_std ** 2 + prior2_std ** 2))
    denominator = density.evaluate(0)[0]

    return denominator / numerator

for l_ in np.arange(len(results)):
    wr = results[l_]
    sep_ = wr[0]

    # Find index of clusters
    idx1 = np.zeros(2, dtype=np.int)
    for i_ in np.arange(2):
        dtemp = np.linalg.norm(wr[2][i_, :] - wr[4][0, :])
        idx1[i_] = 0
        for j_, m in enumerate(wr[4]):
            if np.linalg.norm(wr[2][i_, :] - m) < dtemp:
                idx1[i_] = j_

    # Metrics
    ratio = np.mean([wr[3][0]**2 / wr[5][idx1[0]], wr[3][1]**2 / wr[5][idx1[1]]])

    dist1 = np.linalg.norm(wr[2][0, :] - wr[4][idx1[0], :])
    dist2 = np.linalg.norm(wr[2][1, :] - wr[4][idx1[1], :])
    dist = np.mean([dist1, dist2])

    ks_t1, ks_p1 = stats.kstest(wr[1][:, 0, 0], lambda xx: stats.norm.cdf(xx, loc=wr[4][idx1[0], 0],
                                                                          scale=wr[5][idx1[0], 0] ** 0.5))
    ks_t2, ks_p2 = stats.kstest(wr[1][:, 1, 1], lambda xx: stats.norm.cdf(xx, loc=wr[4][idx1[1], 1],
                                                                          scale=wr[5][idx1[1], 1] ** 0.5))
    ks_p = ks_p1 * ks_p2

    bf = savagedickey(samples1=wr[1][:, 0, 0:], post_mean=wr[4][idx1[0], 1], post_std=wr[5][idx1[0], 1],
                      prior1_mean=0.0, prior1_std=1000.0, prior2_mean=0.0, prior2_std=1000.0)

    print(
        "Separation [nm]:{}   Avg. Distance VI-MCMC:{:1.2f}   KS:{:1.2f}   BF:{:1.2f}   Avg Ratio MCMC/VI Std:{:1.2f}".
        format(sep_, dist, ks_p, bf, ratio))

print("Finish Script")
