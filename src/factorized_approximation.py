""" Uses a non-centered parameterization of the model.
    Fully factorized Gaussian + IGamma Variational distribution
	q = N(w_ijl | m_ijl, sigma^2_ijl) N(ln \tau_kl | params) IGamma(\lambda_kl| params)
	IGamma(\tau_l | params) IGamma(\lambda_l| params)
"""
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import logsumexp, gammaln, digamma
from src.utility_functions import diag_gaussian_entropy, inv_gamma_entropy, log_normal_entropy


class FactorizedHierarchicalInvGamma:
    def __init__(self, n_weights, lambda_a, lambda_b, lambda_b_global, tau_a, shapes, train_stats, classification=False,
                 n_data=None):
        self.name = "Factorized Hierarchical Inverse Gamma Variational Approximation"
        self.classification = classification
        self.n_weights = n_weights
        self.shapes = shapes
        self.num_hidden_layers = len(shapes) - 1
        self.lambda_a_prior = lambda_a
        self.lambda_b_prior = lambda_b
        self.lambda_a_prior_global = 0.5
        self.lambda_b_prior_global = lambda_b_global
        self.lambda_a_prior_oplayer = 0.5
        self.lambda_b_prior_oplayer = 1.
        self.tau_a_prior = tau_a
        self.tau_a_prior_global = 0.5
        self.tau_a_prior_oplayer = 0.5
        self.l2pi = jnp.log(2 * jnp.pi)
        self.n_data = n_data
        self.noise_entropy = None
        if not self.classification:
            # gamma(6, 6) prior on precision
            self.noise_a = 6.
            self.noise_b = 6.
            self.train_stats = train_stats

    ######### PACK UNPACK PARAMS #################################################
    def initialize_variational_params(self, param_scale=1,seed=0):
        # Initialize weights
        wlist = list()
        key = jr.PRNGKey(seed)
        for m, n in self.shapes:
            wlist.append(jr.normal(key, shape=[n * m]) * jnp.sqrt(2 / m))
            wlist.append(jnp.zeros(n))  # bias
        w = jnp.concatenate(wlist)
        log_sigma = param_scale * jr.normal(key, [w.shape[0]]) - 10.
        # initialize scale parameters
        self.tot_outputs = 0
        for _, num_hl_outputs in self.shapes:
            self.tot_outputs += num_hl_outputs
        # No hs priors on the outputs
        self.tot_outputs = self.tot_outputs - self.shapes[-1][1]
        if not self.classification:
            tau_mu, tau_log_sigma, tau_global_mu, tau_global_log_sigma, tau_oplayer_mu, tau_oplayer_log_sigma, log_a, \
                                log_b = self.initialize_scale_from_prior()
            init_params = jnp.concatenate([w.ravel(), log_sigma.ravel(),
                                          tau_mu.ravel(), tau_log_sigma.ravel(), tau_global_mu.ravel(),
                                          tau_global_log_sigma.ravel(), tau_oplayer_mu, tau_oplayer_log_sigma, log_a,
                                          log_b])
        else:
            tau_mu, tau_log_sigma, tau_global_mu, tau_global_log_sigma, tau_oplayer_mu, tau_oplayer_log_sigma = \
                self.initialize_scale_from_prior()
            init_params = jnp.concatenate([w.ravel(), log_sigma.ravel(),
                                          tau_mu.ravel(), tau_log_sigma.ravel(), tau_global_mu.ravel(),
                                          tau_global_log_sigma.ravel(), tau_oplayer_mu, tau_oplayer_log_sigma])

        return init_params

    def initialize_scale_from_prior(self, seed=0):
        key = jr.PRNGKey(seed)
        _, key2 = jr.split(key)
        # scale parameters (hidden + observed),
        self.lambda_a_hat = (self.tau_a_prior + self.lambda_a_prior) * jnp.ones([self.tot_outputs, 1]).ravel()
        self.lambda_b_hat = (1.0 / self.lambda_b_prior ** 2) * jnp.ones([self.tot_outputs, 1]).ravel()
        self.lambda_a_hat_global = (self.tau_a_prior_global + self.lambda_a_prior_global)  \
            * jnp.ones([self.num_hidden_layers, 1]).ravel()
        self.lambda_b_hat_global = (1.0 / self.lambda_b_prior_global ** 2) * jnp.ones(
            [self.num_hidden_layers, 1]).ravel()
        # set oplayer lambda param
        self.lambda_a_hat_oplayer = jnp.array(self.tau_a_prior_oplayer + self.lambda_a_prior_oplayer).reshape(-1)
        self.lambda_b_hat_oplayer = (1.0 / self.lambda_b_prior_oplayer ** 2) * jnp.ones([1]).ravel()
        # sample from half cauchy and log to initialize the mean of the log normal
        sample = jnp.abs(self.lambda_b_prior * (jr.normal(key, [self.tot_outputs]) / jr.normal(key2, [self.tot_outputs])))
        _, key = jr.split(key2)
        _, key2 = jr.split(key)
        tau_mu = jnp.log(sample)
        tau_log_sigma = jr.normal(key, [self.tot_outputs]) - 10.
        _, key = jr.split(key2)
        _, key2 = jr.split(key)
        # one tau_global for each hidden layer
        sample = jnp.abs(
            self.lambda_b_prior_global * (jr.normal(key, [self.num_hidden_layers]) / jr.normal(key2, [self.num_hidden_layers])))
        _, key = jr.split(key2)
        _, key2 = jr.split(key)
        tau_global_mu = jnp.log(sample)
        tau_global_log_sigma = jr.normal(key, [self.num_hidden_layers]) - 10.
        _, key = jr.split(key2)
        _, key2 = jr.split(key)
        # one tau for all op layer weights
        sample = jnp.abs(self.lambda_b_hat_oplayer * (jr.normal(key, ) / jr.normal(key2, )))
        _, key = jr.split(key2)
        _, key2 = jr.split(key)
        tau_oplayer_mu = jnp.log(sample)
        tau_oplayer_log_sigma = jr.normal(key, [1]) - 10.
        if not self.classification:
            log_a = jnp.array(jnp.log(self.noise_a)).reshape(-1)
            log_b = jnp.array(jnp.log(self.noise_b)).reshape(-1)
            return tau_mu, tau_log_sigma, tau_global_mu, tau_global_log_sigma, tau_oplayer_mu, tau_oplayer_log_sigma, \
                   log_a, log_b
        else:
            return tau_mu, tau_log_sigma, tau_global_mu, tau_global_log_sigma, tau_oplayer_mu, tau_oplayer_log_sigma

    def unpack_params(self, params):
        # unpack params
        w_vect = params[:self.n_weights]
        num_std = 2 * self.n_weights
        sigma = jnp.log(1 + jnp.exp(params[self.n_weights:num_std]))
        tau_mu = params[num_std:num_std + self.tot_outputs]
        tau_sigma = jnp.log(
            1 + jnp.exp(params[num_std + self.tot_outputs:num_std + 2 * self.tot_outputs]))
        tau_mu_global = params[num_std + 2 * self.tot_outputs: num_std + 2 * self.tot_outputs + self.num_hidden_layers]
        tau_sigma_global = jnp.log(1 + jnp.exp(params[num_std + 2 * self.tot_outputs + self.num_hidden_layers:num_std +
                                                                    2 * self.tot_outputs + 2 * self.num_hidden_layers]))
        tau_mu_oplayer = params[num_std + 2 * self.tot_outputs + 2 * self.num_hidden_layers: num_std +
                                                                2 * self.tot_outputs + 2 * self.num_hidden_layers + 1]
        tau_sigma_oplayer = jnp.log(
            1 + jnp.exp(params[num_std + 2 * self.tot_outputs + 2 * self.num_hidden_layers + 1:]))
        if not self.classification:
            a = tau_sigma_oplayer[1]
            b = tau_sigma_oplayer[2]
            tau_sigma_oplayer = tau_sigma_oplayer[0]
            egamma = a / b
            elog_gamma = digamma(a) - jnp.log(b)
            self.noise_entropy = inv_gamma_entropy(a, b)
            #  we will just use a point estimate of noise_var b/a+1 (noise_var ~ IGamma) for computing predictive ll
            self.noisevar = (b / (a + 1)) * self.train_stats['sigma'] ** 2
            return w_vect, sigma, tau_mu, tau_sigma, tau_mu_global, tau_sigma_global, tau_mu_oplayer, \
                   tau_sigma_oplayer, elog_gamma, egamma
        else:
            return w_vect, sigma, tau_mu, tau_sigma, tau_mu_global, tau_sigma_global, tau_mu_oplayer, tau_sigma_oplayer

    def unpack_layer_weight_variances(self, sigma_vect):
        for m, n in self.shapes:
            yield sigma_vect[:m * n].reshape((m, n)), sigma_vect[m * n:m * n + n]
            sigma_vect = sigma_vect[(m + 1) * n:]

    def unpack_layer_weight_priors(self, tau_vect):
        for m, n in self.shapes:
            yield tau_vect[:n]
            tau_vect = tau_vect[n:]

    def unpack_layer_weights(self, w_vect):
        for m, n in self.shapes:
            yield w_vect[:m * n].reshape((m, n)), w_vect[m * n:m * n + n]
            w_vect = w_vect[(m + 1) * n:]

    ######### Fixed Point Updates ################################## #####
    def fixed_point_updates(self, params):
        if self.classification:
            w_vect, sigma, tau_mu, tau_sigma, tau_mu_global, tau_sigma_global, tau_mu_oplayer, tau_sigma_oplayer = \
                self.unpack_params(params)
        else:
            w_vect, sigma, tau_mu, tau_sigma, tau_mu_global, tau_sigma_global, tau_mu_oplayer, tau_sigma_oplayer, _, _ \
                = self.unpack_params(params)
        # update lambda moments
        self.lambda_b_hat = jnp.exp(-tau_mu + 0.5 * tau_sigma ** 2) + (1. / self.lambda_b_prior ** 2)
        self.lambda_b_hat_global = jnp.exp(-tau_mu_global + 0.5 * tau_sigma_global ** 2) + (
            1. / self.lambda_b_prior_global ** 2)
        self.lambda_b_hat_oplayer = jnp.exp(-tau_mu_oplayer + 0.5 * tau_sigma_oplayer ** 2) + (
            1. / self.lambda_b_prior_oplayer ** 2)
        return None

    ######### ELBO CALC ################################################
    def lrpm_forward_pass(self, mu_vect, sigma_vect, tau_mu_vect, tau_sigma_vect, tau_mu_global, tau_sigma_global,
                          tau_mu_oplayer, tau_sigma_oplayer, inputs, seed=0):
        for layer_id, (mu, var, tau_mu, tau_sigma) in enumerate(
                zip(self.unpack_layer_weights(mu_vect), self.unpack_layer_weight_variances(sigma_vect),
                    self.unpack_layer_weight_priors(tau_mu_vect),
                    self.unpack_layer_weight_priors(tau_sigma_vect))):
            w, b = mu
            sigma__w, sigma_b = var
            key = jr.PRNGKey(seed)
            if layer_id < len(self.shapes) - 1:
                scale_mu = 0.5 * (tau_mu + tau_mu_global[layer_id])
                scale_v = 0.25 * (tau_sigma ** 2 + tau_sigma_global[layer_id] ** 2)
                scale = jnp.exp(scale_mu + jnp.sqrt(scale_v) * jr.normal(key, [tau_mu.shape[0]]))
                _, key = jr.split(key)
                mu_w = jnp.dot(inputs, w) + b
                v_w = jnp.dot(inputs ** 2, sigma__w ** 2) + sigma_b ** 2
                outputs = (jnp.sqrt(v_w) / jnp.sqrt(inputs.shape[1])) * jr.normal(key, shape=mu_w.shape) + mu_w
                _, key = jr.split(key)
                outputs = scale * outputs
                inputs = outputs * (outputs > 0)
            else:
                op_scale_mu = 0.5 * tau_mu_oplayer
                op_scale_v = 0.25 * tau_sigma_oplayer ** 2
                Ekappa_half = jnp.exp(op_scale_mu + jnp.sqrt(op_scale_v) * jr.normal(key, ))
                _, key = jr.split(key)
                mu_w = jnp.dot(inputs, w) + b
                v_w = jnp.dot(inputs ** 2, sigma__w ** 2) + sigma_b ** 2
                outputs = Ekappa_half * (jnp.sqrt(v_w) / jnp.sqrt(inputs.shape[1])) * jr.normal(
                    key, shape=mu_w.shape) + mu_w
                _, key = jr.split(key)
        return outputs

    def EPw_Gaussian(self, prior_precision, w, sigma):
        """"\int q(z) log p(z) dz, assuming gaussian q(z) and p(z)"""
        wD = w.shape[0]
        prior_wvar_ = 1. / prior_precision
        a = - 0.5 * wD * jnp.log(2 * jnp.pi) - 0.5 * wD * jnp.log(prior_wvar_) - 0.5 * prior_precision * (
            jnp.dot(w.T, w) + jnp.sum((sigma ** 2)))
        return a

    def EP_Gamma(self, Egamma, Elog_gamma):
        """ Enoise precision """
        return self.noise_a * jnp.log(self.noise_b) - gammaln(self.noise_a) - \
            (- self.noise_a - 1) * Elog_gamma - self.noise_b * Egamma

    def EPtaulambda(self, tau_mu, tau_sigma, tau_a_prior, lambda_a_prior,
                    lambda_b_prior, lambda_a_hat, lambda_b_hat):
        """ E[ln p(\tau | \lambda)] + E[ln p(\lambda)]"""
        etau_given_lambda = -gammaln(tau_a_prior) - tau_a_prior * (jnp.log(lambda_b_hat) - digamma(lambda_a_hat)) + (
                            -tau_a_prior - 1.) * tau_mu - jnp.exp(-tau_mu + 0.5 * tau_sigma ** 2) * (lambda_a_hat /
                                               lambda_b_hat)
        elambda = -gammaln(lambda_a_prior) - 2 * lambda_a_prior * jnp.log(lambda_b_prior) + (-lambda_a_prior - 1.) * (
            jnp.log(lambda_b_hat) - digamma(lambda_a_hat)) - (1. / lambda_b_prior ** 2) * (lambda_a_hat / lambda_b_hat)
        return jnp.sum(etau_given_lambda) + jnp.sum(elambda)

    def entropy(self, sigma, tau_sigma, tau_mu, tau_sigma_global, tau_mu_global, tau_sigma_oplayer, tau_mu_oplayer):
        ent_w = diag_gaussian_entropy(jnp.log(sigma), self.n_weights)
        ent_tau = log_normal_entropy(jnp.log(tau_sigma), tau_mu, self.tot_outputs) + log_normal_entropy(
            jnp.log(tau_sigma_global), tau_mu_global, self.num_hidden_layers) + log_normal_entropy(
            jnp.log(tau_sigma_oplayer), tau_mu_oplayer, 1)
        ent_lambda = inv_gamma_entropy(self.lambda_a_hat, self.lambda_b_hat) + inv_gamma_entropy(
            self.lambda_a_hat_global, self.lambda_b_hat_global) + inv_gamma_entropy(self.lambda_a_hat_oplayer,
                                                                                    self.lambda_b_hat_oplayer)
        return ent_w, ent_tau, ent_lambda

    def compute_elbo_contribs(self, params, x, y, seed=0):
        if self.classification:
            w_vect, sigma, tau_mu, tau_sigma, tau_mu_global, tau_sigma_global, tau_mu_oplayer, tau_sigma_oplayer \
                = self.unpack_params(params)
        else:
            w_vect, sigma, tau_mu, tau_sigma, tau_mu_global, tau_sigma_global, tau_mu_oplayer, tau_sigma_oplayer, \
            Elog_gamma, Egamma = self.unpack_params(params)
        preds = self.lrpm_forward_pass(w_vect, sigma, tau_mu, tau_sigma,
                                       tau_mu_global, tau_sigma_global, tau_mu_oplayer, tau_sigma_oplayer, x, seed)
        if self.classification:
            preds = preds - logsumexp(preds, axis=1, keepdims=True)
            log_lik = jnp.sum(jnp.sum(y * preds, axis=1), axis=0)
        else:
            log_lik = -0.5 * jnp.sum((preds - y.reshape(-1, 1)) ** 2) * Egamma - 0.5 * preds.shape[0] * self.l2pi \
                      + 0.5 * preds.shape[0] * Elog_gamma

        log_prior = self.EPw_Gaussian(1., w_vect, sigma)
        log_prior = log_prior + \
                    self.EPtaulambda(tau_mu, tau_sigma, self.tau_a_prior, self.lambda_a_prior, self.lambda_b_prior,
                                     self.lambda_a_hat, self.lambda_b_hat) + \
                    self.EPtaulambda(tau_mu_global, tau_sigma_global, self.tau_a_prior_global,
                                     self.lambda_a_prior_global, self.lambda_b_prior_global, self.lambda_a_hat_global,
                                     self.lambda_b_hat_global) + \
                    self.EPtaulambda(tau_mu_oplayer, tau_sigma_oplayer, self.tau_a_prior_oplayer,
                                     self.lambda_a_prior_oplayer, self.lambda_b_prior_oplayer,
                                     self.lambda_a_hat_oplayer, self.lambda_b_hat_oplayer)
        ent_w, ent_tau, ent_lambda = self.entropy(sigma, tau_sigma, tau_mu, tau_sigma_global, tau_mu_global,
                                                  tau_sigma_oplayer, tau_mu_oplayer)

        if not self.classification:
            log_prior = log_prior + self.EP_Gamma(Egamma, Elog_gamma)
            ent_lambda = ent_lambda + self.noise_entropy  # hack add it to lambda entropy
        return log_lik, log_prior, ent_w, ent_tau, ent_lambda

    def compute_train_err(self, params, X, y, seed=0):
        if self.classification:
            W_vect, sigma, tau_mu, tau_sigma, tau_mu_global, tau_sigma_global, tau_mu_oplayer, tau_sigma_oplayer = \
                self.unpack_params(params)
        else:
            W_vect, sigma, tau_mu, tau_sigma, tau_mu_global, tau_sigma_global, tau_mu_oplayer, tau_sigma_oplayer, _, _ \
                = self.unpack_params(params)
        preds = self.lrpm_forward_pass(W_vect, sigma, tau_mu, tau_sigma, tau_mu_global, tau_sigma_global,
                                       tau_mu_oplayer, tau_sigma_oplayer, X, seed)

        if self.classification:
            preds = jnp.exp(preds - logsumexp(preds, axis=1, keepdims=True))
            tru_labels = jnp.argmax(y, axis=1)
            pred_labels = jnp.argmax(preds, axis=1)
            err_ids = tru_labels != pred_labels
            return 1. * jnp.sum(err_ids) / y.shape[0]
        else:
            return jnp.sqrt(jnp.mean((preds - y.reshape(-1, 1)) ** 2))

    def compute_test_ll(self, params, x, y_test, num_samples=1, seed=0):
        if self.classification:
            W_vect, sigma, tau_mu, tau_sigma, tau_mu_global, tau_sigma_global, tau_mu_oplayer, tau_sigma_oplayer = \
                self.unpack_params(params)
        else:
            W_vect, sigma, tau_mu, tau_sigma, tau_mu_global, tau_sigma_global, tau_mu_oplayer, tau_sigma_oplayer, \
            Elog_gamma, Egamma = self.unpack_params(params)
        err_rate = 0.
        # test_ll = jnp.zeros([num_samples, y_test.shape[0]])
        test_ll = []
        test_ll_dict = dict()
        key = jr.PRNGKey(seed)
        seeds = jr.randint(key, shape=[num_samples], minval=0, maxval=1000000)
        for i in jnp.arange(num_samples):
            y = self.lrpm_forward_pass(W_vect, sigma, tau_mu, tau_sigma, tau_mu_global, tau_sigma_global,
                                       tau_mu_oplayer, tau_sigma_oplayer, x, seed=seeds[i])
            if y_test.ndim == 1:
                y = y.ravel()
            if self.classification:
                yraw = y - logsumexp(y, axis=1, keepdims=True)
                y = jnp.exp(yraw)
                tru_labels = jnp.argmax(y_test, axis=1)
                pred_labels = jnp.argmax(y, axis=1)
                err_ids = tru_labels != pred_labels
                err_rate = err_rate + jnp.sum(err_ids) / y_test.shape[0]
                # test_ll is scaled by number of test_points
                # test_ll[i] = jnp.mean(jnp.sum(y_test * jnp.log(y + 1e-32), axis=1))
                test_ll.append(jnp.mean(jnp.sum(y_test * jnp.log(y + 1e-32), axis=1)))
            else:
                # scale by target stats
                y_scaled = y * self.train_stats['sigma'] + self.train_stats['mu']
                # rmse
                err_rate = err_rate + jnp.sqrt(jnp.mean((y_test - y_scaled) ** 2))
                # test_ll[i] = (-0.5 * (1. / self.noisevar) * (y_test - y_scaled) ** 2 - 0.5 * self.l2pi - 0.5 * jnp.log(
                    # self.noisevar)).ravel()
                test_ll.append((-0.5 * (1. / self.noisevar) * (y_test - y_scaled) ** 2 - 0.5 * self.l2pi - 0.5 * jnp.log(
                    self.noisevar)).ravel())
        test_ll = jnp.asarray(test_ll)

        err_rate = err_rate / num_samples
        test_ll_dict['mu'] = jnp.mean(logsumexp(test_ll, axis=0) - jnp.log(num_samples))
        return test_ll_dict, err_rate
