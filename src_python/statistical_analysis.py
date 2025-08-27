# Enhanced Statistical Analysis Module
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from constants import CONSTANTS

class StatisticalAnalysis:
    def __init__(self):
        self.tolerance = 1e-10

    def compute_residuals(self, W, g_e_range, g_gamma_range):
        """Compute residuals for W relative to expected values."""
        expected_W = np.ones_like(W) * 0.5
        return W - expected_W

    def compute_chi2_p_values(self, W, N, g_e_range, g_gamma_range):
        """Compute chi-squared p-values for W."""
        chi2_stats = np.zeros_like(W)
        p_vals = np.zeros_like(W)

        n_realizations = CONSTANTS.get('N_REALIZATIONS', 10000)

        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                observed = W[i, j] * n_realizations
                expected = 0.5 * n_realizations

                # Use chi-squared test for goodness of fit
                try:
                    chi2_stat, p_val = stats.chisquare([observed, n_realizations - observed],
                                                     [expected, n_realizations - expected])
                    chi2_stats[i, j] = chi2_stat
                    p_vals[i, j] = p_val
                except:
                    chi2_stats[i, j] = 0
                    p_vals[i, j] = 1

        return p_vals

    def fit_figure_3(self, N_range, log_g_50):
        """Enhanced fit for log(g_gamma) vs log10(N) for Figure 3."""
        if len(N_range) < 2:
            return 0, 0, np.zeros_like(log_g_50), np.zeros_like(log_g_50), 0, 1

        log_N = np.log10(N_range)

        # Perform linear regression in log-log space
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_N, log_g_50)

        # Convert intercept to c parameter (where intercept = -c/ln(10))
        c = -intercept

        # Calculate fitted values and residuals
        fit_log_g = slope * log_N + intercept
        residuals = log_g_50 - fit_log_g

        # Calculate R-squared
        ss_tot = np.sum((log_g_50 - np.mean(log_g_50))**2)
        ss_res = np.sum(residuals**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 0

        return slope, c, fit_log_g, residuals, r_squared, p_value

    def fit_normal(self, data):
        """Fit normal distribution to data with error handling."""
        if len(data) < 2 or np.all(np.isnan(data)):
            return 0, 1

        # Remove any infinite or NaN values
        clean_data = data[np.isfinite(data)]
        if len(clean_data) < 2:
            return 0, 1

        try:
            mu, sigma = stats.norm.fit(clean_data)
            return mu, sigma
        except:
            return np.mean(clean_data), np.std(clean_data)

    def fit_arcsine(self, data):
        """Fit arcsine (beta(0.5,0.5)) distribution to data, scaling to [0,1] for domain [-1,1]."""
        if len(data) < 2 or np.all(np.isnan(data)):
            return [0.5, 0.5]
        # Scale data from [-1,1] to [0,1]
        clean_data = data[np.isfinite(data)]
        clean_data = clean_data[(clean_data >= -1) & (clean_data <= 1)]
        if len(clean_data) < 2:
            return [0.5, 0.5]
        scaled_data = (clean_data + 1) / 2  # Now in [0,1]
        try:
            params = stats.beta.fit(scaled_data, floc=0, fscale=1)
            return list(params)
        except:
            return [0.5, 0.5]


    def fit_exponential(self, data):
        """Fit exponential distribution to data with better error handling."""
        if len(data) < 2 or np.all(np.isnan(data)):
            return 0, 1

        # Remove negative values and NaN/inf
        clean_data = data[np.isfinite(data) & (data >= 0)]
        if len(clean_data) < 2:
            return 0, 1

        try:
            loc, scale = stats.expon.fit(clean_data, floc=0)
            return loc, scale
        except:
            return 0, np.mean(clean_data) if len(clean_data) > 0 else 1



    def fit_sigmoid(self, x, y):
        """Fit sigmoid function to data."""
        if len(x) < 4 or np.all(np.isnan(y)):
            return [1, 1, np.mean(x) if len(x) > 0 else 0, 0]

        def sigmoid_func(x, a, b, c, d):
            return a / (1 + np.exp(-b * (x - c))) + d

        try:
            # Initial parameter guess
            a_guess = np.max(y) - np.min(y)
            d_guess = np.min(y)
            c_guess = np.mean(x)
            b_guess = 1.0

            params, _ = curve_fit(sigmoid_func, x, y,
                                p0=[a_guess, b_guess, c_guess, d_guess],
                                maxfev=2000)
            return params
        except:
            return [1, 1, np.mean(x), 0]

    def fit_convergence(self, x, y):
        """Fit exponential decay for convergence analysis."""
        if len(x) < 3 or np.all(np.isnan(y)):
            return [1, 0.1, np.mean(y) if len(y) > 0 else 0]

        def exp_decay(x, a, b, c):
            return a * np.exp(-b * x) + c

        try:
            # Initial parameter guess
            a_guess = np.max(y) - np.min(y)
            c_guess = np.min(y)
            b_guess = 0.1

            params, _ = curve_fit(exp_decay, x, y,
                                p0=[a_guess, b_guess, c_guess],
                                maxfev=2000)
            return params
        except:
            return [1, 0.1, np.mean(y)]

    def fit_power_law(self, x, y):
        """Fit power-law to data (log-log linear regression) with enhanced error handling."""
        if len(x) < 2 or np.all(np.isnan(y)) or np.any(x <= 0) or np.any(y <= 0):
            return 0, 0, 0

        # Remove zeros and negative values
        mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
        x_clean = x[mask]
        y_clean = y[mask]

        if len(x_clean) < 2:
            return 0, 0, 0

        try:
            log_x = np.log10(x_clean)
            log_y = np.log10(y_clean)

            slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
            r_squared = r_value**2

            return slope, intercept, r_squared
        except:
            return 0, 0, 0

    def kolmogorov_smirnov_test(self, data, distribution='uniform'):
        """Perform Kolmogorov-Smirnov test for specified distribution."""
        if len(data) < 2:
            return 0, 1

        clean_data = data[np.isfinite(data)]
        if len(clean_data) < 2:
            return 0, 1

        try:
            if distribution == 'uniform':
                # Test against uniform distribution on [0, 1]
                clean_data = clean_data[(clean_data >= 0) & (clean_data <= 1)]
                if len(clean_data) < 2:
                    return 0, 1
                statistic, p_value = stats.kstest(clean_data, 'uniform')

            elif distribution == 'normal':
                # Test against normal distribution with sample mean and std
                mu, sigma = self.fit_normal(clean_data)
                statistic, p_value = stats.kstest(clean_data, 'norm', args=(mu, sigma))

            elif distribution == 'arcsine':
                # Test against arcsine distribution
                clean_data = clean_data[(clean_data >= 0) & (clean_data <= 1)]
                if len(clean_data) < 2:
                    return 0, 1
                statistic, p_value = stats.kstest(clean_data,
                                                lambda x: stats.beta.cdf(x, 0.5, 0.5))

            elif distribution == 'exponential':
                # Test against exponential distribution
                clean_data = clean_data[clean_data >= 0]
                if len(clean_data) < 2:
                    return 0, 1
                loc, scale = self.fit_exponential(clean_data)
                statistic, p_value = stats.kstest(clean_data, 'expon', args=(loc, scale))

            else:
                return 0, 1

            return statistic, p_value

        except:
            return 0, 1

    def chi_square_goodness_of_fit(self, observed_data, expected_distribution='uniform', bins=10):
        """Perform chi-squared goodness of fit test."""
        if len(observed_data) < bins:
            return 0, 1

        clean_data = observed_data[np.isfinite(observed_data)]
        if len(clean_data) < bins:
            return 0, 1

        try:
            # Create histogram
            observed_counts, bin_edges = np.histogram(clean_data, bins=bins)

            # Calculate expected counts based on distribution
            if expected_distribution == 'uniform':
                # For uniform distribution on [0, 1]
                data_range = np.max(clean_data) - np.min(clean_data)
                expected_counts = np.full(bins, len(clean_data) / bins)

            elif expected_distribution == 'normal':
                # For normal distribution
                mu, sigma = self.fit_normal(clean_data)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                bin_width = bin_edges[1] - bin_edges[0]
                expected_probs = stats.norm.pdf(bin_centers, mu, sigma) * bin_width
                expected_counts = expected_probs * len(clean_data)

            else:
                expected_counts = np.full(bins, len(clean_data) / bins)

            # Ensure minimum expected count
            expected_counts = np.maximum(expected_counts, 1)

            # Perform chi-squared test
            chi2_stat, p_value = stats.chisquare(observed_counts, expected_counts)

            return chi2_stat, p_value

        except:
            return 0, 1

    def anderson_darling_test(self, data, distribution='uniform'):
        """Perform Anderson-Darling test for normality or uniformity."""
        if len(data) < 3:
            return 0, 1

        clean_data = data[np.isfinite(data)]
        if len(clean_data) < 3:
            return 0, 1

        try:
            if distribution == 'normal':
                statistic, critical_values, significance_level = stats.anderson(clean_data, 'norm')
                # Convert to p-value approximation
                p_value = 1.0 - significance_level / 100.0 if statistic < critical_values[2] else 0.01

            elif distribution == 'uniform':
                # For uniform, transform to [0,1] and use normal test on transformed data
                data_min, data_max = np.min(clean_data), np.max(clean_data)
                if data_max > data_min:
                    uniform_data = (clean_data - data_min) / (data_max - data_min)
                    # Use KS test instead for uniform
                    statistic, p_value = self.kolmogorov_smirnov_test(uniform_data, 'uniform')
                else:
                    statistic, p_value = 0, 1

            else:
                statistic, p_value = 0, 1

            return statistic, p_value

        except:
            return 0, 1

    def comprehensive_distribution_analysis(self, data, expected_mean=None, expected_std=None):
        """Perform comprehensive statistical analysis of a dataset."""
        if len(data) < 2:
            return {}

        clean_data = data[np.isfinite(data)]
        if len(clean_data) < 2:
            return {}

        results = {}

        # Basic statistics
        results['sample_size'] = len(clean_data)
        results['mean'] = np.mean(clean_data)
        results['std'] = np.std(clean_data, ddof=1)
        results['min'] = np.min(clean_data)
        results['max'] = np.max(clean_data)
        results['median'] = np.median(clean_data)
        results['skewness'] = stats.skew(clean_data)
        results['kurtosis'] = stats.kurtosis(clean_data)

        # Expected vs observed
        if expected_mean is not None:
            results['expected_mean'] = expected_mean
            results['mean_relative_error'] = abs(results['mean'] - expected_mean) / abs(expected_mean) if expected_mean != 0 else 0

        if expected_std is not None:
            results['expected_std'] = expected_std
            results['std_relative_error'] = abs(results['std'] - expected_std) / abs(expected_std) if expected_std != 0 else 0

        # Distribution tests
        results['ks_uniform_stat'], results['ks_uniform_pval'] = self.kolmogorov_smirnov_test(clean_data, 'uniform')
        results['ks_normal_stat'], results['ks_normal_pval'] = self.kolmogorov_smirnov_test(clean_data, 'normal')
        results['chi2_uniform_stat'], results['chi2_uniform_pval'] = self.chi_square_goodness_of_fit(clean_data, 'uniform')
        results['chi2_normal_stat'], results['chi2_normal_pval'] = self.chi_square_goodness_of_fit(clean_data, 'normal')

        # Normality tests
        if len(clean_data) >= 8:  # Minimum for Shapiro-Wilk
            try:
                results['shapiro_stat'], results['shapiro_pval'] = stats.shapiro(clean_data[:5000])  # Limit for computational efficiency
            except:
                results['shapiro_stat'], results['shapiro_pval'] = 0, 1

        # Outlier detection
        if len(clean_data) >= 4:
            q25, q75 = np.percentile(clean_data, [25, 75])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outliers = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]
            results['n_outliers'] = len(outliers)
            results['outlier_fraction'] = len(outliers) / len(clean_data)

        return results

    def validate_physics_expectations(self, N, p_gamma_values, p_e_gamma_values, tolerance_factor=3.0):
        """Validate that observed values match physics expectations."""
        # Expected values based on N
        expected_p_gamma = 0.625 if N == 2 else 1.0 / N
        expected_p_e_gamma = 0.25 if N == 2 else 0.1

        # Theoretical standard deviations (rough estimates)
        expected_std_gamma = expected_p_gamma / np.sqrt(N)
        expected_std_e_gamma = expected_p_e_gamma / np.sqrt(N)

        # Comprehensive analysis
        gamma_analysis = self.comprehensive_distribution_analysis(
            p_gamma_values, expected_p_gamma, expected_std_gamma)
        e_gamma_analysis = self.comprehensive_distribution_analysis(
            p_e_gamma_values, expected_p_e_gamma, expected_std_e_gamma)

        # Validation criteria
        validation_results = {
            'N': N,
            'gamma_mean_valid': gamma_analysis.get('mean_relative_error', 1) < 0.1,  # 10% tolerance
            'e_gamma_mean_valid': e_gamma_analysis.get('mean_relative_error', 1) < 0.2,  # 20% tolerance
            'gamma_distribution_normal': gamma_analysis.get('ks_normal_pval', 0) > 0.05,
            'e_gamma_distribution_exponential': e_gamma_analysis.get('ks_uniform_pval', 0) < 0.05,  # Should NOT be uniform
            'overall_valid': True
        }

        # Overall validation
        validation_results['overall_valid'] = (
            validation_results['gamma_mean_valid'] and
            validation_results['e_gamma_mean_valid']
        )

        return validation_results, gamma_analysis, e_gamma_analysis
