import numpy as np
import pandas as pd
from scipy.stats import norm
from empyrical.stats import *


def df_to_array_dict(df: pd.DataFrame) -> dict:
    """
    df columns will be the key, values are stored in array
    """

    return {k: np.array(v) for k, v in df.copy().to_dict(orient="list").items()}

def simulate_ret(ret_df, ld, no_of_observations=5000, lookback=260, std_dev_estimation_method="exponential",
                 is_random_seed=True, asarray=True, random_type="uniform", fill_random=None, seed=None):

    """
    ret_df is relative_returns_% of all risk factors (i.e. generic curves and physical prices) based on X-day window
    simulatedRet_df is also relative_returns_% of all risk factors based on number of simulations
    """

    if is_random_seed:
        np.random.seed(seed)

    # Ensure ret is in ascending order
    if ret_df.index[0] > ret_df.index[-1]:
        ret_df = ret_df.iloc[::-1]

    # ret = ret_df.values

    std_dev_estimation_method = std_dev_estimation_method.lower()

    # assert w.shape[0] == ret.shape[1], "Weights dimension mismatch. Please ensure weight matrix to be of size
    # (numAssets, ?)"
    assert ret_df.shape[0] == lookback, f"Lookback period {lookback} is different from ret length {ret_df.shape[0]}"
    assert type(std_dev_estimation_method) == str and std_dev_estimation_method.lower() in ["equal", "exponential"], \
        "std_dev_estimation_method only takes 'equal' or 'exponential' as input"
    assert 0 < ld < 1, "Lambda value has to be greater than 0 and less than 1"

    # Taper returns to lookback period
    # ret = ret_df[-lookback:, :]

    # retSD, covar, correl = covar_SD(ret, sdEstimationMethod, lookback, ld)

    mean, retSD, variance, covar, correl = ret_df.pipe(calc_ret_stats, std_dev_estimation_method, ld)

    covar_values = covar.values

    ret = ret_df.values

    # Simulate returns using Cholesky Decomposition
    # expReturn = np.mean(ret, axis=0)
    # simulatedRet = np.dot(np.random.normal(loc=0, scale=1, size=(numObs, ret.shape[1])),
    # np.linalg.cholesky(covar).T) + expReturn

    random_number = None

    if random_type == "standard":
        if fill_random == "col":
            random_number = np.random.normal(loc=0, scale=1, size=no_of_observations * ret.shape[1])
            random_number = random_number.reshape(ret.shape[1], no_of_observations).T
        elif fill_random == "row":
            random_number = np.random.normal(loc=0, scale=1, size=(no_of_observations, ret.shape[1]))
    elif random_type == "uniform":
        # Generate uniform random numbers and fill column
        uniform_random_number = np.random.uniform(low=0.0, high=1.0, size=no_of_observations * ret.shape[1])
        uniform_random_number = uniform_random_number.reshape(ret.shape[1], no_of_observations).T

        # Convert uniform dist to standard normal dist of each column
        random_number = np.zeros_like(uniform_random_number)
        for i in range(random_number.shape[1]):
            random_number[:, i] = norm.ppf(uniform_random_number[:, i], loc=0, scale=1)

    random_number_df = pd.DataFrame(random_number)

    try:
        # simulatedRet = np.dot(np.random.normal(loc=0, scale=1, size=(numObs, ret.shape[1])),
        # np.linalg.cholesky(covar).T)
        simulatedRet = np.dot(random_number, np.linalg.cholesky(covar_values).T)
    except np.linalg.LinAlgError:
        # print(e)
        # eigenvalues = np.linalg.eigvals(covar)
        # if np.all(eigenvalues > 0):
        jitter = 1e-12
        covar_values = covar_values + np.eye(covar_values.shape[0]) * jitter
        # simulatedRet = np.dot(np.random.normal(loc=0, scale=1, size=(numObs, ret.shape[1])),
        # np.linalg.cholesky(covar_values).T)
        simulatedRet = np.dot(random_number, np.linalg.cholesky(covar_values).T)

    simulatedRet = pd.DataFrame(simulatedRet)
    simulatedRet.columns = ret_df.columns
    simulatedRet_df = simulatedRet.copy()

    random_number_df.columns = simulatedRet.columns

    mean_simulated, retSD_simulated, variance_simulated, covar_simulated, correl_simulated = (
        simulatedRet_df.pipe(calc_ret_stats, std_dev_estimation_method="equal", ld=None))

    skewness = pd.DataFrame(stats.skew(simulatedRet_df.values)).T
    skewness.columns = simulatedRet_df.columns

    random_skewness = pd.DataFrame(stats.skew(random_number_df.values)).T
    random_skewness.columns = simulatedRet_df.columns

    kurtosis = pd.DataFrame(stats.kurtosis(simulatedRet_df.values)).T
    kurtosis.columns = simulatedRet_df.columns

    random_kurtosis = pd.DataFrame(stats.kurtosis(random_number_df.values)).T
    random_kurtosis.columns = simulatedRet_df.columns

    # if asarray:
    simulatedRet = simulatedRet.pipe(df_to_array_dict)

    return (random_number_df, simulatedRet_df, simulatedRet, mean, variance, retSD, covar, correl, mean_simulated,
            variance_simulated, retSD_simulated, covar_simulated, correl_simulated, skewness, kurtosis, random_skewness,
            random_kurtosis)


# def simulate_ret(ret_df, ld, numObs=5000, lookback=260, sdEstimationMethod="exponential", isRamSeed=False,
# asarray=True, ramdom_type= "uniform", fill_random=None):
#
#     """
#     ret_df is df format
#     """
#
#     if isRamSeed:
#         np.random.seed(42)
#
#     # Ensure ret is in ascending order
#     if ret_df.index[0] > ret_df.index[-1]:
#         ret_df = ret_df.iloc[::-1]
#
#     ret = ret_df.values
#
#     sdEstimationMethod = sdEstimationMethod.lower()
#
#     # assert w.shape[0] == ret.shape[1], "Weights dimension mismatch. Please ensure weight matrix to be of size
#     (numAssets, ?)"
#     assert ret.shape[0] >= lookback, f"Lookback period {lookback} exceeds ret length {ret.shape[0]}"
#     assert type(sdEstimationMethod) == str and sdEstimationMethod.lower() in ["equal", "exponential"],
#     "sdEstimationMethod only takes 'equal' or 'exponential' as input"
#     assert ld > 0 and ld < 1, "Lambda value has to be greater than 0 and less than 1"
#
#     # Taper returns to lookback period
#     ret = ret[-lookback:, :]
#
#     retSD, covar, correl = covar_SD(ret, sdEstimationMethod, lookback, ld)
#
#     # Simulate returns using Cholesky Decomposition
#     # expReturn = np.mean(ret, axis=0)
#     # simulatedRet = np.dot(np.random.normal(loc=0, scale=1, size=(numObs, ret.shape[1])),
#     np.linalg.cholesky(covar).T) + expReturn
#
#     random_number = None
#
#     if ramdom_type == "standard":
#         if fill_random =="col":
#             random_number = np.random.normal(loc=0, scale=1, size=numObs * ret.shape[1])
#             random_number = random_number.reshape(ret.shape[1], numObs).T
#         elif fill_random =="row":
#             random_number = np.random.normal(loc=0, scale=1, size=(numObs, ret.shape[1]))
#     elif ramdom_type == "uniform":
#         # Generate uniform random numbers and fill column
#         uniform_random_number = np.random.uniform(low=0.0, high=1.0, size=numObs * ret.shape[1])
#         uniform_random_number = uniform_random_number.reshape(ret.shape[1], numObs).T
#
#         # Convert uniform dist to standard normal dist of each column
#         random_number = np.zeros_like(uniform_random_number)
#         for i in range(random_number.shape[1]):
#             random_number[:, i] = norm.ppf(uniform_random_number[:, i], loc=0, scale=1)
#
#     try:
#         # simulatedRet = np.dot(np.random.normal(loc=0, scale=1, size=(numObs, ret.shape[1])), np.linalg.cholesky(covar).T)
#         simulatedRet = np.dot(random_number, np.linalg.cholesky(covar).T)
#     except np.linalg.LinAlgError:
#         # print(e)
#         # eigenvalues = np.linalg.eigvals(covar)
#         # if np.all(eigenvalues > 0):
#         jitter = 1e-12
#         covar = covar + np.eye(covar.shape[0]) * jitter
#         # simulatedRet = np.dot(np.random.normal(loc=0, scale=1, size=(numObs, ret.shape[1])), np.linalg.cholesky(covar).T)
#         simulatedRet = np.dot(random_number, np.linalg.cholesky(covar).T)
#
#     simulatedRet = pd.DataFrame(simulatedRet)
#     simulatedRet.columns = ret_df.columns
#
#     if asarray:
#         simulatedRet = simulatedRet.pipe(df_to_array_dict)
#
#     return simulatedRet, retSD, correl


def ewma(lookback=260, ld=0.94):
    # weight in ascending order

    assert 0 < ld < 1, "Lambda value has to be greater than 0 and less than 1"

    weights = np.power(ld, range(lookback - 1, -1, -1)) * (1 - ld)

    return weights.reshape(-1, 1)

def covar_SD(ret, std_dev_estimation_method="equal", lookback=260, ld=0.94):

    assert ret.shape[0] >= lookback, f"Lookback period {lookback} exceeds ret length {ret.shape[0]}"
    assert type(std_dev_estimation_method) == str and std_dev_estimation_method.lower() in ["equal", "exponential"], \
        "std_dev_estimation_method only takes 'equal' or 'exponential' as input"
    assert 0 < ld < 1, "Lambda value has to be greater than 0 and less than 1"

    # Align letter casing
    std_dev_estimation_method = std_dev_estimation_method.lower()

    # Ensure returns are of required length
    ret = ret[-lookback:, :]

    numAssets = ret.shape[1]

    if std_dev_estimation_method == "exponential":

        sdWeights = ewma(lookback, ld)
        covar = np.zeros(shape=(numAssets, numAssets))

        for i in range(numAssets):
            for j in range(numAssets):
                if j < i:
                    continue

                # Calculate covariance matrix
                covar[i, j] = np.dot(sdWeights.T, ret[:, i] * ret[:, j])

                # Copy value from upper half to lower half
                covar[j, i] = covar[i, j]

    elif std_dev_estimation_method == "equal":

        covar = np.cov(ret.T, ddof=0)

        if covar.ndim == 0:
            covar = np.reshape(covar, (1,))

    # Calculate standard deviation for each asset / risk factors
    retSD = np.sqrt(np.diag(covar)).reshape(-1, 1)

    # Calculate correlation from covariance matrix
    correl = covar / retSD / retSD.T
    correl[np.isnan(correl)] = 0

    return retSD, covar, correl


def covarSD(w, ret, std_dev_estimation_method="equal", lookback=260, ld=0.94):

    assert w.shape[0] == ret.shape[1], ("Weights dimension mismatch. Please ensure weight matrix to be of size "
                                        "(numAssets, ?)")
    assert ret.shape[0] >= lookback, f"Lookback period {lookback} exceeds ret length {ret.shape[0]}"
    assert type(std_dev_estimation_method) == str and std_dev_estimation_method.lower() in ["equal", "exponential"], \
        "std_dev_estimation_method only takes 'equal' or 'exponential' as input"
    assert ld > 0 and ld < 1, "Lambda value has to be greater than 0 and less than 1"

    # Align letter casing
    std_dev_estimation_method = std_dev_estimation_method.lower()

    # Ensure returns are of required length
    ret = ret[-lookback:, :]

    numAssets = ret.shape[1]

    if std_dev_estimation_method == "exponential":

        sdWeights = ewma(lookback, ld)
        covar     = np.zeros(shape=(numAssets, numAssets))

        for i in range(numAssets):
            for j in range(numAssets):
                if j < i:
                    continue

                # Calculate covariance matrix
                covar[i, j] = np.dot(sdWeights.T, ret[:, i] * ret[:, j])

                # Copy value from upper half to lower half
                covar[j, i] = covar[i, j]

    elif std_dev_estimation_method == "equal":

        covar = np.cov(ret.T, ddof=0)

    weightedCovar = np.dot(w.T, covar)
    portSD = np.zeros(shape=(weightedCovar.shape[0], 1))

    for i in range(weightedCovar.shape[0]):
        portSD[i] = np.sqrt(np.dot(weightedCovar[i, :], w[:, i]))

    # Calculate standard deviation for each asset / risk factors
    retSD  = np.sqrt(np.diag(covar)).reshape(-1, 1)

    # Calculate correlation from covariance matrix
    correl = covar / retSD / retSD.T
    correl[np.isnan(correl)] = 0

    return portSD, retSD, covar, correl


def calc_ewm_stats(df_raw, ld):
    """
    Calculate EWM mean, variance, standard deviation, and covariance.

    Parameters:
        df_raw (pd.DataFrame): Input DataFrame with shape (n_rows, n_cols)
        ld ; span (float): Span for EWMA smoothing (related to lambda)

    Returns:
        ewm_mean (pd.DataFrame): EWM mean for each column.
        ewm_var (pd.DataFrame): EWM variance for each column.
        ewm_std (pd.DataFrame): EWM standard deviation for each column.
        ewm_cov (pd.DataFrame): EWM covariance matrix at the last time step.
    """

    df = df_raw.copy()

    # Ensure ret is in ascending order
    if df.index[0] > df.index[-1]:
        df = df.iloc[::-1]

    # Convert lambda to span
    # ld = 1 - alpha
    # alpha = 2 / (1 + span)
    span = 2 / (1 - ld) - 1

    # EWM mean
    ewm_mean = df.ewm(span=span, adjust=True).mean()

    # EWM variance
    ewm_var = df.ewm(span=span, adjust=True).var()

    # EWM standard deviation
    ewm_std = df.ewm(span=span, adjust=True).std()

    # EWM covariance matrix (extract the last covariance matrix)
    ewm_cov = df.ewm(span=span, adjust=True).cov(pairwise=True)
    ewm_cov = ewm_cov.loc[df.index[-1]]  # Covariance matrix at last time step

    return ewm_mean, ewm_var, ewm_std, ewm_cov


def calc_ret_stats(df_raw, std_dev_estimation_method, ld=None):
    """
    Calculate weighted mean, variance, standard deviation, and covariance matrix
    using exponential weights.

    Ensure consistency of using population as the same impact of ddof=0

    Parameters:
        df (pd.DataFrame): DataFrame of log returns (rows=observations, cols=contracts).
        ld (float): Decay factor for exponential weights (0 < lambda_ < 1).

    """

    df = df_raw.copy()

    # Ensure ret is in ascending order
    if df.index[0] > df.index[-1]:
        df = df.iloc[::-1]

    if std_dev_estimation_method == "exponential":
        # Number of rows
        n_rows = len(df)

        # Define exponential weights
        weights = np.array([ld ** i for i in range(n_rows)])[::-1]  # Decaying weights
        weights /= weights.sum()  # Normalize weights to sum to 1

        # 1. Weighted Mean
        weighted_mean = (df.T @ weights).T  # Apply weights to each column

        # 2. Weighted Variance
        df_centered = df - weighted_mean  # Center the data
        weighted_variance = (df_centered ** 2).T @ weights  # Weighted squared deviations

        # 3. Weighted Standard Deviation
        weighted_std = np.sqrt(weighted_variance)

        # 4. Weighted Covariance Matrix
        weighted_cov = (df_centered.mul(weights, axis=0).T @ df_centered) / weights.sum()

        # 5. Weighted Correlation Matrix
        # Correlation = Covariance / (std_i * std_j)
        std_matrix = np.outer(weighted_std, weighted_std)  # Outer product of std deviations
        weighted_corr = weighted_cov / std_matrix  # Element-wise division to get correlation matrix
        np.fill_diagonal(weighted_corr.values, 1.0)  # Set diagonal to 1 for correlation matrix

        # weighted_mean = weighted_mean.to_dict()
        # weighted_variance = weighted_variance.to_dict()
        # weighted_std = weighted_std.to_dict()
        # weighted_cov = weighted_cov.values
        # weighted_corr = weighted_corr.values

        return weighted_mean, weighted_std, weighted_variance, weighted_cov, weighted_corr

    elif std_dev_estimation_method == "equal":
        return df_raw.mean(), df_raw.std(), df_raw.var(), df_raw.cov(ddof=0), df_raw.corr()

# def calc_standard_stats(df_raw):
#
#     # return df_raw.mean().to_dict(), df_raw.std().to_dict(), df_raw.var().to_dict(), df_raw.cov(ddof=0).values, df_raw.corr().values
#     return df_raw.mean(), df_raw.std(), df_raw.var(), df_raw.cov(ddof=0), df_raw.corr()