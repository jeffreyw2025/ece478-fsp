# Jeffrey Wong | ECE-478 | PSet #2
# Portfolio Analysis- Data Analysis

# Performs market portfolio and sector analysis on S&P 500 data acquired from the dataack program.
# IMPORTANT NOTE: The "market portfolios" that we compute are not always a literal market porfolio!
# For more information, see the compute_market_portfolio function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Helper function to compute the minimum variance portfolio of a set of stocks as a column vector
def compute_mvp(df):
    one_vec = np.ones((np.shape(df)[1], 1)) # Column vector of all 1s of length equal to # of stocks
    Cinv = np.linalg.inv(np.cov(df, rowvar = False))
    w = (Cinv @ one_vec) / (one_vec.T @ Cinv @ one_vec)
    return w

# Helper function to compute the market portfolio of a set of stocks as a column vector
def compute_market_portfolio(df, label = None):
    one_vec = np.ones((np.shape(df)[1], 1)) # Column vector of all 1s of length equal to # of stocks
    mu_vec = np.asmatrix(np.mean(df, axis = 0)).T # Note that our returns are excess returns so we can just take a direct mean
    Cinv = np.linalg.inv(np.cov(df, rowvar = False))
    w = 0
    # Note that a true market portfolio only exists when mu_MVP > r_rf (0 here since we deal with excess returns)
    # If MVP return negative, instead select arbitrary target mu_ex and find a w that minimizes w^T @ C @ w
    # We'll use a target return of 0.001
    (sigma_mvp, mu_mvp) = compute_portfolio_returns(df, compute_mvp(df))
    if(mu_mvp < 0):
        m_tilde = np.concatenate((mu_vec, one_vec), axis = 1)
        mu_tilde = np.matrix('0.001; 1')
        B = m_tilde.T @ Cinv @ m_tilde
        w = Cinv @ m_tilde @ np.linalg.inv(B) @ mu_tilde
    else:
        w = (Cinv @ mu_vec) / (one_vec.T @ Cinv @ mu_vec)
    if label is not None:
        print("Market portfolio coefficients for {0}:".format(label))
        print(w)
    return w

# Helper function to compute (sigma, mu) for a given portfolio on a given set of assets
def compute_portfolio_returns(df, w):
    # w is expected to be a column vector so need to transpose to form row vector
    mu = w.T @ np.mean(df, axis = 0)
    var = w.T @ np.cov(df, rowvar = False) @ w
    if(var < 0): 
        var = 0
    # print("Sigma = {0}, Mu = {1}".format(np.sqrt(var), mu))
    return (np.sqrt(var), mu)

# Helper functions to generate (sigma, mu) graphs for a given dataset
def analyze_portfolio_returns(df, plot_title, plot_fname):
    mvp_weights = compute_mvp(df)
    # Note that if MVP is negative the weights given are not a "true" market portfolio but an approximate one
    # formed by aiming for some target return
    mktp_weights = compute_market_portfolio(df, plot_title)

    # Generate efficient frontier points from MVP and MP
    (sigma_mvp, mu_mvp) = compute_portfolio_returns(df, mvp_weights)
    (sigma_mktp, mu_mktp) = compute_portfolio_returns(df, mktp_weights)
 
    sigma_ef = np.zeros((21, 1))
    mu_ef = np.zeros((21, 1))
    for i in range(21):
        comb_weights = 0.1*i*mktp_weights + (1 - 0.1*i)*mvp_weights
        (sigma_comb, mu_comb) = compute_portfolio_returns(df, comb_weights)
        sigma_ef[i] = sigma_comb
        mu_ef[i] = mu_comb

    # Generate CML from market portfolio
    sigma_cml = 0.1 * sigma_mktp * np.arange(21)
    mu_cml = 0.1 * mu_mktp * np.arange(21)

    plt.figure()
    plt.plot(sigma_ef, mu_ef, 'r', label = "Efficient Frontier")
    plt.plot(sigma_cml.T, mu_cml, 'b--', label = "Capital Market Line")
    plt.plot(sigma_mktp, mu_mktp, 'g*', label = "Market Portfolio")
    plt.plot(sigma_mvp, mu_mvp, 'k*', label = "Minimum Variance Portfolio")
    plt.title("Sigma vs mu for "+plot_title)
    plt.xlabel("Risk (sigma)")
    plt.ylabel("Expected excess return (mu)")
    plt.legend()
    plt.savefig(plot_fname)
    return 0            

# Helper function to generate betas given the S&P 500
def compute_betas_vs_SP500(df, SP_500_returns):
    cov_matrix = np.cov(df.to_numpy(), SP_500_returns, rowvar = False)
    betas = cov_matrix[-1, :-1] / cov_matrix[-1, -1]
    return betas

# Helper function to generate betas given a market portfolio (either passed as argument or computed from data)
def compute_betas_vs_MP(df, w_mkt = None):
    # By default compute market portfolio from given dataframe
    if w_mkt is None:
        w_mkt = compute_market_portfolio(df)
    w_mkt = w_mkt / np.sum(w_mkt) # Normalize weights
    mkt_returns = np.multiply(w_mkt.T, df.to_numpy())
    mkt_returns = np.sum(mkt_returns, axis = 1)
    # print(mkt_returns)
    cov_matrix = np.cov(df.to_numpy(), mkt_returns, rowvar = False)
    betas = cov_matrix[-1, :-1] / cov_matrix[-1, -1]
    return betas

# Problem 1- Full-Year Data Analysis
def full_year_analysis(data_full, symbols_sector_1, symbols_sector_2):
    # Create dfs for sector data and seperate out S&P 500 index
    data_sector_1 = data_full[symbols_sector_1]
    data_sector_2 = data_full[symbols_sector_2]
    data_index = data_full["^SPX"]
    data_full = data_full[symbols_sector_1 + symbols_sector_2]
    
    # Parts a and b
    print("Parts a and b")
    print("S&P 500 Index (sigma, mu)")
    mu_SP500 = np.mean(data_index, axis = 0)
    sigma_SP500 = np.sqrt(np.cov(data_index, rowvar = False))
    print(mu_SP500)
    print(sigma_SP500)
    print("Full Set expected returns, condition #, correlation")
    mu_full = np.mean(data_full, axis = 0)
    cov_full = np.cov(data_full, rowvar = False)
    print("Full Set Expected Returns")
    print(mu_full)
    print("Full Set Covariance Matrix")
    print(cov_full)
    eigvs_full = np.abs(np.linalg.eigvals(cov_full))
    print("Condition number for full covariance matrix: {0}".format(np.max(eigvs_full)/np.min(eigvs_full))) # Based on what I see this is extremely close to singular!
    pcorr_full = np.corrcoef(data_full, rowvar=False)
    print(pcorr_full)
    print("Mean absolute Pearson correlation for market: {0}".format(np.mean(np.abs(pcorr_full))))
    
    print("Sector 1 condition #, correlation")
    cov_sec1 = np.cov(data_sector_1, rowvar = False)
    print(cov_sec1)
    eigvs_sec1 = np.abs(np.linalg.eigvals(cov_sec1))
    print("Condition number for full covariance matrix: {0}".format(np.max(eigvs_sec1)/np.min(eigvs_sec1))) # The condition numbers in the sectors are lower!
    pcorr_sec1 = np.corrcoef(data_sector_1, rowvar=False)
    print(pcorr_sec1)
    print("Mean absolute Pearson correlation for Sector 1: {0}".format(np.mean(np.abs(pcorr_sec1))))

    print("Sector 2 condition #, correlation")
    cov_sec2 = np.cov(data_sector_2, rowvar = False)
    print(cov_sec2)
    eigvs_sec2 = np.abs(np.linalg.eigvals(cov_sec2))
    print("Condition number for full covariance matrix: {0}".format(np.max(eigvs_sec2)/np.min(eigvs_sec2)))
    pcorr_sec2 = np.corrcoef(data_sector_2, rowvar=False)
    print(pcorr_sec2)
    print("Mean absolute Pearson correlation for Sector 2: {0}".format(np.mean(np.abs(pcorr_sec2))))

    # Considering the average absolute value of the Pearson coefficients for each sector versus both sectors combined, 
    # we can see there thends to be a higher correlation within sectors than in the market as a whole. In fact, the
    # cross-sector correlation seems to match the market correlation!
    
    # Part c
    print("\nPart c: Single Factor Model of Market")
    mu_SP500_avg = np.mean(data_full, axis = 1)
    # For beta of the S&P 500 we consider the portfolio to essentially be one of everything!
    beta_SP500 = compute_betas_vs_SP500(data_full, data_index)
    alpha_SP500 = mu_full - beta_SP500 * np.mean(mu_SP500_avg)
    epsilon_SP500 = data_full - alpha_SP500 - np.outer(mu_SP500_avg, beta_SP500)
    print("Single Factor Model Betas")
    print(beta_SP500)
    print("Single Factor Model Alphas")
    print(alpha_SP500)
    print("Single Factor Model Residuals")
    print(epsilon_SP500)
    print(np.mean(np.mean(epsilon_SP500, axis = 0))) # Check if residuals are zero-mean
    epsilon_SP500_corr = np.corrcoef(epsilon_SP500, rowvar = False)
    print("Correlation matrix for epsilon_k, Full Set")
    print(epsilon_SP500_corr)
    epsilon_sec1_corr = np.corrcoef(epsilon_SP500[symbols_sector_1], rowvar = False)
    print("Correlation matrix for epsilon_k, Tech Stocks")
    print(epsilon_sec1_corr)
    epsilon_sec2_corr = np.corrcoef(epsilon_SP500[symbols_sector_2], rowvar = False)
    print("Correlation matrix for epsilon_k, Health Stocks")
    print(epsilon_sec2_corr)
    # From what I can see it seems there is a slightly stronger correlation within sectors than across sectors
    # The diagonal model is probably not a great assumption here since there are nontrivial correlations between
    # the residuals of different stocks!
    
    # Part d
    print("\nPart d- Sigma-mu graphs")
    analyze_portfolio_returns(data_full, "Full Market", "fullmkt_portfolio_plot.png")

    analyze_portfolio_returns(data_sector_1, "Tech Stocks", "tech_portfolio_plot.png")

    analyze_portfolio_returns(data_sector_2, "Health Stocks", "health_portfolio_plot.png")
    # Our graph for health stocks does not have the "CML" tangent to the efficient frontier because
    # the "market portfolio" we computed is not an actual market portfolio (which doesn't exist because
    # mu_ex_MVP < 0), so the "CML" is just a CAL with no special properties
    
    # Part e

    # Every one of the market portfolios incorporates some degree of short selling.
    # Of the three MPs, the one for the health stocks seems "closest" to the S&P 500,
    # although none of them really match my expectation of being around 0.002 (or 0.2) for the
    # 5-stock subsets with scaling based on returns or something like that.

    # Part f
    
    print("\nPart f- Betas of stocks vs Mkt Portfolio")
    print(compute_betas_vs_MP(data_full))
    print("Betas for tech stocks")
    print(compute_betas_vs_MP(data_sector_1))
    print("Betas for market stocks")
    print(compute_betas_vs_MP(data_sector_2))
    # Betas can be more than 1 if the variance of the stock exceeds the variance of the market portfolio,
    # as beta is not a true correlation coefficient!

    return 0

# Problem 2- Security Market Line Analysis
def sml_analysis(data_full, symbols_sector_1, symbols_sector_2):
    data_jan = data_full["2023/01/01":"2023/01/31"]
    data_feb = data_full["2023/02/01":"2023/02/28"]
    betas = compute_betas_vs_SP500(data_jan[symbols_sector_1 + symbols_sector_2], data_jan["^SPX"])
    # Want to compute beta and return for market portfolios (Remember the health "Market Portfolio"
    # is NOT a true market portfolio)
    sector_1_mp_jan_returns = compute_market_portfolio(data_jan[symbols_sector_1]).T @ data_jan[symbols_sector_1].T
    sector_2_mp_jan_returns = compute_market_portfolio(data_jan[symbols_sector_2]).T @ data_jan[symbols_sector_2].T
    beta_sector_1 = compute_betas_vs_SP500(sector_1_mp_jan_returns.T, data_jan["^SPX"])
    beta_sector_2 = compute_betas_vs_SP500(sector_2_mp_jan_returns.T, data_jan["^SPX"])
    (_, sector_1_mp_feb_returns) = compute_portfolio_returns(data_feb[symbols_sector_1], compute_market_portfolio(data_jan[symbols_sector_1]))
    (_, sector_2_mp_feb_returns) = compute_portfolio_returns(data_feb[symbols_sector_2], compute_market_portfolio(data_jan[symbols_sector_2]))
    print("Individual Stock Betas:")
    print(betas)
    print("Tech MP Beta: {0}\t Health MP Beta: {1}".format(beta_sector_1, beta_sector_2))
    feb_returns = np.mean(data_feb, axis = 0)
    print("February Returns")
    print(feb_returns)
    print("TECH_MP\t{0}".format(sector_1_mp_feb_returns))
    print("HLTH_MP\t{0}".format(sector_2_mp_feb_returns))

    # Plot stuff
    plt.figure()
    plt.plot(0.1 * np.arange(26), 0.1 * np.arange(26) * feb_returns["^SPX"], "b--", label = "Security Market Line")
    for i in range(10):
        plt.plot(betas[i], feb_returns[i], "k*")
        plt.text(betas[i], feb_returns[i], data_full.columns[i])
    plt.plot(1, feb_returns["^SPX"], "r*", label = "S&P 500")
    plt.text(1, feb_returns["^SPX"], "S&P 500")
    plt.plot(beta_sector_1, sector_1_mp_feb_returns, "g*", label = "Tech MP")
    plt.plot(beta_sector_2, sector_2_mp_feb_returns, "y*", label = "Health MP")
    plt.text(beta_sector_1, sector_1_mp_feb_returns, "Tech MP")
    plt.text(beta_sector_2, sector_2_mp_feb_returns, "Health MP")
    plt.title("Beta vs returns for Full Market")
    plt.xlabel("Beta")
    plt.ylabel("Expected excess return (mu)")
    plt.legend()
    plt.savefig("sml_full.png")
    return 0

# Helper function to compute monthly returns from a (assumed) month-long block of data
def compute_monthly_returns(df):
    month_returns = np.prod(df + 1, axis = 0) - 1
    return month_returns

# Problem 3- Market Portfolio Analysis
def seq_mp_analysis(data_full, symbols_sector_1, symbols_sector_2):
    # A negative portfolio value can arise due to short selling- If you short many shares of a stock that ends up
    # massively increasing in price, it can cause a large enough loss to take you into the negative!
    # Thankfully it doesn't seem to happen here!
    month_starts = ["2023/01/01", "2023/02/01", "2023/03/01", "2023/04/01", "2023/05/01", "2023/06/01", "2023/07/01", "2023/08/01", "2023/09/01", "2023/10/01", "2023/11/01", "2023/12/01"]
    month_ends = ["2023/01/31", "2023/02/28", "2023/03/31", "2023/04/30", "2023/05/31", "2023/06/30", "2023/07/31", "2023/08/31", "2023/09/30", "2023/10/31", "2023/11/30", "2023/12/31"]
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    wealth_full_mkt = np.ones((12, 1))
    wealth_sec1_only = np.ones((12, 1))
    wealth_sec2_only = np.ones((12, 1))
    for i in range(11):
        data_thismonth = data_full[month_starts[i]:month_ends[i]]
        mp_full = compute_market_portfolio(data_thismonth, label = "{0} Full Mkt Portfolio Weights".format(month_names[i]))
        mp_sector_1 = compute_market_portfolio(data_thismonth[symbols_sector_1], label = "{0} Tech Mkt Portfolio Weights".format(month_names[i]))
        mp_sector_2 = compute_market_portfolio(data_thismonth[symbols_sector_2], label = "{0} Health Mkt Portfolio Weights".format(month_names[i]))
        data_nextmonth = data_full[month_starts[i+1]:month_ends[i+1]]
        returns_nextmonth = compute_monthly_returns(data_nextmonth)
        print("{0} Stock Returns:".format(month_names[i+1]))
        print(returns_nextmonth)
        wealth_full_mkt[i+1] = wealth_full_mkt[i]*(1 + np.sum(np.dot(mp_full, np.asmatrix(returns_nextmonth))))
        wealth_sec1_only[i+1] = wealth_sec1_only[i]*(1 + np.sum(np.dot(mp_sector_1, np.asmatrix(returns_nextmonth[symbols_sector_1]))))
        wealth_sec2_only[i+1] = wealth_sec2_only[i]*(1 + np.sum(np.dot(mp_sector_2, np.asmatrix(returns_nextmonth[symbols_sector_2]))))
    print("Full MP Portfolio Value:")
    print(wealth_full_mkt.T) # Display as row vector instead of column vector
    print("Tech MP Portfolio Value:")
    print(wealth_sec1_only.T)
    print("Health MP Portfolio Value:")
    print(wealth_sec2_only.T)
    print("Overall Yearly Returns: ")
    print(compute_monthly_returns(data_full))
    plt.figure()
    plt.plot(month_names, wealth_full_mkt, "k-", label = "Full Market")
    plt.plot(month_names, wealth_sec1_only, "b-", label = "Tech Stocks")
    plt.plot(month_names, wealth_sec2_only, "r-", label = "Health Stocks")
    # Since I'm plotting the discounted price the money market portfolio value is constant!
    plt.plot(month_names, np.ones((12, 1)), "g--", label = "Money Market")
    plt.title("Market Portfolio Value over 2023")
    plt.xlabel("Month")
    plt.ylabel("Expected discounted portfolio value ($)")
    plt.legend()
    plt.savefig("seq_MP_investment.png")
    return 0

def main():
    tech_symbols = ["AAPL", "CSCO", "INTC", "QCOM", "IBM"] # Selection from companies labeled with GICS "Information Technology"
    health_symbols = ["CI", "CVS", "DGX", "MRNA", "RMD"]
    SP_ex_returns = pd.read_csv("SP500_ex_returns.csv", index_col = 0, parse_dates=True, date_format = "%Y/%m/%d")
    SP_ex_returns.index = pd.to_datetime(SP_ex_returns.index, format='%Y-%m-%d') # Convert index to datetime
    # full_year_analysis(SP_ex_returns, tech_symbols, health_symbols)
    # sml_analysis(SP_ex_returns, tech_symbols, health_symbols)
    seq_mp_analysis(SP_ex_returns[tech_symbols + health_symbols], tech_symbols, health_symbols)
    return 0

if __name__ == "__main__":
    main()