%% Jeffrey Wong | ECE-478 | PSet #5- Time Series

clear
close all
clc

% For this assignment specifically, I will request that you check parts 2
% and 3 seperately from 4!

%% Problem 1- Heavy Tail Distributions
N = 1e6;
alpha = 0.544; % "Scaling" for Cauchy
gaussian_data = randn(1, N);
uniform_data = rand(1, N); % Use U(0,1) data to form Cauchy
cauchy_data = alpha * tan(pi*uniform_data);
t5_data = sqrt(3/5) *trnd(5,1,N); % Students-t w/ nu = 5
t10_data = sqrt(8/10) *trnd(10,1,N); % nu = 10

disp("The fraction of values in the normal distribution with absolute value more than 4 is " + mean(abs(gaussian_data) > 4));
disp("The fraction of values in the Cauchy distribution with absolute value more than 4 is " + mean(abs(cauchy_data) > 4));
disp("The fraction of values in the t-distribution (nu = 5) with absolute value more than 4 is " + mean(abs(t5_data) > 4));
disp("The fraction of values in the t-distribution (nu = 10) with absolute value more than 4 is " + mean(abs(t10_data) > 4));

figure
subplot(2,2,1)
plot(1:N, gaussian_data)
yline(4, "r-")
yline(-4, "r-")
title("Standard Gaussian Data")
subplot(2,2,2)
plot(1:N, cauchy_data)
yline(4, "r-")
yline(-4, "r-")
title("Cauchy Data")
subplot(2,2,3)
plot(1:N, t5_data)
yline(4, "r-")
yline(-4, "r-")
title("Students-t (\nu = 5) Data")
subplot(2,2,4)
plot(1:N, t10_data)
yline(4, "r-")
yline(-4, "r-")
title("Students-t (\nu = 10) Data")

%% Problem 2- AR and ARMA

% Part a + b
N = 250;
M = 10; % Max filter order
v_t = randn(1,N); % We will use the same underlying Gaussian white noise for both parts to allow for comparison
r_t_ARMA = filter([1 0.7 0.1], [1 -1.5 0.56], v_t);
gamma_m = xcorr(r_t_ARMA)/N;
% Note M_0(N) corresponds to m = 0 here
figure
stem(0:M, abs(gamma_m(N:N+M)/gamma_m(N)))
yline(0.2, "r--")
title("\gamma(m)/\gamma(0) for ARMA(2,2)")
C = toeplitz(gamma_m(N:N+M)); 
disp("Eigenvalues of C for ARMA(2,2): ")
disp(eig(C))

% Part c + d + e
[L,D] = better_ldl(C); % See my note below on how standard ldl sucks

% Only use correlations where lag >= 0
[ar_lev, F, P_m, k] = generate_FPEFs(gamma_m(N:end), M);
FCFT = F * C * F';
Linv = inv(L);
disp("AR(10) model from Levinson-Durbin")
disp(ar_lev)

% With the "better" ldl, it produced an exact match! The built-in ldl
% function produced weird results in comparison

% Part f
ar_ls = generate_LS_AR_model(r_t_ARMA, M);
disp("AR(10) model from Least Squares")
disp(ar_ls)
% The least squares process produces a reasonable approximation to the
% Levinson-Durbin AR model, with values deviating by at most 10%.

% Part g
% The high reflection coefficients corresponded with large decreases in
% prediction variance and low reflection coefficients with small decreases,
% as expected (P_{m+1} = P_m(1-|k_m|^2)). Generally, the first three
% coefficents tend to be significant (|k| > 0.2), with an overall falloff
% and occasional spike depending on what data generates.

% Part h
[v_0_AR2, rho_AR2] = compute_residuals(r_t_ARMA, fliplr(F(3,1:3)));
[v_0_AR5, rho_AR5] = compute_residuals(r_t_ARMA, fliplr(F(6,1:6)));
[v_0_AR10, rho_AR10] = compute_residuals(r_t_ARMA, fliplr(F(11,1:11)));
M_0 = 3; % Anticipated optimal AR model order
[v_0_ARopt, rho_ARopt] = compute_residuals(r_t_ARMA, fliplr(F(M_0 + 1,1:M_0 + 1)));
% It seems that in some cases none of the AR models are sufficient to work,
% but if it does usually it fails for M = 2 only, with M = 3 being sufficient.


%% Problem 3- ARIMA & First Difference

N = 250; M = 10;
% Part a+b (r_t)
r_t_ARIMA = filter([1 0.7 0.1], conv([1 -1.5 0.56], [1 -0.99]), v_t); % Add extra pole @ z = 0.99 to ARMA model
gamma_m_r = xcorr(r_t_ARIMA)/N;
% Note M_0(N) corresponds to m = 0 here
figure
stem(0:M, abs(gamma_m_r(N:N+M)/gamma_m_r(N)))
yline(0.2, "r--")
title("\gamma(m)/\gamma(0) for ARIMA(2,1,2)")
% The correlation barely decreases from M = 0 to 10 for ARIMA.
C_r = toeplitz(gamma_m_r(N:N+M)); 
disp("Eigenvalues of C_r for ARIMA(2,1,2): ")
disp(eig(C_r))
% Rather than not being positive definite, C_r happens to have very large
% eigenvalues! That's odd considering the rows should be strongly
% correlated and there should thus be a lot of redundancy! 
% Anyways, the unit root nonstantionarity clearly manifests
% here as the correlation dying out very slowly, probably due to the
% presence of the pole @ z = 0.99.

% Part c + d + e (r_t)
[L_r,D_r] = better_ldl(C_r);
[ar_lev_r, F_r, P_m_r, k_r] = generate_FPEFs(gamma_m_r(N:end), M);
FCFT_r = F_r * C_r * F_r';
Linv_r = inv(L_r);
disp("AR(10) model from Levinson-Durbin")
disp(ar_lev_r)
% At least we still have F = L^-1 and D = P_m!

% Part f
ar_ls_r = generate_LS_AR_model(r_t_ARIMA, M);
disp("AR(10) model from Least Squares")
disp(ar_ls_r)
% The least squares approximation is poor because everything is thrown off
% by the nonstationarity!

% Part g
% The first and second reflection coefficients are high while the others
% don't really have an impact. Despite this, AR(2) fails to work, as
% described below!)

% Part h
[v_0_AR2_r, rho_AR2_r] = compute_residuals(r_t_ARIMA, fliplr(F_r(3,1:3)));
[v_0_AR5_r, rho_AR5_r] = compute_residuals(r_t_ARIMA, fliplr(F_r(6,1:6)));
[v_0_AR10_r, rho_AR10_r] = compute_residuals(r_t_ARIMA, fliplr(F_r(11,1:11)));
% It is no surprise that since the correlation doesn't die out with ARIMA,
% none of the AR models are even close to whitening the residuals!


% Part a+b (s_t)
N = 249;
s_t_ARIMA = diff(r_t_ARIMA);
gamma_m_s = xcorr(s_t_ARIMA)/N;
figure
stem(0:M, abs(gamma_m_s(N:N+M)/gamma_m_s(N)))
yline(0.2, "r--")
title("\gamma(m)/\gamma(0) for first difference of ARIMA(2,1,2)")
C_s = toeplitz(gamma_m_s(N:N+M)); 
disp("Eigenvalues of C_s for ARIMA(2,1,2): ")
disp(eig(C_s))

% Part c + d + e (s_t)
[L_s,D_s] = better_ldl(C_s);
[ar_lev_s, F_s, P_m_s, k_s] = generate_FPEFs(gamma_m_s(N:end), M);
FCFT_s = F_s * C_s * F_s';
Linv_s = inv(L_s);
disp("AR(10) model from Levinson-Durbin")
disp(ar_lev_s)

% Part f
ar_ls_s = generate_LS_AR_model(s_t_ARIMA, M);
disp("AR(10) model from Least Squares")
disp(ar_ls_s)
% Since the model did not have an exact nonstationarity but we cancel it out
% as if it were, the least-squares approximation for the difference is
% still a bit off!

% Part g
% The reflection coefficients are closer in behavior to those in Problem 2
% than in the first part of problem 3!

% Part h
[v_0_AR2_s, rho_AR2_s] = compute_residuals(s_t_ARIMA, fliplr(F_s(3,1:3)));
[v_0_AR5_s, rho_AR5_s] = compute_residuals(s_t_ARIMA, fliplr(F_s(6,1:6)));
[v_0_AR10_s, rho_AR10_s] = compute_residuals(s_t_ARIMA, fliplr(F_s(11,1:11)));
% As with Problem 2, it's likely a 3rd order AR process should be
% sufficient to model the behavior of the first difference!

%% Problem 4- Students-T ARMA/ARIMA
N = 250; M = 10;
v_t = trnd(5,1,N); % Note that the t-distribution has a "heavy tail" so more outliers
r_t_ARMA = filter([1 0.7 0.1], [1 -1.5 0.56], v_t);
gamma_m = xcorr(r_t_ARMA)/N;
figure
stem(0:M, abs(gamma_m(N:N+M)/gamma_m(N)))
yline(0.2, "r--")
title("\gamma(m)/\gamma(0) for ARMA(2,2)")
C = toeplitz(gamma_m(N:N+M)); 
disp("Eigenvalues of C for ARMA(2,2): ")
disp(eig(C))

% Part c + d + e
[L,D] = better_ldl(C); 
[ar_lev, F, P_m, k] = generate_FPEFs(gamma_m(N:end), M);
FCFT = F * C * F';
Linv = inv(L);
disp("AR(10) model from Levinson-Durbin")
disp(ar_lev)

% Part f
ar_ls = generate_LS_AR_model(r_t_ARMA, M);
disp("AR(10) model from Least Squares")
disp(ar_ls)

% Part h
[v_0_AR2, rho_AR2] = compute_residuals(r_t_ARMA, fliplr(F(3,1:3)));
[v_0_AR5, rho_AR5] = compute_residuals(r_t_ARMA, fliplr(F(6,1:6)));
[v_0_AR10, rho_AR10] = compute_residuals(r_t_ARMA, fliplr(F(11,1:11)));
M_0 = 6; % The optimal filter order goes up overall compared to Question 2
[v_0_ARopt, rho_ARopt] = compute_residuals(r_t_ARMA, fliplr(F(M_0 + 1,1:M_0 + 1)));

% Part a+b (r_t)
r_t_ARIMA = filter([1 0.7 0.1], conv([1 -1.5 0.56], [1 -0.99]), v_t); % Add extra pole @ z = 0.99 to ARMA model
gamma_m_r = xcorr(r_t_ARIMA)/N;
figure
stem(0:M, abs(gamma_m_r(N:N+M)/gamma_m_r(N)))
yline(0.2, "r--")
title("\gamma(m)/\gamma(0) for ARIMA(2,1,2)")
C_r = toeplitz(gamma_m_r(N:N+M)); 
disp("Eigenvalues of C_r for ARIMA(2,1,2): ")
disp(eig(C_r))

% Part c + d + e (r_t)
[L_r,D_r] = better_ldl(C_r);
[ar_lev_r, F_r, P_m_r, k_r] = generate_FPEFs(gamma_m_r(N:end), M);
FCFT_r = F_r * C_r * F_r';
Linv_r = inv(L_r);
disp("AR(10) model from Levinson-Durbin")
disp(ar_lev_r)

% Part f
ar_ls_r = generate_LS_AR_model(r_t_ARIMA, M);
disp("AR(10) model from Least Squares")
disp(ar_ls_r)

% Part h
[v_0_AR2_r, rho_AR2_r] = compute_residuals(r_t_ARIMA, fliplr(F_r(3,1:3)));
[v_0_AR5_r, rho_AR5_r] = compute_residuals(r_t_ARIMA, fliplr(F_r(6,1:6)));
[v_0_AR10_r, rho_AR10_r] = compute_residuals(r_t_ARIMA, fliplr(F_r(11,1:11)));

% Part a+b (s_t)
N = 249;
s_t_ARIMA = diff(r_t_ARIMA);
gamma_m_s = xcorr(s_t_ARIMA)/N;
figure
stem(0:M, abs(gamma_m_s(N:N+M)/gamma_m_s(N)))
yline(0.2, "r--")
title("\gamma(m)/\gamma(0) for first difference of ARIMA(2,1,2)")
C_s = toeplitz(gamma_m_s(N:N+M)); 
disp("Eigenvalues of C_s for ARIMA(2,1,2): ")
disp(eig(C_s))

% Part c + d + e (s_t)
[L_s,D_s] = better_ldl(C_s);
[ar_lev_s, F_s, P_m_s, k_s] = generate_FPEFs(gamma_m_s(N:end), M);
FCFT_s = F_s * C_s * F_s';
Linv_s = inv(L_s);
disp("AR(10) model from Levinson-Durbin")
disp(ar_lev_s)

% Part f
ar_ls_s = generate_LS_AR_model(s_t_ARIMA, M);
disp("AR(10) model from Least Squares")
disp(ar_ls_s)

% Part h
[v_0_AR2_s, rho_AR2_s] = compute_residuals(s_t_ARIMA, fliplr(F_s(3,1:3)));
[v_0_AR5_s, rho_AR5_s] = compute_residuals(s_t_ARIMA, fliplr(F_s(6,1:6)));
[v_0_AR10_s, rho_AR10_s] = compute_residuals(s_t_ARIMA, fliplr(F_s(11,1:11)));

% Comments:
% The heavy tail of the Students-t distribution doesn't seem to be causing
% enough outliers to break numerical stability- the eigenvalue spread only
% slightly increased with the largest eigenvalues going up by a factor of
% 2-3 and the smallest eigenvalues also decreasing. It does seem like the
% optimal order for the AR models goes up as more terms are needed to
% mitigate the effect of outliers, but otherwise everything lines up with
% the previous sections' results!

%% Problem 5- ARCH/GARCH

% Part a
N = 250;
omega = 0.5; alpha = 0.6; beta = 0.3; % For stability we need alpha + beta < 1
z_t = randn(1, N);
sigma_squared = zeros(1, N);
epsilon = zeros(1, N);
for i = 1:N-1
    epsilon(i) = sqrt(sigma_squared(i)) * z_t(i);
    sigma_squared(i+1) = omega + alpha * epsilon(i).^2 + beta * sigma_squared(i);
end
epsilon(1,end) = sqrt(sigma_squared(1,end)) * z_t(1,end);
sigma = sqrt(sigma_squared);
figure
subplot(2,1,1)
hold on
stem(1:N, epsilon, "k")
plot(1:N, epsilon + sigma, "b--")
plot(1:N, epsilon - sigma, "r--")
title("Return and Volatility envelope of GARCH(1,1) for Gaussian z")
xlabel("t")
ylabel("r(t)")
subplot(2,1,2)
plot(1:N, sigma(1:N))
title("Volatility of GARCH(1,1) for Gaussian z")
xlabel("t")
ylabel("\sigma(t)")

GARCH_gauss = garch(1,1);
est_GARCH_gauss = estimate(GARCH_gauss, epsilon');
disp(est_GARCH_gauss)
ARCH_gauss = garch(0,2);
est_ARCH_gauss = estimate(ARCH_gauss, epsilon');
disp(est_ARCH_gauss)
[V_ARCH_gauss, ~] = filter(est_ARCH_gauss, z_t);

figure
subplot(2,1,1)
hold on
stem(1:N, epsilon, "k")
plot(1:N, epsilon + sqrt(V_ARCH_gauss), "b--")
plot(1:N, epsilon - sqrt(V_ARCH_gauss), "r--")
title("Return and Volatility envelope of estimated ARCH(2) for Gaussian z")
xlabel("t")
ylabel("r(t)")
subplot(2,1,2)
plot(1:N, sqrt(V_ARCH_gauss))
title("Volatility of estimated ARCH(2) for Gaussian z")
xlabel("t")
ylabel("\sigma(t)")

% Now with Students-t
z_t = sqrt(3/5) * trnd(5, 1, N);
sigma_squared = zeros(1, N);
epsilon = zeros(1, N);
for i = 1:N-1
    epsilon(i) = sqrt(sigma_squared(i)) * z_t(i);
    sigma_squared(i+1) = omega + alpha * epsilon(i).^2 + beta * sigma_squared(i);
end
epsilon(1,end) = sqrt(sigma_squared(1,end)) * z_t(1,end);
sigma = sqrt(sigma_squared);
figure
subplot(2,1,1)
hold on
stem(1:N, epsilon, "k")
plot(1:N, epsilon + sigma, "b--")
plot(1:N, epsilon - sigma, "r--")
title("Return and Volatility envelope of GARCH(1,1) for Students-t(\nu = 5) z")
xlabel("t")
ylabel("r(t)")
subplot(2,1,2)
plot(1:N, sigma)
title("Volatility of GARCH(1,1) for Students-t(\nu = 5) z")
xlabel("t")
ylabel("\sigma(t)")
xlim([2 251])

GARCH_t = garch(1,1);
est_GARCH_t = estimate(GARCH_t, epsilon');
disp(est_GARCH_t)
ARCH_t = garch(0,2);
est_ARCH_t = estimate(ARCH_t, epsilon');
disp(est_ARCH_t)
% Note that the "volatility" returned by filter is a conditional *variance*
[V_ARCH_t, ~] = filter(est_ARCH_gauss, z_t);

figure
subplot(2,1,1)
hold on
% We want to compare how ARCH envelops the actual returns, modeled by r_t = epsilon_t
stem(1:N, epsilon, "k")
plot(1:N, epsilon + sqrt(V_ARCH_t), "b--")
plot(1:N, epsilon - sqrt(V_ARCH_t), "r--")
title("Return and Volatility envelope of estimated ARCH(2) for Students-t(\nu = 5) z")
xlabel("t")
ylabel("r(t)")
subplot(2,1,2)
plot(1:N, sqrt(V_ARCH_t))
title("Volatility of estimated ARCH(2) for Students-t(\nu = 5) z")
xlabel("t")
ylabel("\sigma(t)")

% Unfortunately, estimating the GARCH model does not seem to return
% something particularly close to the coefficients even for the Gaussian,
% and is worse with the Students-t, likely because the number of values we
% are studying is not especially large. This happens even in the MATLAB
% documentation for estimate's own example with beta_hat becoming 0.46 
% from 0.5 and alpha_hat becoming 0.26 from 0.2. ARCH(2) does a mediocre job 
% capturing the returns in an envelope, and does not readily match the coefficients.

%% Part b

% Read data from CSV, ignoring header and date index
data = readmatrix("SP500_log_returns.csv", 'Range', [2 2]);
N = length(data);
% Data columns should be, in order, Apple, IBM, and ^SPX

% Initialize and fit models
disp("Fitting ARCH/GARCH for AAPL")
GARCH_stock1 = garch(1,1);
est_GARCH_stock1 = estimate(GARCH_stock1, data(:,1));
ARCH2_stock1 = garch(0,2);
est_ARCH2_stock1 = estimate(ARCH2_stock1, data(:,1));
V_GARCH_1 = infer(est_GARCH_stock1, data(:,1));
V_ARCH2_1 = infer(est_ARCH2_stock1, data(:,1));

disp("Fitting ARCH/GARCH for IBM")
GARCH_stock2 = garch(1,1);
est_GARCH_stock2 = estimate(GARCH_stock2, data(:,2));
ARCH2_stock2 = garch(0,2);
est_ARCH2_stock2 = estimate(ARCH2_stock2, data(:,2));
V_GARCH_2 = infer(est_GARCH_stock2, data(:,2));
V_ARCH2_2 = infer(est_ARCH2_stock2, data(:,2));

disp("Fitting ARCH/GARCH for ^SPX")
GARCH_index = garch(1,1);
est_GARCH_index = estimate(GARCH_index, data(:,3));
ARCH2_index = garch(0,2);
est_ARCH2_index = estimate(ARCH2_index, data(:,3));
V_GARCH_index = infer(est_GARCH_index, data(:,3));
V_ARCH2_index = infer(est_ARCH2_index, data(:,3));

% Plot approximate fit for stocks and index

figure
subplot(1,2,1)
hold on
stem(1:N, data(:,1), "k")
plot(1:N, data(:,1) + sqrt(V_GARCH_1), "b--")
plot(1:N, data(:,1) - sqrt(V_GARCH_1), "r--")
title("Return and Volatility envelope of GARCH(1,1) for AAPL")
xlabel("t")
ylabel("r(t)")
xlim([1 N])
subplot(1,2,2)
hold on
stem(1:N, data(:,1), "k")
plot(1:N, data(:,1) + sqrt(V_ARCH2_1), "b--")
plot(1:N, data(:,1) - sqrt(V_ARCH2_1), "r--")
title("Return and Volatility envelope of ARCH(2) for AAPL")
xlabel("t")
ylabel("r(t)")
xlim([1 N])

figure
subplot(1,2,1)
hold on
stem(1:N, data(:,2), "k")
plot(1:N, data(:,2) + sqrt(V_GARCH_2), "b--")
plot(1:N, data(:,2) - sqrt(V_GARCH_2), "r--")
title("Return and Volatility envelope of GARCH(1,1) for IBM")
xlabel("t")
ylabel("r(t)")
subplot(1,2,2)
hold on
stem(1:N, data(:,2), "k")
plot(1:N, data(:,2) + sqrt(V_ARCH2_2), "b--")
plot(1:N, data(:,2) - sqrt(V_ARCH2_2), "r--")
title("Return and Volatility envelope of ARCH(2) for IBM")
xlabel("t")
ylabel("r(t)")
xlim([1 N])

figure
subplot(1,2,1)
hold on
stem(1:N, data(:,3), "k")
plot(1:N, data(:,3) + sqrt(V_GARCH_index), "b--")
plot(1:N, data(:,3) - sqrt(V_GARCH_index), "r--")
title("Return and Volatility envelope of GARCH(1,1) for ^SPX")
xlabel("t")
ylabel("r(t)")
subplot(1,2,2)
hold on
stem(1:N, data(:,3), "k")
plot(1:N, data(:,3) + sqrt(V_ARCH2_2), "b--")
plot(1:N, data(:,3) - sqrt(V_ARCH2_2), "r--")
title("Return and Volatility envelope of ARCH(2) for ^SPX")
xlabel("t")
ylabel("r(t)")
xlim([1 N])

% From a mix of the p-values and visual inspection, it seems that the GARCH
% models are slightly better fits than the ARCH(2) models for all three
% sets of data, and Apple and the overall S&P 500 exhibit stronger
% heteroskadasticity than IBM does, which lines up with what we observe
% from the plots of the autocorrelation of the squared returns!

%% Function Definitions

% The ldl estimate given by MATLAB's LDL function is really crappy (L
% doesn't even have all 1s on the diagonal!), so I'm making my own version!
function [L,D] = better_ldl(A)
    L_chol = chol(A, 'lower');
    D = diag(diag(L_chol.^2));
    L = L_chol * D^(-1/2);
end

% Computes AR filters up to order M, as well as errors and reflection
% coefficents, using the Levinson-Durbin recursion
function [ar, F, P_m, k] = generate_FPEFs(gamma_m, M)
    F = eye(M+1);
    [ar, ~, k] = levinson(gamma_m, M);
    P_m = zeros(1,M+1);
    for i = 0:M
        [a, e_m, ~] = levinson(gamma_m, i);
        F(i+1, 1:i+1) = fliplr(a);
        P_m(1, i+1) = e_m;
    end
    P_m = diag(P_m); % Convert to diagonal matrix for easy comparison
end

% Computes an order M AR model using a min-norm least-square solution.
function ar = generate_LS_AR_model(r, M)
    N = length(r);
    % Form time series data into matrixx
    A = toeplitz(r);
    A = A(M:N-1, 1:M);
    % Augment A matrix to account for potential non-zero mean
    A = [ones(N-M, 1) A];
    y = r(M+1:N)';
    w_0 = A \ y; % Left divide performs pseudo-inverses as well as standard inverses!
    ar = [1 -w_0(2:end,1)'];
end

% Computes residuals from a time series given an AR model with M+1
% coefficients, as well as the time-lag correlation between residuals for 
% 0 <= m <= 20
function [v_0, rho] = compute_residuals(r, ar)
    v_0 = filter(ar, 1, r);
    v_0 = v_0(length(ar):end); % Discard the first M samples to avoid violating causality
    N = length(v_0);
    rho = xcorr(v_0)/N;
    rho = abs(rho / rho(N));
    figure
    stem(0:20, rho(N:N+20))
    title("\rho(m) for AR("+(length(ar)-1)+")")
    yline(0.2, "r--")
end