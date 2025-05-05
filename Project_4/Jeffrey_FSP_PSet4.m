%% Jeffrey Wong | ECE-478 | PSet #4- Stochastic Calculus

clear
close all
clc

%% Problem 2.1- Geometric Brownian Motion

% Part b- See attached PDF

% Part c
dt = 1/260;
N = 520; % T_final = T = Ndt = 2
alpha = 0.1;
r = 0.05;
K_call = exp(alpha * N * dt);
num_paths = 100;

[S_lvar, invalid_paths_lvar] = generate_Brownian_motion(alpha, 0.05, dt, N, num_paths);
[S_mvar, invalid_paths_mvar] = generate_Brownian_motion(alpha, 0.1, dt, N, num_paths);
[S_hvar, invalid_paths_hvar] = generate_Brownian_motion(alpha, 0.3, dt, N, num_paths);

figure
hold on
for i = 1:10
    plot(linspace(0, 2, 521), S_lvar(i,:))
end
xlabel("Time (yr)")
ylabel("Stock value ($)")
title("Sample Geometric Brownian Motion trajectories, \sigma = 0.05")

% Part d
disp("Expected E(S[N/2]): " + exp(alpha))
disp("Expected E(S[N]): " + exp(2 * alpha))
disp("E(S[N/2]) when sigma = 0.05: " + mean(S_lvar(:,N/2+1)))
disp("E(S[N]) when sigma = 0.05: = " + mean(S_lvar(:,N+1)))
disp("E(S[N/2]) when sigma = 0.1: " + mean(S_mvar(:,N/2+1)))
disp("E(S[N]) when sigma = 0.1: = " + mean(S_mvar(:,N+1)))
disp("E(S[N/2]) when sigma = 0.3: " + mean(S_hvar(:,N/2+1)))
disp("E(S[N]) when sigma = 0.3: = " + mean(S_hvar(:,N+1)))
% In practice, since we are taking a relatively limited number of paths,
% which can either collapse to zero or increase wildly, it is unlikely for
% the experimental values to match the theoretical values. In particular,
% for sigma = 0.3, the volatility becomes so large that all of the paths
% decrease to zero at some point, leading to a zero expected value even
% though that is far below what we would get from the raw expectation.

% Part e

% Note that n = N/2 corresponds to t = 1, giving us tau = T - t = 1
halfN_prices = linspace(0,2.5,251);
V_halfN_lvar = compute_BSM_value(1, halfN_prices, K_call, r, 0.05);
V_halfN_mvar = compute_BSM_value(1, halfN_prices, K_call, r, 0.1);
V_halfN_hvar = compute_BSM_value(1, halfN_prices, K_call, r, 0.3);

figure
hold on
plot(halfN_prices, V_halfN_lvar, 'b', 'DisplayName', "\sigma = 0.05")
plot(halfN_prices, V_halfN_mvar, 'g', 'DisplayName', "\sigma = 0.1")
plot(halfN_prices, V_halfN_hvar, 'r', 'DisplayName', "\sigma = 0.3")
xlabel("S[N/2] ($)")
ylabel("V[N/2] ($)")
title("European Call Price vs Security Value at n = N/2")
legend

% Part f

theoretical_V_halfN_lvar = compute_BSM_value(1, S_lvar(1:10, N/2+1), K_call, r, 0.05);
experimental_V_halfN_lvar = zeros(10, 1);
theoretical_V_halfN_mvar = compute_BSM_value(1, S_mvar(1:10, N/2+1), K_call, r, 0.1);
experimental_V_halfN_mvar = zeros(10, 1);
theoretical_V_halfN_hvar = compute_BSM_value(1, S_hvar(1:10, N/2+1), K_call, r, 0.3);
experimental_V_halfN_hvar = zeros(10, 1);
for k = 1:10
    S_k_lvar = BM_riskneutral_extension(r, 0.05, dt, N/2, 1000, S_lvar(k, N/2+1));
    experimental_V_halfN_lvar(k,:) = mean((S_k_lvar(:, end) - K_call) .* (S_k_lvar(:, end) > K_call));
    S_k_mvar = BM_riskneutral_extension(r, 0.1, dt, N/2, 1000, S_mvar(k, N/2+1));
    experimental_V_halfN_mvar(k,:) = mean((S_k_mvar(:, end) - K_call) .* (S_k_mvar(:, end) > K_call));
    S_k_hvar = BM_riskneutral_extension(r, 0.3, dt, N/2, 1000, S_hvar(k, N/2+1));
    experimental_V_halfN_hvar(k,:) = mean((S_k_hvar(:, end) - K_call) .* (S_k_hvar(:, end) > K_call));
end

disp("Theoretical vs. Experimental Call Price, sigma = 0.05")
disp(table(S_lvar(1:10, N/2+1), theoretical_V_halfN_lvar, experimental_V_halfN_lvar, 'VariableNames', ["S[N/2]", "Theoretical V[N/2]", "Experimental V[N/2]"]))
disp("Theoretical vs. Experimental Call Price, sigma = 0.1")
disp(table(S_mvar(1:10, N/2+1), theoretical_V_halfN_mvar, experimental_V_halfN_mvar, 'VariableNames', ["S[N/2]", "Theoretical V[N/2]", "Experimental V[N/2]"]))
disp("Theoretical vs. Experimental Call Price, sigma = 0.3")
disp(table(S_hvar(1:10, N/2+1), theoretical_V_halfN_hvar, experimental_V_halfN_hvar, 'VariableNames', ["S[N/2]", "Theoretical V[N/2]", "Experimental V[N/2]"]))

% It seems that for high prices, the theoretical and experimental 
% models largely agree, but for medium-low prices (below 1),
% experimental values tend to be a lot higher than predicted by BSM. This
% likely occurs because during the Monte Carlo simulation there are a few
% paths that grow quickly in value and produce a large payout, while those
% that decrease or stagnate are all lumped together as zero-payout. Thus,
% the experimental means are dominated by outliers.

%% Problem 2.2- CIR Interest Rate Model

% Part a- See attached PDF

% Part b
beta = 1;
alpha = 0.1;
r = 0.05; % Initial Interest Rate
sigma = 0.1;
num_paths = 1000;
T = 10; % End time
dt = 0.01; % Time increment

% Try out number of valid paths, save one set of Rs for computing expected
% mean and variance of R at various times
R_direct = 0;
vp_count = zeros(1, 100);
for i = 1:100
    [R_direct, valid_paths] = generate_direct_CIR_rates(beta, alpha, r, sigma, num_paths, dt, T/dt);
    vp_count(i) = valid_paths;
end
disp("Average # valid paths: " + mean(vp_count))
disp("Stdev # valid paths: " + std(vp_count))

% Part c
X_log = generate_log_CIR_rates(beta, alpha, r, sigma, num_paths, dt, T/dt);
R_log = exp(X_log);
figure
hold on
for j = 1:10
    plot(0:dt:T, R_log(j,:))
end
xlabel("Time (s)")
ylabel("Interest Rate")
title("Simulated Interest Rates from CIR Model")

% Part d

% Shorthand for exponential beta series
exp_betat = exp(0:-1:-10 * beta);
exp_beta2t = exp(0:-2:-20 * beta);
theoretical_E_R = r .* exp_betat + (alpha / beta) .* (1 - exp_betat);
exp_E_R_direct = mean(R_direct);
exp_E_R_log = mean(R_log, "omitnan");
% Note that values can swing wildly positive or negative in the log model, 
% leading to values of NaN or very large interest rates, which will lead to
% some spikes in value since we are ignoring NaNs as indicated by the omitnan flag.
% I'm assuming this divergence arises from discretization error.
% Actually, I double checked and the drift term associated with dt will
% always be negative from the equation. This will cause X to just decrease
% over time. No wonder it keeps collapsing to 0.

figure
hold on
plot(0:10, theoretical_E_R, 'b-','DisplayName', "Theoretical Expectation")
plot(0:10, exp_E_R_direct(1:100:1001), 'ro', 'DisplayName', 'Experimental Expectation using Direct Simulation of R')
plot(0:10, exp_E_R_log(1:100:1001), 'g*', 'DisplayName', 'Experimental Expectation using Log Simulation of R')
yline(alpha / beta, 'k--','DisplayName', "Theoretical Asymptotic Expectation")
title("Theoretical and Experimental Expected Interest Rate for CIR Model")
xlabel("Time (s)")
ylabel("Interest Rate")
legend

% Part e
theoretical_var_R = (sigma^2 * r / beta) .* (exp_betat - exp_beta2t) + (0.5*alpha*sigma^2/beta^2) .* (1 - 2*exp_betat + exp_beta2t);
exp_var_R_direct = var(R_direct);
exp_var_R_log = var(R_log, "omitnan");

figure
hold on
plot(0:10, theoretical_var_R, 'b-','DisplayName', "Theoretical Variance")
plot(0:10, exp_var_R_direct(1:100:1001), 'ro', 'DisplayName', 'Experimental Variance using Direct Simulation of R')
plot(0:10, exp_var_R_log(1:100:1001), 'g*', 'DisplayName', 'Experimental Variance using Log Simulation of R')
yline(0.5*alpha*sigma^2/beta^2, 'k--','DisplayName', "Theoretical Asymptotic Variance")
title("Theoretical and Experimental Variance for CIR Model")
xlabel("Time (s)")
ylabel("Variance")
legend

%% Problem 2.3- Jump-Diffusion Process

% The process is governed by SDE dX = alpha * Xdt + sigma * XdW + dJ
alpha = 0.1;
sigma = 0.2;
lambda = 3;
dt = 0.005;
T = 2;
N = T/dt;
num_paths = 5;
X = ones(num_paths, N + 1);
dX = zeros(num_paths, N);

% Precompute increments dW and dJ
dW = sqrt(dt) * randn(num_paths, N);
jump_points = rand(num_paths, N) < (lambda * dt);
% Students-t with 3 dof has variance 3/(3-2) = 3, so need to divide by sqrt(3) to get variance of 1
jump_heights = sqrt(1/3) * trnd(3, num_paths, N);
dJ = jump_heights .* jump_points;
% Simulate evolution over time in parallel
for i = 1:N
    dX(:,i) = alpha .* X(:,i) .* dt + sigma .* X(:,i) .* dW(:,i) + dJ(:,i);
    X(:, i+1) = X(:,i) + dX(:,i);
end

figure
hold on
plot(0:dt:T, X(1,:), 'DisplayName',"Run 1")
plot(0:dt:T, X(2,:), 'DisplayName',"Run 2")
plot(0:dt:T, X(3,:), 'DisplayName',"Run 3")
title("Jump-Diffusion Process from t = 0 to t = 2")
xlabel("Time (s)")
ylabel("Value")
legend

%% Function Definitions

% Generates geometric Brownian motion with drift alpha and volatility sigma
function [S, invalid_paths] = generate_Brownian_motion(alpha, sigma, dt, N, num_paths)
    % The process is governed by SDE dS = alpha * Sdt + sigma * SdW
    S = zeros(num_paths, N+1);
    S(:,1) = 1; % S(0) = 1 for each path
    dS = zeros(num_paths, N);
    invalid_paths = 0;
    for i = 1:num_paths
        % Generate paths until all values are strictly positive
        while sum((S(i,:) <= eps)) > 0
            dW = randn(1, N);
            for j = 1:N
                dS(i,j) = alpha * S(i,j) * dt + sigma * S(i,j) * dW(j);
                S(i,j+1) = S(i,j) + dS(i,j);
                if S(i,j+1) < eps
                    disp("Invalid path generated! (Sigma = " + sigma +")")
                    invalid_paths = invalid_paths + 1;
                    break
                end
            end
        end
    end
end

function S = BM_riskneutral_extension(r, sigma, dt, N, num_paths, initial_value)
    S = zeros(num_paths, N + 1);
    S(:,1) = initial_value;
    % The process is governed by SDE dS = r * Sdt + sigma * Sd\~W
    dS = zeros(num_paths, N);
    for i = 1:num_paths
        % Generate paths until all values are strictly positive
        while sum((S(i,:) <= eps)) > 0
            dW = randn(1, N);
            for j = 1:N
                dS(i,j) = r * S(i,j) * dt + sigma * S(i,j) * dW(j);
                S(i,j+1) = S(i,j) + dS(i,j);
                if S(i,j+1) < eps
                    % Regenerate path if negative value encountered
                    break
                end
            end
        end
    end
end

% Computes price of a European call based on the Black-Sholes-Merton model
function c = compute_BSM_value(tau, S, K, r, sigma)
    % Assume log in BSM computation is natural log
    d_plus = (log(S./K) + (r + 0.5 * sigma^2) * tau)/(sigma * sqrt(tau));
    d_minus = (log(S./K) + (r - 0.5 * sigma^2) * tau)/(sigma * sqrt(tau));
    c = S .* normcdf(d_plus) - K * exp(-r*tau) * normcdf(d_minus);
end

% Simulates discretized CIR model, generating R directly
function [result, valid_paths] = generate_direct_CIR_rates(beta, alpha, r, sigma, num_paths, dt, N)
    % The process is governed by SDE dR = (alpha - beta*R)dt + sigma * sqrt(R)dW
    R = r * ones(num_paths, N + 1);
    dR = zeros(num_paths, N);
    dW = sqrt(dt) * randn(num_paths, N);
    % Simulate evolution over time in parallel
    for i = 1:N
        dR(:, i) = (alpha - beta .* R(:,i)) * dt + sigma * sqrt(R(:,i)) .* dW(:, i);
        R(:, i+1) = R(:, i) + dR(:, i);
    end
    % Check for paths with non-positive values, save only "valid" paths
    result = zeros(num_paths, N+1);
    invalid_paths = 0;
    valid_paths = 0;
    for j = 1:num_paths
        if sum(R(j, :) <= 0) > 0
            invalid_paths = invalid_paths + 1;
        else
            valid_paths = valid_paths + 1;
            result(valid_paths, :) = R(j, :);
        end
    end
    if invalid_paths > 0
        disp("Number of invalid paths generated: " + invalid_paths)
    end
    if valid_paths == 0
        disp("Error: No valid paths generated!")
        return
    end
    result = result(1:valid_paths, :);
end

% Simulates discretized CIR model, generating R by simulating X = ln(R) or R = e^X
function X = generate_log_CIR_rates(beta, alpha, r, sigma, num_paths, dt, N)
    % The process is governed by SDE dX = (alpha/R - sigma^2 / 2R - beta)dt+ (sigma/sqrt(R))dW
    X = log(r) * ones(num_paths, N + 1);
    dX = zeros(num_paths, N);
    dW = sqrt(dt) * randn(num_paths, N);
    for i = 1:N
        dX(:, i) = ((alpha - sigma^2/2) .* exp(-X(:,i)) - beta) .* dt + (sigma .* sqrt(exp(-X(:,i)))) .* dW(:,i);
        X(:, i+1) = X(:, i) + dX(:, i);
    end
end