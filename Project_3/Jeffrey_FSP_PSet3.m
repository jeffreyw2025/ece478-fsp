%% Jeffrey Wong | ECE-478 | PSet #3- BAPM

clear
close all
clc

%% Problem 1- Exact Simulation

% Part ci- Path Independence
% We note that since X_n is computed iteratively from future payouts
% V_{n+1}(H) and V_{n+1}(T), if the V_Ns are known and path independent, the 
% Delta_ns must be path independent, as only the future (path-independent)
% values are used in the iterative step, not the past values, which would
% correspond to path dependence.

% Part d- Test with European call

r = 0.05; u = 1.1; d = 1.01; N = 5;

p_rn = ((1+r)-d)/(u-d); % Risk-neutral p (Should be 4/9)
p_low = 0.2;
p_high = 0.7;

[S_rn, V_rn] = expected_discounted_payout(p_rn, 5, 1.1, 1.01, 0.05);
[S_low, V_low] = expected_discounted_payout(p_low, 5, 1.1, 1.01, 0.05);
[S_high, V_high] = expected_discounted_payout(p_high, 5, 1.1, 1.01, 0.05);

% We observe that using the risk-free interest rate gives the correct
% values of S_0 and V_0 (we know S_0 should be 1 because every discounted
% security in the BAPM is a martingale). If we use a probability that is
% higher than the risk-neutral probability, we observe a risk premium as
% indicated by an E(S_N) > 1, while we observe a lower return with a 
% p_low < p_rn

[V_0, Delta] = compute_replicating_portfolio(N, u, d, r);

% We observe that our V_0 obtained from the replicating portfolio matches
% what the discounted E_p(V~_n) for p = p_rn. There is no short selling
% involved (delta always >= 0), but we will need to borrow at the beginning
% (X_0 - Delta_0 = V_0 - Delta_0 = 0.0390 - 0.5352 < 0), and possibly more
% to buy more shares if the price goes up.

%% Problem 2- Monte Carlo Simulation

% Part a

M = 10;
[S_euro1, V_euro1] = monte_carlo(M, N, u, d, r, [], 'european');
% Even with M maximized at 32 there still seems to be noticeable variance
% in S and V, likely because we aren't sampling all the paths and sometimes
% it will repick a higher or lower value path. This is obviously worse for
% lower values of M.

% Part b

M = 1000;
N = 100; u = 1+5e-3; d = 1+1e-4; r = 1e-3;
[S_euro2, V_euro2] = monte_carlo(M, N, u, d, r, [], 'european');
p_rn = (1+r - d)/(u-d);
[~, V_euro2_ref] = expected_discounted_payout(p_rn, N, u, d, r);
% Thanks to a bit of wizardry I made the Monte carlo simulator really fast,
% so I was able to test 10^5 runs near instantly! Obviously with M = 1e5 we
% converge to almost exactly the correct answer. With 1e3 runs (and even 
% down to 150 runs) the stock price varies only slightly around 1, but the 
% payout experiences slightly more (relative) fluctuation.

% Part c

p_eff = 0.9*p_rn;
starter_paths_euro2 = double(rand(5, 10) < p_eff);
[S_10_euro2_ref, ~] = compute_MC_path_values(starter_paths_euro2, 10, u, d, r, 'european');
S_10_euro2_ref = S_10_euro2_ref(:,end); % Reference security prices for length 10 paths
% Compare computed S_10 to experimental S_10 from Monte Carlo
S_10_euro2_mc = zeros(5, 1); V_10_euro2_mc = zeros(5, 1); 
for i = 1:5
    [S_10_euro2_mc(i), V_10_euro2_mc(i)] = monte_carlo(M, N, u, d, r, starter_paths_euro2(i,:), 'european');
end

% Single path analysis

M_onepath = 100; % Try path 2 experiment 100 times
S_10_euro2_onepath = zeros(M_onepath, 1); V_10_euro2_onepath = zeros(M_onepath, 1); 
for j = 1:M_onepath
    [S_10_euro2_onepath(j), V_10_euro2_onepath(j)] = monte_carlo(M, N, u, d, r, starter_paths_euro2(2,:), 'european');
end
S_10_euro2_onepath_smean = mean(S_10_euro2_onepath);
S_10_euro2_onepath_svar = var(S_10_euro2_onepath);
V_10_euro2_onepath_smean = mean(V_10_euro2_onepath);
V_10_euro2_onepath_svar = var(V_10_euro2_onepath);

% The predicted and experimental S_10s largely aligned, with similar
% variance as was observed in part B. The observed V_10s also seemed to
% grow with increasing S_10, which makes sense. When analyzing the
% single-path case with 100 repeated experiments, the sample mean matched the
% expected price and the variance was on the order of 1e-7, which is very
% small. The one-path payout ended up slightly off relative to the first
% experimental V_10 but within 1 standard deviation, which makes some sense
% as the first iteration is itself subject to variance and is thus not
% necessarily a perfect reference. It thus seems that Monte Carlo works!

% Part d
[S_lookback, V_lookback] = monte_carlo(M, N, u, d, r, [], 'lookback');
starter_paths_lookback = double(rand(5, 10) < p_eff);
[S_10_lookback_ref, ~] = compute_MC_path_values(starter_paths_lookback, 10, u, d, r, 'lookback');
S_10_lookback_ref = S_10_lookback_ref(:,end); % Reference security prices for length 10 paths
S_10_lookback_mc = zeros(5, 1); V_10_lookback_mc = zeros(5, 1); 
for i = 1:5
    [S_10_lookback_mc(i), V_10_lookback_mc(i)] = monte_carlo(M, N, u, d, r, starter_paths_lookback(i,:), 'lookback');
end

% You could iterate through the coin tosses to evaluate the lookback
% option, or just take the max over the discounted prices! Proof is
% attached in the PDF.

% Single path analysis

S_10_lookback_onepath = zeros(M_onepath, 1); V_10_lookback_onepath = zeros(M_onepath, 1); 
for j = 1:M_onepath
    [S_10_lookback_onepath(j), V_10_lookback_onepath(j)] = monte_carlo(M, N, u, d, r, starter_paths_euro2(2,:), 'lookback');
end
S_10_lookback_onepath_smean = mean(S_10_lookback_onepath);
S_10_lookback_onepath_svar = var(S_10_lookback_onepath);
V_10_lookback_onepath_smean = mean(V_10_lookback_onepath);
V_10_lookback_onepath_svar = var(V_10_lookback_onepath);

% The Monte Carlo method seems to be working just as well for the lookback
% option as it is for the European call option! 

%% Function Definitions

%% Problem 1- Exact Simulation Functions

% European call payout for part d (K = (1+r)^N * S_0)
function V_N = payout(S_N, N, r)
    V_N = S_N - (1 + r).^N; % Note that S_0 assumed to be 1
    V_N = V_N .* (V_N > 0); % To ensure payout is bounded below at zero
end

% Part a- Compute expected discounted security price and option payout
function [Sp_V_N, Ep_V_N] = expected_discounted_payout(p, N, u, d, r)
    % Check for no arbitrage, abort if no arbitrage failed
    if(d > 1 + r | u < 1 + r)
        disp("Warning: Arbitrage detected, simulation aborted")
        return
    end
    prob_nheads = binopdf(0:N, N, p); % Gets the probability of getting n heads
    S_nheads = (u/d).^([0:N]) .* (d^N); % Computes price based on number of heads (S_n) = S_0 * u^n * d^(N-n) = (u/d)^n * d^N
    V_nheads = (1+r)^(-N) * payout(S_nheads, N, r);
    Sp_V_N = sum(prob_nheads .* ((1+r)^(-N) * S_nheads));
    Ep_V_N = sum(prob_nheads .* V_nheads);
end

% Part b- Compute one step of the replicating portfolio
function [V_n, delta_n] = replicating_portfolio_step(S_n, S_np1_H, S_np1_T, V_np1_H, V_np1_T, u, d, r)
    % Solve system of two wealth equations @ Heads & Tails
    result = [1+r, (u-(1+r)) * S_n; 1+r, (d-(1+r)) * S_n] \ [V_np1_H; V_np1_T];
    result = result.';
    V_n = result(1);
    delta_n = result(2);
end

% Part c- Compute full replicating portfoilo
function [V_0, Delta] = compute_replicating_portfolio(N, u, d, r)
    Delta = zeros(N); % There are N timesteps (0 <= n <= N-1), with timestep n having n+1 associated deltas
    S = zeros(N+1); % Matrix of security prices from n = 0 to N, given k heads and n-k tails
    V = zeros(N+1); % Matrix of payouts from n = 0 to N, given k heads and n-k tails
    S(N+1,:) = (u/d).^([0:N]) .* d.^(N);
    V(N+1,:) = payout(S(N+1,:), N, r);
    for i = N:-1:1 % Traverse backwards in time from N-1, note that index corresponds to time n + 1
        S(i,1:i) = (u/d).^([0:i-1]) .* d.^(i-1);
        for j = 1:i % Loop across # of heads @ n = i-1
            [v, delta] = replicating_portfolio_step(S(i,j), S(i+1,j+1), S(i+1,j), V(i+1,j+1), V(i+1,j), u, d, r);
            Delta(i,j) = delta;
            V(i,j) = v;
        end
    end
    V_0 = V(1,1);
end

%% Problem 2- Monte Carlo Simulation Functions

% Monte Carlo "wrapper" function to control the overall process
% M is # of generated paths, N is total path length, omega_n is an initial
% path (which may be empty) as a row vector, derivative_type is 
% "european" by default or "lookback"
function [E_S, E_V] = monte_carlo(M, N, u, d, r, omega_n, derivative_type)
    arguments
        M
        N
        u
        d
        r
        omega_n = []
        derivative_type = 'european'
    end
    % Check for no arbitrage, abort if no arbitrage failed
    if(d > 1 + r | u < 1 + r)
        disp("Warning: Arbitrage detected, simulation aborted")
        return
    end
    % Compute risk-neutral probability p~ (gives q~ implicitly), and form
    % matrix of coin tosses (we'll use 0 for tails and 1 for heads)
    p_rn = (1+r - d)/(u-d);
    paths = double(rand(M, N) < p_rn);

    n = 0; % n is effectively a "starting time" for us to calculate values off of.
    % Check if input array is empty (start from 0), then overwrite starting
    % flips with given flips from omega_n
    if(~isempty(omega_n))
        n = length(omega_n);
        paths(:, 1:length(omega_n)) = repmat(omega_n, M, 1);
    end
    [S_discounted, V_discounted] = compute_MC_path_values(paths, n, u, d, r, derivative_type);
    E_S = mean(S_discounted(:,end));
    E_V = mean(V_discounted);
end

% Computes the value of X(omega_N) for a specific path or set of paths of 
% length N omega_N and some specified input function X(omega)
function [S_discounted, V_discounted] = compute_MC_path_values(paths, n, u, d, r, derivative_type)
    arguments
        paths
        n
        u
        d
        r
        derivative_type = 'european'
    end
    M = size(paths, 1); % # rows = # paths
    N = size(paths, 2); % # cols = path length
    S_increments = u * (paths == 1) + d * (paths == 0); % Computes how much S changes at each step, either by u or d based on coin flip
    S = cumprod(S_increments, 2); % Get stock value at each step using a cumulative product
    discount_factor = repmat((1+r).^([1-n:N-n]), M, 1); % Starting time is n so discount by t-n
    S_discounted = S ./ discount_factor;
    V_discounted = 0;
    % Shamelessly stolen from uhlprocsim from Adaptive!
    if strcmpi(derivative_type(1), 'e') % European Call payout
        V_discounted = max(S_discounted(:, end) - 1,0);
    elseif strcmpi(derivative_type(1), 'l') % Lookback payout- See attached pdf for a proof!
        V_discounted = max(S_discounted,[],2) - S_discounted(:, end);
    else
        disp('Error: specify European or lookback');
        return
    end
end