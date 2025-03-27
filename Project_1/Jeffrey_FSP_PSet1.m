%% Jeffrey Wong | ECE-478 | PSet #1

clear
close all
clc

%% Problem 1- Be a Financial Engineer!

% Variable Definitions

S_T = linspace(0,100,10001);

% Part a - Basic Options
K = 50;
long_call = european_call(S_T, K);
long_put = european_put(S_T, K);
short_call = -european_call(S_T, K);
short_put = -european_put(S_T, K);

% Note that the loss from short calls is unbounded- security can grow
% arbitrarily large in value! (In theory, at least)

plot_payout(S_T, long_call, K, "K = " + K, [-10 60], "Long Call")
plot_payout(S_T, long_put, K, "K = " + K, [-10 60], "Long Put")
plot_payout(S_T, short_call, K, "K = " + K, [-60 10], "Short Call")
plot_payout(S_T, short_put, K, "K = " + K, [-60 10], "Short Put")

% Part b - Straddle
% See attached file for derivation that V(S_T) = |S_T - K|
K = 45; % For a slightly off-center graph
straddle = european_call(S_T, K) + european_put(S_T, K);
plot_payout(S_T, straddle, K, "K = " + K, [-10 60], "Straddle")

% Part c - Call-put spead
K_1 = 22;
K_2 = 68;
call_put_spread = european_call(S_T, K_1) - european_call(S_T, K_2);
plot_payout(S_T, call_put_spread, [K_1 K_2], {"K_1 = " + K_1, "K_2 = " + K_2}, [-10 60], "Call-Put Spread")

% Part d - Butterfly
K_1 = 20; % Nicer values that can be subdivided into halves and thirds nicely!
K_2 = 80;
[butterfly_one_third, K_star_one_third] = butterfly(S_T, K_1, K_2, 1/3);
[butterfly_one_half, K_star_one_half] = butterfly(S_T, K_1, K_2, 1/2);
[butterfly_two_thirds, K_star_two_thirds] = butterfly(S_T, K_1, K_2, 2/3);
plot_payout(S_T, butterfly_one_third, [K_1 K_star_one_third K_2], {"K_1 = " + K_1, "K_* = " + K_star_one_third,"K_2 = " + K_2}, [-10 30], "Butterfly with \lambda = 1/3")
plot_payout(S_T, butterfly_one_half, [K_1 K_star_one_half K_2], {"K_1 = " + K_1, "K_* = " + K_star_one_half,"K_2 = " + K_2}, [-10 30], "Butterfly with \lambda = 1/2")
plot_payout(S_T, butterfly_two_thirds, [K_1 K_star_two_thirds K_2], {"K_1 = " + K_1, "K_* = " + K_star_two_thirds,"K_2 = " + K_2}, [-10 30], "Butterfly with \lambda = 2/3")

% Part e - Call ladder
K_1 = 5;
K_2 = 32;
K_3 = 55;
call_ladder = european_call(S_T, K_1) - european_call(S_T, K_2) - european_call(S_T, K_3);
plot_payout(S_T, call_ladder, [K_1 K_2 K_3], {"K_1 = " + K_1, "K_2 = " + K_2, "K_3 = " + K_3}, [-75 35], "Call Ladder")
% The call ladder also experiences unbounded loss due to having more shorts
% than longs for S_T > K_3!

% Part f - Digital call spread
K_1 = 42; 
K_2 = 73;
dig_call_spread = digital_call(S_T, K_1) - digital_call(S_T, K_2);
plot_payout(S_T, dig_call_spread, [K_1 K_2], {"K_1 = "+ K_1, "K_2 = "+ K_2}, [-0.2 1.2], "Digital Call Spread")

%% Function Definitions

% Basic options

function VS_T = european_call(S_T, K)
    VS_T = (S_T - K) .* (S_T > K);
end

function VS_T = european_put(S_T, K)
    VS_T = (K - S_T) .* (S_T < K);
end

function VS_T = digital_call(S_T, K)
    VS_T = double(S_T > K); % Convert logical result to numerical
end

function VS_T = digital_put(S_T, K)
    VS_T = double(S_T < K);
end

% Consolidated function for plotting all the graphs nicely!
function plot_payout(S_T, VS_T, line_S_Ts, line_labels, Vbound, case_string)
    figure
    plot(S_T, VS_T)
    xlabel("S_T")
    ylabel("V(S_T)")
    xline(line_S_Ts, "r--", line_labels)
    ylim(Vbound) % Adjust bounds to make things look nicer
    title("Security value vs Payout: " + case_string)
end

% Problem 1 Part d- Butterfly
function [VS_T, K_star] = butterfly(S_T, K_1, K_2, lambda)
    K_star = lambda * K_1 + (1 - lambda) * K_2;
    VS_T = lambda * european_call(S_T, K_1) + (1 - lambda) * european_call(S_T, K_2) - european_call(S_T, K_star);
end