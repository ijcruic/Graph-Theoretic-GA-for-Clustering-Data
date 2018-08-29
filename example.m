%% Load in a Test Data set and define what similarity metric you would like to use%%
load('fisheriris');
Data = meas;
Target = species;
Metric='euclidean';

%% Run Hybrid GA to get cluster paritions and latent, best-fit graph %%
[GA_degree_results.subgroups, GA_degree_results.network,  GA_degree_results.output] = ...
         HybridNetworkGA(Data, Metric);
     
%% Get ARI value for found cluster paritions versus formal classes %%
RandIndex(grp2idx(GA_degree_results.subgroups), grp2idx(Target))

%% Plot Latent Network to observe relationships between entities and clsuters %%
figure;
plot(GA_degree_results.network,'Layout','force','NodeLabel',Target,'NodeCData',GA_degree_results.subgroups);