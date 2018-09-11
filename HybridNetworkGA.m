function [final_subgroups, latent_network, output] = HybridNetworkGA(A, metric)

%% Main Program. Set optimization parameters for GA solver, record output, and get the final Phenotype %%

lb = ones(size(A,1),1);
ub = size(A,1) .* ones(size(A,1),1)-1;
Distance = squareform(pdist(A, metric));
Distance(logical(eye(size(Distance)))) =Inf;
[~, nearestNeighbors] = sort(Distance, 2);
[~,  NN_Rank] = sort(nearestNeighbors , 2, 'ascend');

opts = gaoptimset('PlotFcn', @gaplotbestf, 'CrossoverFcn', @one_point_cross, ...
                    'MutationFcn',  @indegree_mut,'CreationFcn', @uniform_init,...
                    'SelectionFcn', @selectiontournament,  'UseParallel', true);

[x, fval, exitflag, Output] = ga(@evaluationFunction, size(A,1), [], [],[],[],lb,ub,[],[],opts);

output.x = x;
output.fval = fval;
output.exitflag =exitflag;
output.Output = Output;

[final_subgroups, latent_network] = findFinalSolution(x, 'assym_eval');

%% Evaluation Function %%

    function result = evaluationFunction(x)
        G = digraph( double( NN_Rank <= x ));
        normalized_diff = min(centrality(G, 'indegree'), centrality(G,'outdegree')) .* (centrality(G, 'indegree')- centrality(G,'outdegree'));
        result = norm(normalized_diff, 2); 
    end

%% Genetic Operators %%
    %% Initiation Functions. Best is uniform initiation %%
    
    function Population = uniform_init(Genomelength, FitnessFcn, options)
        pop_size = options.PopulationSize;
        Population = randi([floor(log2(size(A,1))),size(A,1)-1], pop_size, size(A,1));
    end

    function Population = zeta_init(Genomelength, FitnessFcn, options)
        pop_size = options.PopulationSize;

        Population = randraw('zeta', 3, pop_size, size(A,1));
        Population(Population > size(A,1)-1) = size(A,1)-1;
    end

    function Population = network_init(Genomelength, FitnessFcn, options)
        pop_size = options.PopulationSize;
        
        init_Population = ceil(log2(size(A,1)))*ones(pop_size, size(A,1));
        Population = arrayfun(@(i) indegree_mut_child(init_Population(i, :)), ...
            1:size(init_Population, 1), 'UniformOutput', false);
        
        Population = cell2mat(Population');
    end

    %% Crossover Functions. Best is One-point crossover %%

    function xoverKids = one_point_cross(parents, options, nvars, FitnessFcn, ...
        unused,thisPopulation)

        xoverKids = arrayfun(@(i) one_point_cross_child(thisPopulation(parents(i), :), thisPopulation(parents(i+1), :)), ...
            1:2:numel(parents)-1, 'UniformOutput', false);
        xoverKids = cell2mat(xoverKids');
    end

        function [child]= one_point_cross_child(parent_1, parent_2)
            crossover_point = randi([1,size(A,1)],1,1);
            child = [parent_1(1:crossover_point), parent_2(crossover_point+1:end)];
        end

    function xoverKids = uniform_cross(parents, options, nvars, FitnessFcn, ...
        unused,thisPopulation)
    
        xoverKids = arrayfun(@(i) uniform_cross_child(thisPopulation(parents(i), :), thisPopulation(parents(i+1), :)), ...
            1:2:numel(parents)-1, 'UniformOutput', false);
        xoverKids = cell2mat(xoverKids');       
    end

        function [child]= uniform_cross_child(parent_1, parent_2)
            crossover_mask = randi([0,1],1,size(A,1));
            child = parent_1 .* crossover_mask + parent_2 .* (1-crossover_mask);
        end
    
    %% Mutation Functions. Best are the degree mutation functions %%

    function mutationChildren = uniform_mut(parents, options, nvars, FitnessFcn, ...
        state,thisScore, thisPopulation, rate)
        
        mutationChildren  = thisPopulation(parents, :);
        num_elements = ceil(rate*size(mutationChildren,2));
        mutationChildren = arrayfun(@(i) uniform_mut_child(mutationChildren(i, :), num_elements), ...
            1:size(mutationChildren, 1), 'UniformOutput', false);
        mutationChildren = cell2mat(mutationChildren');
        
    end

        function [mutant]= uniform_mut_child(mutant, num_elements)
            indices_to_mutate = randperm(size(mutant,2),num_elements);
            mutant(:, indices_to_mutate) = randi([0,size(A,1)-1],1,numel(indices_to_mutate));
        end

    function mutationChildren = zeta_mut(parents, options, nvars, FitnessFcn, ...
        state,thisScore, thisPopulation, rate)
        
        mutationChildren  = thisPopulation(parents, :);
        num_elements = ceil(rate*size(mutationChildren,2));
        for i = 1:size(mutationChildren, 1)
            indices_to_mutate = randperm(size(mutationChildren,2),num_elements);
            mutationChildren(i, indices_to_mutate)= randraw('zeta', 3, 1,numel(indices_to_mutate));
        end
        mutationChildren(mutationChildren > size(A,1)-1) = size(A,1) -1;
    end

    function mutationChildren = indegree_mut(parents, options, nvars, FitnessFcn, ...
        state,thisScore, thisPopulation)
        
        mutationChildren  = thisPopulation(parents, :);
        mutationChildren = arrayfun(@(i) indegree_mut_child(mutationChildren(i, :)), ...
            1:size(mutationChildren, 1), 'UniformOutput', false);
        mutationChildren = cell2mat(mutationChildren');
        
    end

        function [mutant]= indegree_mut_child(mutant)
            nearest_neighbor_graph = digraph( double( NN_Rank <= mutant ));
            centralities = centrality(nearest_neighbor_graph, 'indegree');
            [values,nodes] = sort(centralities);
            bottomNodes = nodes(1:size(values( values < mean(values) - std(values)),1));
            mutant(bottomNodes) = arrayfun(@(x) randi([1,x],1,1), mutant(bottomNodes));
            [values,nodes] = sort(centralities, 'descend');
            topNodes = nodes(1:size(values( values > mean(values) + std(values)),1));
            mutant(topNodes) = arrayfun(@(x) randi([x,size(A,1)-1],1,1), mutant(topNodes));
        end
    
    function mutationChildren = delta_degree_mut(parents, options, nvars, FitnessFcn, ...
        state,thisScore, thisPopulation)
        
        mutationChildren  = thisPopulation(parents, :);
        mutationChildren = arrayfun(@(i) delta_degree_mut_child(mutationChildren(i, :)), ...
            1:size(mutationChildren, 1), 'UniformOutput', false);
        mutationChildren = cell2mat(mutationChildren');
        
    end

        function [mutant]= delta_degree_mut_child(mutant)
            G = digraph( double( NN_Rank <= mutant ));
            delta_centralities = centrality(G, 'indegree')- centrality(G,'outdegree'); 
            bottomNodes = delta_centralities<0;
            mutant(bottomNodes) = arrayfun(@(x) randi([1,x],1,1), mutant(bottomNodes));
            topNodes = delta_centralities >0;
            mutant(topNodes) = arrayfun(@(x) randi([x,size(A,1)-1],1,1), mutant(topNodes));
        end

%% Creating final solution %%

    function [subgroups, nearest_neighbor_graph] = findFinalSolution(x, network_type)
        nearest_neighbor_graph = digraph( double( NN_Rank <= x ));
        G = adjacency(nearest_neighbor_graph);

        if strcmp(network_type,'sym_eval')
            G = G .* G';
            nearest_neighbor_graph = graph(G);
        end
        subgroups = GCModulMax1(G);
    end
end