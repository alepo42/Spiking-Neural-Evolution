""" 
This source code performs a grid search over the parameters
that control the minimum and maximum number of hidden layers
and the size of the population in randomly generated circuits.

The target function is a 5-bits XOR.
"""

push!(LOAD_PATH, "../../src/")

using SpikingNeuralEvolution
using Plots
using DataFrames
using Statistics

# For speed
using ThreadSafeDicts

n = UInt16(5)

f_2 = [[1], [1, 4], [1, 5], [2, 4], [3, 5], [4, 5], [1, 2, 3], [1, 2, 4], [1, 2, 5], [2, 3, 4], [2, 3, 5], [1, 2, 3, 4], [1, 3, 4, 5], [2, 3, 4, 5], [1, 2, 3, 4, 5], false]

function ANF(inputs::Vector{Bool}) 
    return SpikingNeuralEvolution.Utils.EvaluateANFFunction(inputs, f_2)
end


# Grid search

results = DataFrame(NewLayer = Float64[], NewNeuron = Float64[], RemoveNeuron = Float64[], MeanFitness = Float64[], MaxFitness = Float64[], SuccessfulSimulations = UInt16[])
results_ts = ThreadSafeDict()

@Threads.threads for prob_new_layer in [0.001, 0.003, 0.005]
    @Threads.threads for prob_new_neuron in [0.030, 0.010]
        for prob_remove_neuron in [0.05, 0.15, 0.25]

            # These parameters have been fixed from the previous phase
            evolution_parameters = EvolutionParameters(
                UInt32(20),    # How many simulations
                UInt32(500),  # How many iterations per simulation
                UInt32(125),   # Min number of random population
                UInt32(125),   # Max number of random population
                UInt16(2),     # Min number of random hidden layers
                UInt16(3),     # Max number of random hidden layers
                SpikingNeuralEvolution.MaxIterations,
                1
            )  

            mutation_probabilities = MutationProbabilities(
                prob_new_layer,  # New layer
                0.080,  # Remove layer
                prob_new_neuron,  # New neuron
                prob_remove_neuron,  # Remove neuron
                0.005,  # Add rule
                0.080,  # Remove rules
                0.010   # Random input lines
            )

            histories = Evolve(ANF, n, evolution_parameters)

            # I create an array that will contain all the final final_values
            # of fitness for each simulation

            final_values = []

            # This var indicates the iterations required to achieve (if this happens)
            # the threshold value of fitness (in case not specified, 1)

            how_many_successful = 0
            
            for key in keys(histories)
                push!(final_values, histories[key][2][end])

                if histories[key][2][end] == 1 
                    how_many_successful += 1
                end

            end
            
            #push!(results, (lay_min, lay_max, population, mean(final_values), maximum(final_values), mean_iterations_of_max_fitness / how_many_successful, iteration_of_max_fitness))
            results_ts[string(prob_new_layer) * string(prob_new_neuron) * string(prob_remove_neuron)] = (prob_new_layer, prob_new_neuron, prob_remove_neuron, mean(final_values), maximum(final_values), how_many_successful)

            println("$prob_new_layer - $prob_new_neuron - $prob_remove_neuron: done, " * string(how_many_successful) * "/" * string(evolution_parameters.simulations))
        end
    end
end

for key in keys(results_ts)
    push!(results, results_ts[key])
end

sort!(results, :MeanFitness, rev = true)

println(results)

"""
Row │ NewLayer  NewNeuron  RemoveNeuron  MeanFitness  MaxFitness  SuccessfulSimulations 
│ Float64   Float64    Float64       Float64      Float64     UInt16                
─────┼───────────────────────────────────────────────────────────────────────────────────
1 │    0.003       0.03          0.05     0.871875     0.96875                      0
2 │    0.005       0.01          0.05     0.86875      0.90625                      0
3 │    0.003       0.03          0.25     0.8625       0.96875                      0
4 │    0.005       0.01          0.25     0.8625       0.90625                      0
5 │    0.005       0.03          0.05     0.859375     0.90625                      0
6 │    0.005       0.03          0.25     0.857812     0.9375                       0
7 │    0.001       0.01          0.05     0.85625      0.90625                      0
8 │    0.001       0.01          0.15     0.85625      0.90625                      0
9 │    0.003       0.01          0.15     0.85625      0.90625                      0
10 │    0.005       0.01          0.15     0.85625      0.90625                      0
11 │    0.001       0.01          0.25     0.85625      0.90625                      0
12 │    0.005       0.03          0.15     0.854688     0.90625                      0
13 │    0.003       0.01          0.25     0.854688     0.9375                       0
14 │    0.003       0.03          0.15     0.853125     0.9375                       0
15 │    0.001       0.03          0.25     0.85         0.90625                      0
16 │    0.003       0.01          0.05     0.848437     0.90625                      0
17 │    0.001       0.03          0.05     0.84375      0.90625                      0
18 │    0.001       0.03          0.15     0.84375      0.90625                      0
"""