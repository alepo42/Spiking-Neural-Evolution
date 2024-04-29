""" 
This source code performs a grid search over the parameters
that control the minimum and maximum number of hidden layers
and the size of the population in randomly generated circuits.

The target function is a 8-bits XOR.
"""

push!(LOAD_PATH, "../../src/")

using SpikingNeuralEvolution
using Plots
using DataFrames
using Statistics

# For speed
using ThreadSafeDicts

n = UInt16(5)

f_1 = [[2], [1, 2], [1, 4], [1, 5], [2, 3], [3, 5], [1, 2, 3], [1, 2, 5], [1, 3, 4], [1, 3, 5], [1, 4, 5], [3, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5], [1, 2, 3, 4, 5], true]

function ANF(inputs::Vector{Bool}) 
    return SpikingNeuralEvolution.Utils.EvaluateANFFunction(inputs, f_1)
end


# Grid search

results = DataFrame(NewLayer = Float64[], NewNeuron = Float64[], RemoveNeuron = Float64[], MeanFitness = Float64[], MaxFitness = Float64[], SuccessfulSimulations = UInt16[])
results_ts = ThreadSafeDict()

@Threads.threads for prob_new_layer in [0.0033, 0.0066, 0.0100]
    @Threads.threads for prob_new_neuron in [0.05, 0.10]
        for prob_remove_neuron in [0.10, 0.20, 0.30]

            # These parameters have been fixed from the previous phase
            evolution_parameters = EvolutionParameters(
                UInt32(20),    # How many simulations
                UInt32(1200),  # How many iterations per simulation
                UInt32(100),   # Min number of random population
                UInt32(100),   # Max number of random population
                UInt16(1),     # Min number of random hidden layers
                UInt16(3),     # Max number of random hidden layers
                SpikingNeuralEvolution.MaxIterations
            )  

            mutation_probabilities = MutationProbabilities(
                prob_new_layer,      # New layer
                0.080,               # Remove layer
                prob_new_neuron,     # New neuron
                prob_remove_neuron,  # Remove neuron
                0.005,               # Add rule
                0.080,               # Remove rules
                0.010                # Random input lines
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
"""