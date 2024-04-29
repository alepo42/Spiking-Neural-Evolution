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

function XOR(inputs::Vector{Bool}) 
    return reduce(xor, inputs)
end


# Grid search

results = DataFrame(LayersMin = UInt16[], LayersMax = UInt16[], Population = UInt16[], MeanFitness = Float64[], MaxFitness = Float64[], SuccessfulSimulations = UInt16[])
results_ts = ThreadSafeDict()

@Threads.threads for population in [20, 30, 40]
    @Threads.threads for lay_min in [1, 2, 3]
        for lay_max in lay_min : lay_min + 1
            evolution_parameters = EvolutionParameters(
                UInt32(15),    # How many simulations
                UInt32(200),  # How many iterations per simulation
                UInt32(population),    # Min number of random population
                UInt32(population),    # Max number of random population
                UInt16(lay_min),   # Min number of random hidden layers
                UInt16(lay_max),    # Max number of random hidden layers
                SpikingNeuralEvolution.MaxIterations
            )  
            histories = Evolve(XOR, n, evolution_parameters)

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
            results_ts[string(population) * string(lay_min) * string(lay_max)] = (lay_min, lay_max, population, mean(final_values), maximum(final_values), how_many_successful)

            println("$population - $lay_min - $lay_max: done, " * string(how_many_successful) * "/" * string(evolution_parameters.simulations))
        end
    end
end

for key in keys(results_ts)
    push!(results, results_ts[key])
end

sort!(results, :MeanFitness, rev = true)

println(results)

"""
   │ LayersMin  LayersMax  Population  MeanFitness  MaxFitness  SuccessfulSimulations 
   │ UInt16     UInt16     UInt16      Float64      Float64     UInt16                
───┼──────────────────────────────────────────────────────────────────────────────────
1  │         3          4          40     0.93125      1.0                          2
2  │         1          2          40     0.9125       1.0                          1
3  │         3          3          30     0.9125       1.0                          2
4  │         3          4          30     0.9125       1.0                          2
5  │         2          3          40     0.910417     1.0                          1
6  │         1          1          30     0.908333     1.0                          3
7  │         3          4          20     0.908333     1.0                          5
8  │         2          3          30     0.908333     1.0                          4
9  │         2          2          40     0.9          1.0                          3
10 │         2          2          30     0.895833     1.0                          1
11 │         1          2          20     0.89375      1.0                          3
12 │         1          1          40     0.891667     1.0                          2
13 │         3          3          20     0.891667     1.0                          1
14 │         3          3          40     0.891667     1.0                          1
15 │         2          2          20     0.885417     1.0                          1
16 │         2          3          20     0.883333     0.96875                      0
17 │         1          2          30     0.88125      1.0                          2
18 │         1          1          20     0.879167     1.0                          1
"""