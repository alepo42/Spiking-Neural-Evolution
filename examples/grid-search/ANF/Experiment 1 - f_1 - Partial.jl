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

f_1 = [[2], [1, 2], [1, 4], [1, 5], [2, 3], [3, 5], [1, 2, 3], [1, 2, 5], [1, 3, 4], [1, 3, 5], [1, 4, 5], [3, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5], [1, 2, 3, 4, 5], true]

function ANF(inputs::Vector{Bool}) 
    return SpikingNeuralEvolution.Utils.EvaluateANFFunction(inputs, f_1)
end


# Grid search

results = DataFrame(LayersMin = UInt16[], LayersMax = UInt16[], Population = UInt16[], MeanFitness = Float64[], MaxFitness = Float64[], SuccessfulSimulations = UInt16[])
results_ts = ThreadSafeDict()

@Threads.threads for population in [75, 100, 125]
    @Threads.threads for lay_min in [1, 2, 3]
        for lay_max in lay_min : lay_min + 1
            evolution_parameters = EvolutionParameters(
                UInt32(20),    # How many simulations
                UInt32(500),  # How many iterations per simulation
                UInt32(population),    # Min number of random population
                UInt32(population),    # Max number of random population
                UInt16(lay_min),   # Min number of random hidden layers
                UInt16(lay_max),    # Max number of random hidden layers
                SpikingNeuralEvolution.MaxIterations,
                0.7
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
Row │ LayersMin  LayersMax  Population  MeanFitness  MaxFitness  SuccessfulSimulations 
     │ UInt16     UInt16     UInt16      Float64      Float64     UInt16                
─────┼──────────────────────────────────────────────────────────────────────────────────
   1 │         2          2         100     0.918182    1.0                           3
   2 │         2          2         125     0.915909    1.0                           1
   3 │         2          3         100     0.911364    1.0                           1
   4 │         3          3         100     0.911364    1.0                           1
   5 │         3          4          75     0.911364    1.0                           1
   6 │         3          4         125     0.909091    0.954545                      0
   7 │         1          2         100     0.906818    1.0                           1
   8 │         1          2         125     0.904545    1.0                           1
   9 │         3          4         100     0.902273    1.0                           1
  10 │         1          2          75     0.902273    0.954545                      0
  11 │         2          3         125     0.902273    1.0                           1
  12 │         3          3          75     0.902273    1.0                           1
  13 │         2          2          75     0.902273    1.0                           1
  14 │         1          1         125     0.902273    0.954545                      0
  15 │         3          3         125     0.897727    1.0                           1
  16 │         1          1         100     0.897727    0.954545                      0
  17 │         1          1          75     0.895455    1.0                           1
  18 │         2          3          75     0.888636    0.954545                      0
"""