""" 
This source code performs a grid search over the parameters
that control the minimum and maximum number of hidden layers
in randomly generated circuits.

The target function is a 5-bits XOR.
"""

push!(LOAD_PATH, "../src/")

using SpikingNeuralEvolution
using Plots
using DataFrames
using Statistics

n = UInt16(5)

function XOR(inputs::Vector{Bool}) 
    return reduce(xor, inputs)
end


# Grid search

results = DataFrame(Min = UInt16[], Max = UInt16[], MeanFitness = Float64[], MaxFitness = Float64[])

for min in 1 : 3
    for max in min : min + 3
        evolution_parameters = EvolutionParameters(
            UInt32(20),    # How many simulations
            UInt32(1000),   # How many iterations per simulation
            UInt32(40),    # Min number of random population
            UInt32(60),   # Max number of random population
            UInt16(min),   # Min number of random hidden layers
            UInt16(max)    # Max number of random hidden layers
        )  
        histories = Evolve(XOR, n, evolution_parameters)

        # I create an array that will contain all the final final_values
        # of fitness for each simulation

        final_values = []
        for history in histories
            push!(final_values, history[2][end])
        end
        
        push!(results, (min, max, mean(final_values), maximum(final_values)))
    end
end

sort!(results, :MeanFitness, rev = true)

println(results)

"""

2, 5 is the "best" combination!

Row │ Min     Max     MeanFitness  MaxFitness 
│ UInt16  UInt16  Float64      Float64    
─────┼─────────────────────────────────────────
1 │      2       5     0.903125     1.0
2 │      2       3     0.879687     0.96875
3 │      3       6     0.879687     0.96875
4 │      1       3     0.878125     0.96875
5 │      2       2     0.878125     0.9375
6 │      2       4     0.878125     0.96875
7 │      1       4     0.876563     1.0
8 │      3       5     0.871875     0.90625
9 │      3       4     0.870313     0.9375
10 │      3       3     0.86875      0.9375
11 │      1       2     0.865625     1.0
12 │      1       1     0.851562     0.96875
"""