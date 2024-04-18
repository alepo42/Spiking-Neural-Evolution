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

results = DataFrame(LayersMin = UInt16[], LayersMax = UInt16[], Population = UInt16[], MeanFitness = Float64[], MaxFitness = Float64[])

for population in [80, 90, 100, 110, 120]
    for lay_min in [1, 2, 3]
        for lay_max in lay_min : lay_min + 2
            evolution_parameters = EvolutionParameters(
                UInt32(20),    # How many simulations
                UInt32(800),  # How many iterations per simulation
                UInt32(population),    # Min number of random population
                UInt32(population),    # Max number of random population
                UInt16(lay_min),   # Min number of random hidden layers
                UInt16(lay_max)    # Max number of random hidden layers
            )  
            histories = Evolve(XOR, n, evolution_parameters)

            # I create an array that will contain all the final final_values
            # of fitness for each simulation

            final_values = []
            for history in histories
                push!(final_values, history[2][end])
            end
            
            push!(results, (lay_min, lay_max, population, mean(final_values), maximum(final_values)))
        end
    end
end

sort!(results, :MeanFitness, rev = true)

println(results)

"""
From 2 to 4 layers and 110 individuals is the best combination!

Row │ LayersMin  LayersMax  Population  MeanFitness  MaxFitness      
────┼───────────────────────────────────────────────────────────
1   │         2          4         110     0.915625     1.0
2   │         3          5         110     0.910937     0.96875
3   │         3          5         100     0.909375     1.0
4   │         3          3         110     0.909375     0.96875
5   │         3          4          90     0.901563     1.0
6   │         2          4         120     0.9          1.0
7   │         3          4         120     0.9          1.0
8   │         3          5          90     0.898438     0.96875
9   │         2          4         100     0.898438     0.9375
10  │         3          4         100     0.898438     1.0
11  │         3          3         120     0.898438     0.96875
12  │         3          5         120     0.898438     0.96875
13  │         3          5          80     0.89375      1.0
14  │         2          3          90     0.89375      0.96875
15  │         1          3         110     0.89375      0.96875
16  │         3          4         110     0.89375      1.0
17  │         2          2         120     0.890625     1.0
18  │         1          2         110     0.8875       1.0
19  │         2          3         110     0.8875       0.96875
20  │         1          2          80     0.885938     1.0
21  │         1          1         120     0.885938     0.96875
22  │         1          3          80     0.884375     0.96875
23  │         2          2         100     0.884375     1.0
24  │         2          2         110     0.884375     0.96875
25  │         2          4          80     0.882812     0.96875
26  │         2          3         100     0.882812     0.96875
27  │         1          3         120     0.882812     0.96875
28  │         3          3          80     0.879687     0.9375
29  │         2          4          90     0.879687     0.9375
30  │         1          3         100     0.879687     0.96875
31  │         1          1         110     0.879687     0.9375
32  │         2          2          80     0.878125     0.9375
33  │         2          3         120     0.878125     0.96875
34  │         3          3          90     0.876563     0.9375
35  │         3          3         100     0.875        1.0
36  │         1          1          90     0.873437     0.96875
37  │         2          3          80     0.871875     1.0
38  │         1          3          90     0.871875     0.9375
39  │         2          2          90     0.871875     0.96875
40  │         3          4          80     0.86875      0.96875
41  │         1          2         100     0.86875      0.9375
42  │         1          1         100     0.865625     0.9375
43  │         1          1          80     0.864062     0.9375
44  │         1          2         120     0.860938     0.90625
45  │         1          2          90     0.859375     0.90625