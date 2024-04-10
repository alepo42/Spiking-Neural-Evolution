""" 
This source code exploits Genetic Algorithms (GA) in order to 
find a SN P-System circuit that reproduces the behavior
of a XOR function f : {0, 1}^n -> sum_{i=1}^n (xor(x_i))

It uses the high-level APIs offered by SpikingNeuralEvolution.jl
"""

push!(LOAD_PATH, "src/")

using SpikingNeuralEvolution

function XOR(inputs::Vector{Bool}) 
    return reduce(xor, inputs)
end

#TODO voglio poter scrivere sta roba
Simulate(XOR, inputs)