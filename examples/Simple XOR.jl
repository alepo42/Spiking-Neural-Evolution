""" 
This source code uses the high-level APIs offered by 
SpikingNeuralEvolution.jl to find a circuit that evaluates
the XOR betwen 5 bits
"""

push!(LOAD_PATH, "../src/")

using SpikingNeuralEvolution
using Plots

n = UInt16(5)

function XOR(inputs::Vector{Bool}) 
    return reduce(xor, inputs)
end

histories = Evolve(XOR, n)

plot(histories)