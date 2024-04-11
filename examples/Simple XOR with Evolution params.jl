""" 
This source code uses the high-level APIs offered by 
SpikingNeuralEvolution.jl to find a circuit that evaluates
the XOR betwen 5 bits

Additionally, it gives the user the chance to choose the 
parameters of the simulation
"""

push!(LOAD_PATH, "../src/")

using SpikingNeuralEvolution
using Plots

n = UInt16(5)

function XOR(inputs::Vector{Bool}) 
    return reduce(xor, inputs)
end

evolution_parameters = EvolutionParameters(
    UInt32(20),    # How many simulations
    UInt32(500),   # How many iterations per simulation
    UInt32(80),    # Min number of random population
    UInt32(120),   # Max number of random population
    UInt16(1),     # Min number of random hidden layers
    UInt16(3)      # Max number of random hidden layers
)

histories = Evolve(XOR, n, evolution_parameters)

plot(histories)