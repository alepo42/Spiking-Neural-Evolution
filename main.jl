push!(LOAD_PATH, "src/")

using SpikingNeuralEvolution
using Plots

function XOR(inputs::Vector{Bool}) 
    return reduce(xor, inputs)
end

histories = Evolve(XOR, UInt16(5))

plot(histories)