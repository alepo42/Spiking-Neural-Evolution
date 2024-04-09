push!(LOAD_PATH, "src/")

using SNPCircuit
using GeneticAlgorithms


a = CrossoverHorizontal(GenerateRandomNetwork(4, 1, 2, 2), GenerateRandomNetwork(4, 1, 2, 2))
println(a)