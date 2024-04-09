push!(LOAD_PATH, "src/")

using SNPCircuit
using GeneticAlgorithms

for i in 1:100
    println("Vado")
    
    c = Crossover(GenerateRandomNetwork(4, 1, 2, 2), GenerateRandomNetwork(4, 1, 2, 2), UInt16(4))
    println(c)
    println(EvaluateCircuitOfNeurons(c, [true, true, false, true]))
    
end