push!(LOAD_PATH, "src/")

using SNPCircuit
using GeneticAlgorithms

for i in 1:100
    println("Vado")
    
    c = Crossover(GenerateRandomCircuit(UInt16(4), UInt16(1), UInt16(2), UInt16(2)), 
                  GenerateRandomCircuit(UInt16(4), UInt16(1), UInt16(2), UInt16(2)),
                  UInt16(4))
    println(c)
    println(EvaluateCircuitOfNeurons(c, [true, true, false, true]))
    
end