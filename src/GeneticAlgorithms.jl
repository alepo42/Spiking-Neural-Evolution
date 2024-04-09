module GeneticAlgorithms

    using SNPCircuit

    export Crossover


    # This function performs a Crossover given two input circuits. 
    function Crossover(circuit1::CircuitOfNeurons, circuit2::CircuitOfNeurons)
        
        # Between 30% and 70% taken from the first parent
        slice = rand((3 : 7)) / 10 
        
        index_slice1 = Int64(floor(slice1 * NumberOfLayers(circuit1)))   
        
        # And 1 - slice taken from the second parent
        index_slice2 = NumberOfLayers(circuit2) - Int64(floor((1 - slice) * NumberOfLayers(circuit2)))

        if index_slice2 == NumberOfLayers(circuit2)
            index_slice2 = NumberOfLayers(circuit2) - 1
        end

        # The merged circuit will contain the layers [1:index_slice1] from the first
        # circuit, and the layers [index_slice2:end] from the second circuit
        individual = CircuitOfNeurons(append!(
            deepcopy(circuit1.layers[1:index_slice1]), 
            deepcopy(circuit2.layers[index_slice2:end])))

        # Notice that this procedure can break some connection between neurons, therefore
        # a 'cleaning' procedure is executed in order to remove empty input lines

        return CleanCircuit(individual)
    end
end