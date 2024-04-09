module GeneticAlgorithms

    using SNPCircuit

    include("Utils.jl")

    export Crossover, Fitness

    # This function performs a Crossover given two input circuits. 
    function Crossover(circuit1::CircuitOfNeurons, circuit2::CircuitOfNeurons, inputs::UInt16)

        # Between 30% and 70% taken from the first parent
        slice = rand((3 : 7)) / 10 
        
        index_slice1 = Int64(floor(slice * NumberOfLayers(circuit1)))   

        if index_slice1 == 0
            index_slice1 = 1
        end
        
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
        # a 'cleaning' procedure is executed in the middle layer in order to remove empty input lines
        individual = CleanCircuit(individual, inputs)

        return individual
    end

    function CleanCircuit(c::CircuitOfNeurons, inputs::UInt16)
        circuit = deepcopy(c)
        
        # Clearning the first layer
        for i in 1 : NumberOfNeurons(circuit.layers[1])
            circuit.layers[1].neurons[i].input_lines = filter(x -> x <= inputs, circuit.layers[1].neurons[i].input_lines)
        end
        
        # Cleaning the hidden layers
        for i in 2 : NumberOfLayers(circuit) - 1
            j = 1
            while j <= NumberOfNeurons(circuit.layers[i])
                 
                # I enumerate all the input lines that does not connect to any previous neuron
                unconnected = Differences(NumberOfNeurons(circuit.layers[i - 1]),
                                          circuit.layers[i].neurons[j].input_lines)
                
                
                if length(unconnected) == length(circuit.layers[i].neurons[j].input_lines)
                    # If all the input lines are dis-connected, the whole neuron is removed
                    deleteat!(circuit.layers[i].neurons, j)
                else
                    # Otherwise, only the dis-connected lines are removed
                    circuit.layers[i].neurons[j].input_lines = 
                        setdiff(circuit.layers[i].neurons[j].input_lines, unconnected)
                    
                    j += 1
                 
                end
            end
        end
        
        # The same cleaning operation is performed at the output layer
        # TODO: Può essere inglobato sopra togliendo il -1?
        for i in 1 : NumberOfNeurons(circuit.layers[end])
            unconnected = Differences(NumberOfNeurons(circuit.layers[end - 1]),
                circuit.layers[end].neurons[i].input_lines)
    
            circuit.layers[end].neurons[i].input_lines = setdiff(circuit.layers[end].neurons[i].input_lines, unconnected)
    
            if length(circuit.layers[end].neurons[i].input_lines) < 2
                circuit.layers[end].neurons[i].input_lines = 1 : NumberOfNeurons(circuit.layers[end - 1])
            end
        end
        
        return circuit
    end

    #TODO da commentare
    function Fitness(circuit::CircuitOfNeurons, examples::Vector{Vector{Bool}}, labels::Vector{Bool})
        correct = 0

        # The circuit is cloned so that, at the end of evaluation, spikes are reset (TODO migliorabile)
        circuit_clone = deepcopy(circuit) 

        for i in 1:length(examples)
            evaluation = EvaluateCircuitOfNeurons(circuit_clone, examples[i])[1]
            if evaluation == labels[i]
                correct += 1 
            end
        end

        return correct / length(examples)
    end

    #TODO da commentare
    function ProportionateSelection(fitness::Vector{Float64})
        normalized = sort(fitness) ./ sum(fitness)
        cumulative_probabilities = cumsum(reverse(normalized))
        selected_index = findfirst(x -> x > rand(), cumulative_probabilities)
        return selected_index
    end
end