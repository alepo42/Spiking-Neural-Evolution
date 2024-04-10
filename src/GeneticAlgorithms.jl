"""
    GeneticAlgorithms

A module providing functions to perform procedured based on
Genetic Algorithms (GA) on circuits composed of SN P-System neurons

"""
module GeneticAlgorithms
    # Libraries required
    using SNPCircuit
    using Utils
    using Random

    # Structs and functions that are exported to the user
    export Crossover, 
           Mutation,
           Mutation2,
           Fitness,
           CleanCircuit,
           ProportionateSelection


    """
    Crossover(circuit1, circuit2)

    Performs a crossover between two circuits. In particular, a slicing
    index _n_ is randomly sampled, and the first n layers of the first circuit
    are considered. These layers are attached to the last _m_ layers of the
    second circuit, creating a new circuit.

    # Arguments
    - `circuit1::CircuitOfNeurons`: The first parent.
    - `circuit2::CircuitOfNeurons`: The second parent.
    - `inputs::UInt16`: The number of inputs accepted by both parents.

    # Returns
    - `Circuit`: A new individual obtained by performing the crossover`.
    """
    function Crossover(circuit1::CircuitOfNeurons, circuit2::CircuitOfNeurons, inputs::UInt16)

        # Between 30% and 70% taken from the first parent
        slice = rand((3 : 7)) / 10 
        
        n = Int64(floor(slice * NumberOfLayers(circuit1)))   

        if n == 0
            n = 1
        end
        
        # And 1 - n taken from the second parent
        m = NumberOfLayers(circuit2) - Int64(floor((1 - slice) * NumberOfLayers(circuit2)))

        if m == NumberOfLayers(circuit2)
            m = NumberOfLayers(circuit2) - 1
        end

        # The merged circuit will contain the layers [1:n] from the first
        # circuit, and the layers [m:end] from the second circuit
        individual = CircuitOfNeurons(append!(
            deepcopy(circuit1.layers[1:n]), 
            deepcopy(circuit2.layers[m:end])))

        # Notice that this procedure can break some connection between neurons, therefore
        # a 'cleaning' procedure is executed in the middle layer in order to remove empty input lines
        individual = CleanCircuit(individual, inputs)

        return individual
    end

    """
    Mutation(circuit, probabilities)

    Performs a mutation on a circuit. The implemented possible mutations 
    are the following:

    - Adds a new random layer
    - Removes an existing random layer
    - Adds a new random neuron in a random layer
    - Removes an existing random neuron from a random layer
    - Adds a new rules in a random neuron
    - Removes an existing rule in a random neuron
    - Samples a new set of input lines for a random neuron

    New mutations are possible and easily implementable.

    # Arguments
    - `circuit::CircuitOfNeurons`: The circuit to be mutated.
    - `inputs::UInt16`: The number of inputs for the circuit.
    - `probabilities::MutationProbabilities`: A set of probabilities for the possible mutations.

    # Returns
    - `Circuit`: A new individual obtained by performing the crossover`.
    """
    function Mutation(circuit::CircuitOfNeurons, inputs::UInt16, probabilities = MutationProbabilities(0.01, 0.08, 0.03, 0.2, 0.005, 0.08, 0.01))
        
        # Adds a new layer
        if rand() < probabilities.new_layer

            if length(circuit.layers) == 2
                # TODO check questo due e commenta sta roba
                index = 2
            else
                index = rand((2 : length(circuit.layers)))
            end
            
            n_neurons = length(circuit.layers[index].neurons)

            neurons = Vector{Neuron}()

            for n_neuron in 1:n_neurons
                push!(neurons, GenerateRandomNeuron(NumberOfNeurons(circuit.layers[index - 1])))
            end
                
            insert!(circuit.layers, index, LayerOfNeurons(neurons))
        end
        
        # Remove a random layer (if the number of layers is greater than two)
        if rand() < probabilities.remove_layer
            if NumberOfLayers(circuit) > 2
                deleteat!(circuit.layers, rand((2 : NumberOfLayers(circuit) - 1)))
            end
        end

        # Insert a new random neuron in a random layer (not the last)
        if rand() < probabilities.new_neuron

            # The minus one is done to exclude the last layer from getting new neurons
            index_layer = rand((1 : NumberOfLayers(circuit) - 1))
            layer = circuit.layers[index_layer]

            if index_layer == 1
                neuron = GenerateRandomNeuron(inputs)
                insert!(layer.neurons, rand((1:NumberOfNeurons(layer))), neuron)  
            else
                neuron = GenerateRandomNeuron(NumberOfNeurons(circuit.layers[index_layer - 1]))
                insert!(layer.neurons, rand((1:NumberOfNeurons(layer))), neuron)
            end
        end
        
        # Remove a random neuron from a random layer (not the last)
        if rand() < probabilities.remove_neuron

            index_layer = rand((1 : NumberOfLayers(circuit) - 1))
            layer = circuit.layers[index_layer]

            if length(layer.neurons) > 2
                index_remove = rand((1:length(layer.neurons)))
                deleteat!(layer.neurons, index_remove)
            end
        end
            
        
        # Remove a random rule from a random neuron (consider also the last layer)
        if rand() < probabilities.remove_rule

            index_layer = rand((1 : NumberOfLayers(circuit)))
            layer = circuit.layers[index_layer]

            index_neuron = rand((1 : NumberOfNeurons(layer)))

            if length(layer.neurons[index_neuron].rules) > 2
                index_remove = rand((1:length(layer.neurons[index_neuron].rules)))
                deleteat!(layer.neurons[index_neuron].rules, index_remove)
            end
        end  
            
        # Add a new random rule to a random neuron
        if rand() < probabilities.new_rule

            index_layer = rand((1 : NumberOfLayers(circuit)))
            layer = circuit.layers[index_layer]

            index_neuron = rand((1 : NumberOfNeurons(layer)))

            #TODO Sistema sta merda
            added = false
            while added == false
                rule = rand((1:100))
                if (rule in layer.neurons[index_neuron].rules) == false
                    push!(layer.neurons[index_neuron].rules, rule)
                    added = true
                end
            end
        end  
                
        # Samples a new set of input lines for a random neuron
        if rand() < probabilities.random_input_lines

            index_layer = rand((1 : NumberOfLayers(circuit)))
            layer = circuit.layers[index_layer]

            index_neuron = rand((1 : NumberOfNeurons(layer)))

            if index_layer == 1
                previous_inputs = inputs
            else
                previous_inputs = NumberOfNeurons(circuit.layers[index_layer - 1])
            end
            
            possible_input_lines = randperm(length(collect(1:previous_inputs)))
           
            num_input_lines = rand(1:length(possible_input_lines))

            circuit.layers[index_layer].neurons[index_neuron].input_lines = collect(1:previous_inputs)[possible_input_lines[1:num_input_lines]]
        end
                    
        return circuit
    end

    function Mutation2(circuit::CircuitOfNeurons, inputs::UInt16, probabilities = MutationProbabilities(0.01, 0.08, 0.03, 0.2, 0.005, 0.08, 0.01))
        # Add a new layer
        if rand() < probabilities.new_layer
            if length(circuit.layers) == 2
                index = 2
            else
                index = rand((2 : length(circuit.layers)))
            end
            
            n_neurons = length(circuit.layers[index].neurons)
    
            neurons = Vector{Neuron}()
    
            for n_neuron in 1:n_neurons
                push!(neurons, GenerateRandomNeuron(NumberOfNeurons(circuit.layers[index - 1])))
            end
                
            insert!(circuit.layers, index, LayerOfNeurons(neurons))
                    
            #Correggere i collegamenti del layer dopo
            #for neuron in circuit.layers[index + 1].neurons
        end
        
        # Remove a layer
        if rand() < probabilities.remove_layer
            if length(circuit.layers) > 2
                deleteat!(circuit.layers, rand((2:length(circuit.layers) - 1)))
            end
        end
            
        
        
        for i in 1 : length(circuit.layers) - 1
            layer = circuit.layers[i]
                
            # Insert new neuron
            if rand() < probabilities.new_neuron
                if i == 1
                    neuron = GenerateRandomNeuron(inputs)
                    insert!(layer.neurons, rand((1:length(layer.neurons))), neuron)  
                else
                    neuron = GenerateRandomNeuron(NumberOfNeurons(circuit.layers[i - 1]))
                    insert!(layer.neurons, rand((1:length(layer.neurons))), neuron)
                end
            end
            
            # Remove a neuron
            if rand() < probabilities.remove_neuron
                if length(layer.neurons) > 2
                    index_remove = rand((1:length(layer.neurons)))
                    deleteat!(layer.neurons, index_remove)
                end
            end
                
            for j in 1 : length(circuit.layers[i].neurons)
            
               # Remove rule
               if rand() < probabilities.remove_rule
                   if length(circuit.layers[i].neurons[j].rules) > 2
                        index_remove = rand((1:length(circuit.layers[i].neurons[j].rules)))
                        deleteat!(circuit.layers[i].neurons[j].rules, index_remove)
                   end
               end  
                   
               # New rule
               if rand() <probabilities.new_rule
                   added = false
                   while added == false
                        rule = rand((1:100))
                        if (rule in circuit.layers[i].neurons[j].rules) == false
                            push!(circuit.layers[i].neurons[j].rules, rule)
                            added = true
                        end
                   end
               end  
                    
               # New input_lines
               if rand() < probabilities.random_input_lines
                   if i == 1
                       previous_inputs = inputs
                   else
                       previous_inputs = NumberOfNeurons(circuit.layers[i - 1])
                   end
                   
                   possible_input_lines = randperm(length(collect(1:previous_inputs)))
    
                   num_input_lines = rand(1:length(possible_input_lines))
    
                   circuit.layers[i].neurons[j].input_lines = collect(1:previous_inputs)[possible_input_lines[1:num_input_lines]]
               end
            end     
            
        end
                    
        return circuit
    end


    """
    CleanCircuit(c, inputs)

    Performs a cleaning operation of a given circuit. After performing
    a crossover or a mutation, some input lines of some neurons can be
    invalid (e.g., connected to a neuron that is not there anymore). This
    function removes all the invalid input lines.

    # Arguments
    - `c::CircuitOfNeurons`: The considered circuit.
    - `inputs::UInt16`: The number of inputs of the circuit.

    # Returns
    - `Circuit`: A new circuit with no invalid input lines`.
    """
    function CleanCircuit(c::CircuitOfNeurons, inputs::UInt16)

        # Cloning the input circuit for convenience reasons
        circuit = deepcopy(c)
        
        # Cleaning the first layer
        for i in 1 : NumberOfNeurons(circuit.layers[1])
            circuit.layers[1].neurons[i].input_lines = filter(x -> x <= inputs, circuit.layers[1].neurons[i].input_lines)
        end
        
        # Cleaning the hidden layers
        for i in 2 : NumberOfLayers(circuit) - 1
            j = 1
            while j <= NumberOfNeurons(circuit.layers[i])
                 
                # Enumerating all the input lines that does not connect to any previous neuron
                unconnected = Differences(NumberOfNeurons(circuit.layers[i - 1]),
                                          circuit.layers[i].neurons[j].input_lines)
                
                
                if length(unconnected) == length(circuit.layers[i].neurons[j].input_lines)
                    # If all the input lines are invalid, the whole neuron is removed
                    deleteat!(circuit.layers[i].neurons, j)
                else
                    # Otherwise, only the invalid lines are removed
                    circuit.layers[i].neurons[j].input_lines = 
                        setdiff(circuit.layers[i].neurons[j].input_lines, unconnected)
                    
                    j += 1
                 
                end
            end

            # If, after the clean, the layer has no neurons, adds a random neuron
            if NumberOfNeurons(circuit.layers[i]) == 0
                if i == 1
                    push!(circuit.layers[i].neurons, GenerateRandomNeuron(inputs))
                else
                    push!(circuit.layers[i].neurons, GenerateRandomNeuron(NumberOfNeurons(circuit.layers[i - 1])))
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

        #TODO Considera il caso in cui il layer rimane senza neuroni
        
        return circuit
    end

    """
    Fitness(circuit, examples, labels)

    Computes the fitness value of a circuit with respect to a set of examples and
    relative labels. This value represents the number of correct labels computed by the
    circuit. The value of fitness is between 0 (all wrong) and 1 (all correct).

    # Arguments
    - `circuit::CircuitOfNeurons`: The considered circuit.
    - `examples::Vector{Vector{Bool}}`: The set of examples, typically a set of 2^n combinations
    - `labels::Vector{Bool}`: The corresponding set of labels, one for each example

    # Returns
    - `Float64`: The value of fitness, between 0 and 1`.
    """
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

    """
    ProportionateSelection(fitness)

    Returns the index of the selected individual, according to a sequence of fitness
    values. The larger the fitness of an individual, the larger the probability to
    be selected by the algorithm

    # Arguments
    - `fitness::Vector{Float64}`: The set of fitness values of each circuit.

    # Returns
    - `Float64`: The value of fitness, between 0 and 1`.
    """
    function ProportionateSelection(fitness::Vector{Float64})
        # Normalizing the set of fitnesses
        normalized = sort(fitness) ./ sum(fitness)

        # Computing the cumulative probabilities
        cumulative_probabilities = cumsum(reverse(normalized))

        # Sampling an item from the cumulative probabilities, the larger the
        # fitness, the larger the probability to be chosen
        selected_index = findfirst(x -> x > rand(), cumulative_probabilities)

        return selected_index
    end


    """
    Simulate(function, inputs, n_simulations)

    Returns the index of the selected individual, according to a sequence of fitness
    values. The larger the fitness of an individual, the larger the probability to
    be selected by the algorithm

    # Arguments
    - `fitness::Vector{Float64}`: The set of fitness values of each circuit.

    # Returns
    - `Float64`: The value of fitness, between 0 and 1`.
    """
    #TODO una funzione che prende in input una fuznione e fa le simulazioni da sola
    #function Simulate(function::Function, inputs::UInt16)
    #
    #end

end #Module