module SNPCircuit
    # Library required when generating a random network
    using Random

    # Structs and functions to be exported to the user
    export Neuron, LayerOfNeurons, CircuitOfNeurons, EvaluateCircuitOfNeurons, GenerateRandomNetwork, NumberOfLayers, NumberOfNeurons

    # A neuron represents the unitary entity bla bla
    mutable struct Neuron
        rules::Vector{UInt32}
        input_lines::Vector{Int32}
        number_of_spikes::UInt64
    end

    # A layer of neurons is just a list of neurons, each containing its own rules,
    # its input lines and its own number of spikes. Each neuron takes its inputs
    # from the previous layer; similarly, the output values computed will be taken
    # by the neurons in the next layer, hence it is not necessary to explicitly
    # duplicate the output values produced by the neurons.
    mutable struct LayerOfNeurons
        neurons::Vector{Neuron}
    end

    # A spiking neural circuit is a list of layers of neurons. The output produced
    # by 
    mutable struct CircuitOfNeurons
        layers::Vector{LayerOfNeurons}
    end

    # Getter for the number of neurons in a layer
    function NumberOfLayers(circuit::CircuitOfNeurons)
        return length(circuit.layers)
    end

    # Getter for the number of neurons in a layer
    function NumberOfNeurons(layer::LayerOfNeurons)
        return length(layer.neurons)
    end




    # Given an input vector, compute the output and the new internal state of the
    # neuron.
    function EvaluateNeuron(sigma::Neuron, input_vector::Vector{Bool})
        # Compute the number of input spikes
        TotalNumberOfSpikes::UInt64 = sigma.number_of_spikes
        for i in sigma.input_lines
            if input_vector[i]
              TotalNumberOfSpikes += 1
            end
        end
        # Checks whether the corresponding rule exists
        if TotalNumberOfSpikes in sigma.rules
            sigma.number_of_spikes = 0
            return true
        else
            sigma.number_of_spikes = TotalNumberOfSpikes
            return false
        end
    end

    # Given an input vector, compute the output and the new internal state of each
    # neuron in the layer
    function EvaluateLayerOfNeurons(layer::LayerOfNeurons, input_vector::Vector{Bool})
        output_vector::Vector{Bool} = []
        for neuron in layer.neurons
            push!(output_vector, EvaluateNeuron(neuron, input_vector))
        end
        return output_vector
    end

    # Given an input vector, compute the output and the new internal state of each
    # neuron in the circuit
    function EvaluateCircuitOfNeurons(circuit::CircuitOfNeurons, input_vector::Vector{Bool})
        output_vector::Vector{Bool} = input_vector
        for layer in circuit.layers
            output_vector = EvaluateLayerOfNeurons(layer, output_vector)
        end
        return output_vector
    end

    ## Functions used to generate random networks

    # This function takes as input bla bla
    function GenerateRandomNetwork(inputs::Int64, outputs::Int64, min_layers = 1, max_layers = 2)
        # This variable refers to the number of hidden layers
        n_layers = rand((min_layers:max_layers))
        
        layers = Vector{LayerOfNeurons}()
        
        # Hidden layers
        for n_layer in 1:n_layers

            # We assume that, when generating a randon network, each layer will contain x \in [1, n] 
            # neurons, where n is the number of neurons of the previous layer
            if n_layer == 1
                n_neurons = rand((1 : inputs))
            else
                n_neurons = rand((1 : NumberOfNeurons(layers[n_layer - 1])))
            end

            neurons = Vector{Neuron}()

            for n_neuron in 1:n_neurons
                # Each neuron has random {rules, input_lines, number_of_spikes}
                if n_layer == 1
                    push!(neurons, GenerateRandomNeuron(inputs))
                else
                    push!(neurons, GenerateRandomNeuron(NumberOfNeurons(layers[n_layer - 1])))
                end
            end
            
            push!(layers, LayerOfNeurons(neurons))
        end
                

        # After adding the hidden layers, each circuit must have an output
        # layer, which will contain _outputs_ neurons

        neurons = Vector{Neuron}()
                
        for n_neuron in 1:outputs
            push!(neurons, GenerateRandomNeuron(NumberOfNeurons(layers[end])))
        end
            
        push!(layers, LayerOfNeurons(neurons))
                
        
        return CircuitOfNeurons(layers)
    end
                
    function GenerateRandomNeuron(inputs::Int64)
        possible_input_lines = randperm(length(collect(1:inputs)))

        num_input_lines = rand(1:length(possible_input_lines))

        possible_input_lines = collect(1:inputs)[possible_input_lines[1:num_input_lines]]
        
        possible_rules = randperm(length(collect(0:inputs))) .- 1

        num_rules = rand(1:length(possible_rules))

        possible_rules = possible_rules[1:num_rules]

        return Neuron(possible_rules, possible_input_lines, 0)
    end

    ## Functions used to print

    function Base.show(io::IO, s::Neuron)
        print(io, "    Neuron\n")
        print("      Rules: {")
        for i in 1 : length(s.rules)
            print(string(sort(s.rules)[i]))
            if i < length(s.rules)
                print(", ")
            end
        end
        print("}\n      Input lines: {")
        for i in 1 : length(s.input_lines)
            print(string(sort(s.input_lines)[i]))
            if i < length(s.input_lines)
                print(", ")
            end
        end
        print("}")
    end
    
    function Base.show(io::IO, s::LayerOfNeurons)
        print(io, "  Layer (" * string(length(s.neurons)) * " neurons)\n")
        for neuron in s.neurons
            println(neuron) 
        end
    end
    
    function Base.show(io::IO, s::CircuitOfNeurons)
        println(io, "Circuit (" * string(length(s.layers)) * " layers)")
        for layer in s.layers
            println(layer) 
        end
    end
end