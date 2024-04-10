"""
    Utils

A module providing several useful functions used by
other modules

"""

module Utils

    # Functions that are exported to the user
    export MutationProbabilities,
           Differences 

    """
    MutationProbabilities
        
    A struct containing the probabilities to perform a mutation
        
    # Fields
    - `new_layer::Float64`: The probability to add a new layer
    - `remove_layer::Float64`: The probability to remove a random layer
    - `new_neuron::Float64`: The probability to add a new random neuron
    - `remove_neuron::Float64`: The probability to add a remove a random neuron
    - `new_rule::Float64`: The probability to add a new rule to a random neuron
    - `remove_rule::Float64`: The probability to remove a rule from a random neuron
    - `random_input_lines::Float64`: The probability to sample new input lines of a random neuron
    """
    mutable struct MutationProbabilities
        new_layer::Float64
        remove_layer::Float64
        new_neuron::Float64
        remove_neuron::Float64
        new_rule::Float64
        remove_rule::Float64
        random_input_lines::Float64
    end

    """ 
    Differences(inputs, current_input_lines)

    Finds the invalid input lines

    # Arguments
    - `inputs::UInt16`: The actual inputs received by the neuron
    - `current_input_lines::Vector{UInt16}`: The set of input lines of the neuron.

    # Returns
    - `Vector{UInt16}`: The set of invalid input lines`.
    """
    function Differences(inputs::UInt16, current_input_lines::Vector{UInt16})
        d = Vector{UInt16}()

        for input_line in current_input_lines
            # If the value of input line is larger than the number of inputs,
            # than this line is considered invalid (and should be removed)
            if input_line > inputs
                push!(d, input_line) 
            end
        end

        return d 
    end

end #Module