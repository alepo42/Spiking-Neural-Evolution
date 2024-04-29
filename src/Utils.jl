"""
    Utils

A module providing several useful functions used by
other modules

"""

module Utils
    using Combinatorics

    # Functions that are exported to the user
    export MutationProbabilities,
           Differences,
           GenerateRandomANFFunction,
           EvaluateANFFunction

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

    """
    GenerateRandomANFFunction(degree::Int64)

    Generate a random Algebraic Normal Form (ANF) function.

    # Arguments
    - `degree::Int64`: The degree of the ANF function, which determines the number of terms in the function.

    # Returns
    - `terms::Vector{Any}`: A vector representing the terms of the generated ANF function.

    The function generates a random ANF function by creating a list of terms. It starts by initializing an empty list `terms`. Then, for each possible combination of input variables (from 1 to `degree`), it randomly decides whether to include the combination in the ANF function. After considering all possible combinations, if the function doesn't include the term representing all input variables, it adds it with a probability of 0.5. Finally, it randomly decides whether to include a constant term (true or false) in the ANF function.

    """
    function GenerateRandomANFFunction(degree::UInt16) 
        terms = []
        
        n = degree
        elements = 1:n
        
        for i in 1:length(elements)
            for comb in combinations(elements, i)
                # Decido se considerare la combinazione
                if rand() > 0.5
                   push!(terms, comb) 
                end
            end
        end
        
        if (elements in terms) == false
           push!(terms, collect(elements)) 
        end
        
        if rand() > 0.5
            push!(terms, true)
        else
            push!(terms, false)
        end
        
        return terms
    end
    
    """
    EvaluateANFFunction(inputs::Vector{Bool}, terms::Vector{Any})

    Evaluate an Algebraic Normal Form (ANF) function.
    The function iterates through each term in `terms`, computing the partial result by applying 
    logical AND operation between the input variables specified by the indices in each term. 
    Then, it computes the final result by applying the XOR operation between the partial 
    results of each term.

    # Arguments
    - `inputs::Vector{Bool}`: A vector of boolean values representing the input variables.
    - `terms::Vector{Any}`: A vector of terms, where each term is represented as a vector 
                            of indices corresponding to the input variables.

    # Returns
    - `result::Bool`: The result of evaluating the ANF function.

    """
    function EvaluateANFFunction(inputs::Vector{Bool}, terms::Vector{Any})
        result = false
        
        for i in 1:length(terms) - 1
            
            partial_result = inputs[terms[i][1]]
            
            for j in 2:length(terms[i])
                partial_result = partial_result && inputs[terms[i][j]]
            end
            
            if i == 1
                result = partial_result
            else
                result = xor(result, partial_result)
            end
        end
        
        result = xor(result,terms[end])
        
        return result
    end

    

end #Module