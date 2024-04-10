module SpikingNeuralEvolution

    using SNPCircuit
    using Utils
    using GeneticAlgorithms
    using DataFrames

    export Simulate

    
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
    function Simulate(f::Function, inputs::UInt16)

    end

end