module SpikingNeuralEvolution

    
    include("Utils.jl")
    using .Utils

    include("GeneticAlgorithms.jl")
    using .GeneticAlgorithms
    using .GeneticAlgorithms.SNPCircuit

    

    using DataFrames

    export Evolve,
           EvolutionParameters


    """
    EvolutionParameters

    This struct is a container of options that can be set when asking for
    an evolution

    # Fields
    - `simulations::UInt32`: The number of simulations to be performed
    """
    struct EvolutionParameters
        simulations::UInt32
        iterations_per_simulation::UInt32
        population_min::UInt32
        population_max::UInt32
        random_circuit_min_layers::UInt16
        random_circuit_max_layers::UInt16
    end

    function Simulate(examples::Vector{Vector{Bool}}, labels::Vector{Bool}, evolution_parameters::EvolutionParameters, mutation_probabilities::MutationProbabilities)
        # This array will contain the history of accuracies
        max_fitness_history = []
           
        # The population of circuits will contain from 80 to 120 individuals
        population = rand((evolution_parameters.population_min : evolution_parameters.population_max))
    
        # simulations will contain the informations about a certain iteration: id, network and fitness
        simulations = DataFrame(ID = Int64[], Network = CircuitOfNeurons[], Fitness = Float64[])
    
        # Generating the population of random networks
        min_hidden_layers = evolution_parameters.random_circuit_min_layers
        max_hidden_layers = evolution_parameters.random_circuit_max_layers
   
        inputs = UInt16(length(examples[1]))
        outputs = UInt16(length(labels[1]))
    
        for i in 1 : population
            random_network = GenerateRandomCircuit(inputs, 
                                                outputs, 
                                                min_hidden_layers, 
                                                max_hidden_layers)
    
            push!(simulations, (i, random_network, Fitness(random_network, examples, labels)))
        end
    
        max_iterations = evolution_parameters.iterations_per_simulation
        max_fitness = maximum(simulations.Fitness)
   
        iter = 1
    
        while iter < max_iterations
            elitarism_percentage = 0.10
    
            new_circuits = simulations[1 : Int64(round(elitarism_percentage * population)), :].Network
    
            for i in elitarism_percentage * population : population
                push!(new_circuits, 
                            CleanCircuit(Mutation(Crossover(
                                simulations[ProportionateSelection(simulations.Fitness), :].Network, 
                                simulations[ProportionateSelection(simulations.Fitness), :].Network,
                                inputs), inputs, mutation_probabilities), inputs))     
            end
    
            simulations = DataFrame(Network = CircuitOfNeurons[], Fitness = Float64[])
    
            for i in 1:population
                push!(simulations, (new_circuits[i], Fitness(new_circuits[i], examples, labels)))
            end
    
            push!(max_fitness_history, maximum(simulations.Fitness))
    
            if maximum(simulations.Fitness) > max_fitness
                max_fitness = maximum(simulations.Fitness)
            end
    
            sort!(simulations, :Fitness, rev = true)
    
            iter += 1
        end
        
        return max_fitness_history
   end
    
   function Evolve(f::Function, inputs::UInt16, evolution_parameters::EvolutionParameters, mutation_parameters::MutationProbabilities)
        if inputs > 2^10
            println("Warning: using a large input space (2^$inputs = $(2^inputs))")
        end

        # This vector will contain all the combinations 2^inputs 
        examples = Vector{Vector{Bool}}()
        labels = Vector{Bool}()

        # This line creates a vector containing all possible combinations (2^n) of _inputs_ bits
        combinations = reverse.(Iterators.product(fill(0 : 1, inputs)...))[:]

        for j in 1 : (2 ^ inputs)
            combination = [bitstring(i)[end] == '1' for i in combinations[j]]

            push!(examples, combination)
            push!(labels, f(combination))
        end

        executions = Dict([])

        for exec in 1 : evolution_parameters.simulations
            executions[exec] = Simulate(examples, labels, evolution_parameters, mutation_parameters)
            println("Execution $exec done (max fitness: " * string(executions[exec][end]) * ")")
        end

        histories = collect(sort(executions))
        return [collect(pair) for pair in histories]
    end

    function Evolve(f::Function, inputs::UInt16)
        return Evolve(f, 
                    inputs,
                    EvolutionParameters(
                        UInt32(5),    # Simulations
                        UInt32(500),   # Iterations per simulation
                        UInt32(80),    # Min number of random population
                        UInt32(120),   # Max number of random population
                        UInt16(1),     # Min number of random hidden layers
                        UInt16(3)      # Max number of random hidden layers
                    ),
                    MutationProbabilities(
                        0.010,  # New layer
                        0.080,  # Remove layer
                        0.030,  # New neuron
                        0.200,  # Remove neuron
                        0.005,  # Add rule
                        0.080,  # Remove rules
                        0.010   # Random input lines
                    ))
    end

end