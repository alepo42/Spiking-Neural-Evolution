module SpikingNeuralEvolution

    @enum StoppingCriteria MaxIterations=0 Threshold=1   
    
    include("Utils.jl")
    using .Utils

    include("GeneticAlgorithms.jl")
    using .GeneticAlgorithms
    using .GeneticAlgorithms.SNPCircuit

    using DataFrames

    export Evolve,
           EvolutionParameters,
           MutationProbabilities,
           StoppingCriteria


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
        stopping_criteria::StoppingCriteria
        percentage_of_examples::Float64
    end

    #TODO da commentare
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
        
        sort!(simulations, :Fitness, rev = true)

        best_circuit = simulations[1, :].Network

        iter = 1

        # The iteration in which the maximum (1 or threshold) fitness has been found
        iteration_of_max_fitness = -1
    
        while iter < max_iterations
            elitism_percentage = 0.10
    
            new_circuits = simulations[1 : Int64(round(elitism_percentage * population)), :].Network
    
            for i in elitism_percentage * population : population
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
    
            sort!(simulations, :Fitness, rev = true)

            if simulations[1, :].Fitness > max_fitness
                max_fitness = simulations[1, :].Fitness
                best_circuit = deepcopy(simulations[1, :].Network)
            end

            if max_fitness == 1 && iteration_of_max_fitness == -1
                iteration_of_max_fitness = iter
            end
        
    
            iter += 1
        end
        
        return best_circuit, max_fitness_history, Int16(iteration_of_max_fitness), simulations
   end
    
   #TODO da commentare
   function Evolve(f::Function, inputs::UInt16, evolution_parameters::EvolutionParameters, mutation_parameters::MutationProbabilities, verbose = false)
        if inputs > 2^10
            println("Warning: using a large input space (2^$inputs = $(2^inputs))")
        end

        # This vector will contain all the combinations 2^inputs 
        examples = Vector{Vector{Bool}}()
        labels = Vector{Bool}()

        # This line creates a vector containing all possible combinations (2^n) of _inputs_ bits
        combinations = reverse.(Iterators.product(fill(0 : 1, inputs)...))[:]

        for j in 1 : Int(round((2 ^ inputs) * evolution_parameters.percentage_of_examples))
            combination = [bitstring(i)[end] == '1' for i in combinations[j]]

            push!(examples, combination)
            push!(labels, f(combination))
        end

        executions = Dict([])

        best_circuit = 0
        best_fitness = 0

        for exec in 1 : evolution_parameters.simulations
            executions[exec] = Simulate(examples, labels, evolution_parameters, mutation_parameters)
            
            if (executions[exec][2][end] > best_fitness) 
                best_fitness = executions[exec][2][end]
                best_circuit = executions[exec][1]
            end

            if verbose println("Execution $exec done (max fitness: " * string(executions[exec][2][end]) * ") in " * string(executions[exec][3]) * " iterations.") end
        end

        println("I found a circuit with " * string(NumberOfLayers(best_circuit)) * " layers with a fitness of $best_fitness")

        return executions
    end

    function Evolve(f::Function, inputs::UInt16, verbose = false)
        return Evolve(f, 
                    inputs,
                    EvolutionParameters(
                        UInt32(5),      # Simulations
                        UInt32(1000),   # Iterations per simulation
                        UInt32(80),     # Min number of random population
                        UInt32(120),    # Max number of random population
                        UInt16(1),      # Min number of random hidden layers
                        UInt16(3),      # Max number of random hidden layers
                        MaxIterations,  # Stopping criteria
                        1               # Percentage of input combinations examples
                    ),
                    MutationProbabilities(
                        0.008,  # New layer
                        0.080,  # Remove layer
                        0.030,  # New neuron
                        0.200,  # Remove neuron
                        0.005,  # Add rule
                        0.080,  # Remove rules
                        0.010   # Random input lines
                    ),
                    verbose)
    end

    function Evolve(f::Function, inputs::Int, verbose = false)
        return Evolve(f, 
                    UInt16(inputs),
                    EvolutionParameters(
                        UInt32(5),     # Simulations
                        UInt32(1000),  # Iterations per simulation
                        UInt32(80),    # Min number of random population
                        UInt32(120),   # Max number of random population
                        UInt16(1),     # Min number of random hidden layers
                        UInt16(3),      # Max number of random hidden layers
                        MaxIterations,
                        1
                    ),
                    MutationProbabilities(
                        0.008,  # New layer
                        0.080,  # Remove layer
                        0.030,  # New neuron
                        0.200,  # Remove neuron
                        0.005,  # Add rule
                        0.080,  # Remove rules
                        0.010   # Random input lines
                    ),
                    verbose)
    end

    function Evolve(f::Function, inputs::UInt16, evolution_parameters::EvolutionParameters, verbose = false)
        return Evolve(f, 
                    inputs,
                    evolution_parameters,
                    MutationProbabilities(
                        0.008,  # New layer
                        0.080,  # Remove layer
                        0.030,  # New neuron
                        0.200,  # Remove neuron
                        0.005,  # Add rule
                        0.080,  # Remove rules
                        0.010   # Random input lines
                    ),
                    verbose)
    end

end