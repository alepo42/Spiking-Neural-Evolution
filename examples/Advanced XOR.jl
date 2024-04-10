""" 
This source code exploits Genetic Algorithms (GA) in order to 
find a SN P-System circuit that reproduces the behavior
of a XOR function f : {0, 1}^n -> sum_{i=1}^n (xor(x_i))
"""

push!(LOAD_PATH, "src/")

using SNPCircuit
using Utils
using GeneticAlgorithms
using DataFrames

# Optional, if you want to plot the fitness history
using Plots

# I first define the Boolean function that performs the xor between a set of values
function BooleanFunctionParity(inputs::Vector{Bool}) 
    result = inputs[1]
    for i in 2 : length(inputs)
       result = xor(result, inputs[i]) 
    end
    return result
end

# This is the function that will perform a simulation
function Simulation() 
    # This array will contain the history of accuracies
    max_fitness_history = []
        
    # The population of circuits will contain from 80 to 120 individuals
    population = rand((80 : 120))

    # simulations will contain the informations about a certain iteration: id, network and fitness
    simulations = DataFrame(ID = Int64[], Network = CircuitOfNeurons[], Fitness = Float64[])

    # Generating the population of random networks

    min_hidden_layers = UInt16(1)
    max_hidden_layers = UInt16(3)
    inputs = UInt16(length(examples[1]))
    outputs = UInt16(length(labels[1]))

    for i in 1 : population
        random_network = GenerateRandomCircuit(inputs, 
                                            outputs, 
                                            min_hidden_layers, 
                                            max_hidden_layers)

        push!(simulations, (i, random_network, Fitness(random_network, examples, labels)))
    end

    max_iterations = 1000
    max_fitness = maximum(simulations.Fitness)
    iter = 1

    mutation_probabilities = MutationProbabilities(
        0.010,  # New layer
        0.080,  # Remove layer
        0.030,  # New neuron
        0.200,  # Remove neuron
        0.005,  # Add rule
        0.080,  # Remove rules
        0.010   # Random input lines
    )

    #println("Starting simulation!\n")

    #println("(1) Fitness initial: $max_fitness")

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
            #println("($iter) Fitness update : $max_fitness")

            #if max_fitness == 1
            #    break
            #end
        end

        sort!(simulations, :Fitness, rev = true)

        iter += 1
    end

    #println("\nThe simulation stopped after $iter iterations and it reached a level of fitness of $max_fitness")
    #println("The best circuit is the following:\n")
    #println(simulations[1, :].Network)
    
    return max_fitness_history
end

# n controls the number of inputs
n = 5

examples = Vector{Vector{Bool}}()
labels = Vector{Bool}();

combinations = reverse.(Iterators.product(fill(0:1,n)...))[:]

for j in 1 : 2^n
    vector_of_bools = [bitstring(i)[end] == '1' for i in combinations[j]]

    push!(examples, vector_of_bools)
    push!(labels, BooleanFunctionParity(vector_of_bools))
end
    
# At this stage, examples contains all the 2^n combinations, and labels[i] the corresponding
# result of xor(examples[i])

# I run 20 executions, each history will be save in this dictionary
executions = Dict([])

for exec in 1 : 10
    executions[exec] = Simulation()
    println("Execution $exec done (max fitness: " * string(executions[exec][end]) * ")")
end

# Only if Plots has been loaded
if isdefined(Main, :Plots)
    plot(executions[1], title="Max fitness history", label="Execution 1", linewidth = 1.7, legend=:bottomright)
    for i in 2:length(keys(executions))
       plot!(executions[i], label="Execution $i", linewidth = 1.7) 
    end
    xlabel!("Iteration")
    ylabel!("Fitness")
    ylims!(0.5, 1)
end

