# This source code exploits Genetic Algorithms (GA) in order to 
# find a SN P-System circuit that reproduces the behavior
# of a XOR function f : {0, 1}^n -> sum_{i=1}^n (xor(x_i))
using DataFrames


# I first define the Boolean function that performs the xor between a set of values
function BooleanFunctionParity(inputs::Vector{Bool}) 
    result = inputs[1]
    for i in 2 : length(inputs)
       result = xor(result, inputs[i]) 
    end
    return result
end

# n controls the number of inputs
n = 4

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

# This array will contain the history of accuracies
accuracies = []
    
# The population of circuits will contain 100 individuals
population = 100

# simulations will contain the informations about a certain iteration: id, network and fitness
simulations = DataFrame(ID = Int64[], Network = CircuitOfNeurons[], Fitness = Float64[])

# Generating the population of random networks

min_hidden_layers = 1
max_hidden_layers = 3

for i in 1:population
    random_network = GenerateRandomNetwork(length(examples[1]), length(labels[1]), min_hidden_layers, max_hidden_layers)
    push!(simulations, (i, random_network, Fitness(random_network, examples, labels)))
end

# Sorting the simulations wrt the value of fitness
sort!(simulations, :Fitness, rev = true)

# Adding the value of max accuracy to the history of accuracies
push!(accuracies, maximum(simulations.Fitness))

# Updating the value of max accuracy
max_accuracy = maximum(simulations.Fitness)

# Running the experiment for 500 iterations
for iteration in 1:500

    elitarism_percentage = 0.10
    
    global simulations, max_accuracy

    new_circuits = simulations[1 : Int64(round(elitarism_percentage * population)), :].Network

    for i in elitarism_percentage * population : population
        push!(new_circuits, 
                    CleanCircuit(Mutation(CrossoverHorizontal(
                        simulations[ProportionateSelection(simulations.Fitness), :].Network, 
                        simulations[ProportionateSelection(simulations.Fitness), :].Network))))     
    end

    simulations = DataFrame(Network = CircuitOfNeurons[], Fitness = Float64[])

    for i in 1:population
        push!(simulations, (new_circuits[i], Fitness(new_circuits[i], examples, labels)))
    end

    push!(accuracies, maximum(simulations.Fitness))

    if maximum(simulations.Fitness) > max_accuracy
        max_accuracy = maximum(simulations.Fitness)
        println(max_accuracy)
    end

    sort!(simulations, :Fitness, rev = true)
end

println(max_accuracy)
