function Differences(previous_layer_neurons::Int64, current_input_lines::Vector{Int32})
    d = []
    for input_line in current_input_lines
        if input_line > previous_layer_neurons
           push!(d, input_line) 
        end
    end
    return d 
end