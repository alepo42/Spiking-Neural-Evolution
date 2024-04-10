# Spiking Neural Evolution

This repository contains the source code for the work-in-progress paper _An Evolutionary Approach to the Design of Spiking Neural P Circuits_

# How to use

The source code offers a high-level API that makes it easy to run the simulations on a preferred Boolean function.

First of all, include the library: 

```
using SpikingNeuralEvolution
```

Then, define a Boolean function to be approximated. For instance, the XOR function:
```
function XOR(inputs::Vector{Bool}) 
    return reduce(xor, inputs)
end
```

How many input values should be used? Choose carefully, as the dimension of the truth table grows exponentially ($2^n$)
```
n = 5
```


Then, launch the simulations with a single line:
```
Simulate(XOR, n)
```
