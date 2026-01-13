def pitts_neuron(inputs, weights, threshold):
    weighted_sum = sum(i*w for i, w in zip(inputs, weights))
    output = 1 if weighted_sum >=threshold else 0
    return output

inputs = [1, 0]
weights = [1, 1]
threshold = 2 #for AND 
# =1 for OR
# =3 for NAND

output = pitts_neuron(inputs, weights, threshold)
print(f"Neuron output: {output}")  