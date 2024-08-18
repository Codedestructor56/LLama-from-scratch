import tensor_module  

def test_tensor():
    # Create tensors
    shape = [2, 3]
    tensor = tensor_module.TensorFLOAT32(shape)
    
    # Initialize with random values
    tensor.rand(shape)
    
    # Print tensor
    print("Initial Tensor:")
    print(tensor)
    
    # Modify tensor values
    tensor.set([0, 1], 5.0)
    print("Tensor after setting value at [0, 1]:")
    print(tensor)
    
    # Get a value
    value = tensor.get([0, 1])
    print(f"Value at [0, 1]: {value}")
    
    # Reshape tensor
    new_shape = [3, 2]
    tensor.reshape(new_shape)
    print("Tensor after reshaping:")
    print(tensor)
    
    # Test tensor operations
    tensor_ones = tensor_module.TensorFLOAT32.ones(new_shape)
    tensor_sum = tensor + tensor_ones
    print("Tensor after adding ones tensor:")
    print(tensor_sum)

if __name__ == "__main__":
    test_tensor()
