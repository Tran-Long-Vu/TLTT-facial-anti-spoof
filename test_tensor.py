device = "CPU" # or "GPU"    
if device == "CPU":
    provider = "CPUExecutionProvider"
    print("cpu")
elif device == "GPU":
    provider = "CUDAExecutionProvider"
    print("gpu")
else: 
    print("device issue")