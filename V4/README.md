V4 Minimal Static NN (Vitis-friendly)

This version is intentionally simple to be built upon by future researchers:
- Dense layers only
- Activations: linear, tanh, sigmoid, relu
- Loss: MSE
- Optimizer: plain SGD
- Training modes: baseline SGD and sequential spec1
- Static memory in core library (fixed compile-time caps)

Build:
  make