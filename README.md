🧠 Build-MyOwn-Deep_Learning-From_Scratch

Building an Artificial Intelligence and autodiff engine absolutely from scratch in pure Python. No black boxes.

🎯 The Goal

The idea here is simple: learn Artificial Intelligence by coding the underlying mathematics. Instead of importing PyTorch and just calling a built-in method, I am writing the engine that builds the computational graph, computes the gradients, and propagates the error.

Beyond Machine Learning, this project marks a technical evolution in my journey: I am bringing my software engineering and performance background from C++ (where I built projects like TitanDB and MyOwnRedis-cpp) to master Fluent Python. This involves the advanced use of Object-Oriented Programming and Magic Methods (__add__, __rmul__, etc.) to make my classes behave like native numerical types in the language.

🧱 Engine Roadmap

Development follows a strict rule: no shortcuts. The core engine uses only the Python standard library.

  01 Scalar Values: Value: wrap one number; forward arithmetic via operator overloading.
  
  02 Computational Graph: Record each result's parents (_prev set), op (_op), and a no-op _backward hook.
  
  03 Local Derivatives: Install per-op _backward closures (the local-derivative push).
  
  04 Chain Rule: backward(): topological sort + seed grad=1 + reverse-walk the closures.
  
  05 Backprop Engine: Add tanh, exp, relu on the scalar Value; the complete micrograd engine.
  
  06 Vector Operations: A Vec container of Values: elementwise ops, dot, sum.
  
  07 Matrix Operations: A Mat of Values: matmul/@, transpose, reshape, sum, mean.
  
  08 Tensor Engine: Collapse scalar graphs onto one N-dim NumPy-backed autodiff Tensor.
  
  09 Neuron: A single learnable neuron y = phi(x @ w + b) on the Tensor.
  
  10 Dense Layer: Vectorized fully-connected layer Z = X @ W + b.
  
  11 MLP: Stack Dense layers with activations between them.
  
  12 Loss Functions: Single-example MSE/MAE/cross-entropy (+ stable softmax) and sum/mean reductions.
  
  13 Loss Functions (Batched): Lift softmax/cross-entropy to a (B, C) batch; mean over the batch.
  
  14 SGD Optimizer: The Optimizer/SGD update step abstraction.
  
  15 First Training Loop: Wire MLP + loss + SGD into the canonical learn loop.
  
  16 Weight Initialization: Xavier/Glorot and He/Kaiming init, and why scale matters.
  
  17 Momentum: SGD with momentum.
  
  18 Adam: RMSProp/Adam with bias correction and weight decay.
  
  19 Batch Training: Minibatching, epochs, shuffling; gradient-variance intuition.
  
  20 DataLoader: Dataset/DataLoader batching abstraction.
  
  21 Regularization: L2 / weight decay; train vs eval mode.
  
  22 Dropout: Dropout forward + backward; inverted scaling.
  
  23 BatchNorm: Batch normalization forward + backward by hand.
  
  24 Conv2D Math: Convolution arithmetic and gradients via im2col.
  
  25 Conv2D Implementation: Conv2D/pooling/flatten as Tensor layers.
  
  26 CNN Project: Stack conv/pool/linear; train on image data.
  
  27 Attention Math: Scaled dot-product attention forward + backward (pure NumPy).
  
  28 Self-Attention: Self-attention on the Tensor autodiff engine.
  
  29 Multi-Head Attention: Split/concat heads; the full MHA module.
  
  30 Transformer: Residuals + LayerNorm + FFN; a full Transformer block.
  
  31 Vision Transformer: Patch embeddings + Transformer for image classification.
  
  32 Framework Refactor: Package it all into a clean PyTorch-like Module/Parameter API.
  
  33 Capstone - MNIST: End-to-end MNIST classifier.
  
  34 Capstone - CIFAR-10: End-to-end CIFAR-10 classifier.
  
  35 Capstone - Transformer: End-to-end Transformer language model.

⚙️ Structure and How to Run

The core engine is structured inside the src/ directory, while test-driven validation (TDD) is restricted to the tests/ folder.

To run the project locally (via WSL/Linux):

Clone the repository:

git clone https://github.com/SEU-USUARIO/Build-MyOwn-Deep_Learning-From_Scratch.git
cd Build-MyOwn-Deep_Learning-From_Scratch


Create and activate the virtual environment:

python3 -m venv venv
source venv/bin/activate


Install the testing dependencies:

pip install pytest numpy


Run the test suite:

python -m pytest


"What I cannot create, I do not understand." — Richard Feynman
