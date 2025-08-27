# 🍰 Neural Networks from Scratch: The Cheesecake Recipe
Welcome\! Get yourself a slice of cheesecake, and let's start our journey to understand neural networks without any frameworks—just pure math and logic.

This project is the code implementation of the idea that a Neural Network is just a mathematical function, as simple and layered as a cheesecake. We'll build a simple neural network from the ground up to understand what really happens behind frameworks like TensorFlow or PyTorch.

## The Analogy: A Cheesecake Network

Just like a cheesecake, our Neural Network has layers and a recipe that we need to perfect.

  * **Layers:**
      * **Input Layer:** The crunchy biscuit base.
      * **Hidden Layers:** The rich, creamy cheese filling.
      * **Output Layer:** The delicious topping that tells you the final flavor.
  * **Neurons:** The fundamental ingredients (cream cheese, sugar, eggs) that make up each layer.
  * **The Recipe (Learning Process):** The step-by-step method to create the perfect cheesecake.

-----

## How It Works: The Recipe for Learning

Our code follows a simple recipe to learn from data.

### 1\. Initializing Parameters — *Choose Your Flavour*

We start by choosing our "flavour"—randomly initializing the weights ($W$) and biases ($B$) for our network's neurons. These are the initial guesses for our recipe's ingredient measurements. We'll perfect them as we train.

### 2\. Forward Propagation — *Mix the Cream Cheese*

This is where we mix our ingredients and make the filling. We take our inputs ($X$), process them through the network's layers, and produce an output.

For each neuron in a layer, we calculate a weighted sum:

$$Z = W \cdot X + B$$

Where:

  * $W$ = Weights
  * $B$ = Bias
  * $X$ = Input from the previous layer

Then, we pass this sum through an **Activation Function** to decide how much this neuron "fires" or contributes to the next layer. It's like checking if our mix has the right creamy texture.

Common activation functions implemented here are:

  * **ReLU (Rectified Linear Unit):** Hides the small aftertastes and avoids the "vanishing gradients" problem.
    $$
    $$$$\ \text{ReLU}(z) = \max(0, z)
    $$
    
  * **Sigmoid:** Squashes the neuron's vote into a probability between 0 and 1. *Will the cheesecake be sweet or not?*
    $$
    $$$$ \text{Sigmoid}(z) = \frac{1}{1 + e^{-z}}
    $$
    
  * **SoftMax:** Used in the final layer for multi-class problems, it gives the probability of the input belonging to each class.

### 3\. Backward Propagation — *Taste the Cake and Find Mistakes*

No recipe is perfect on the first try. Our network makes a prediction, and we compare it to the correct answer to calculate the **error** (or "loss").

Backward Propagation is the process of tasting our cheesecake, realizing it's not quite right, and figuring out which ingredient (or parameter) caused the mistake. We trace the error backward from the output layer to the input layer, calculating how much each weight and bias contributed to the total error.

### 4\. Gradient Descent — *Adjust the Recipe*

Once we know which parameters are responsible for the error, we need to adjust them. **Gradient Descent** is the technique we use to make small, smart updates to our weights and biases.

It's like tweaking the amount of sugar or changing the baking time to minimize the error and move closer to the perfect, lip-smacking cheesecake recipe. We repeat this process (Forward Prop -\> Backprop -\> Gradient Descent) many times until our network becomes a master chef, capable of making accurate predictions.

-----

## Project Structure

```
.
├── data/
│   └── (Your dataset files go here)
├── neural_network.py   # The core class for the Neural Network
├── main.py             # Script to load data, train the network, and make predictions
└── README.md           # You are here!
```

-----

## How to Run

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/dhairya-pandya/NeuralNets_FromScratch.git
    cd your-repo-name
    ```

2.  **Install dependencies** (this project likely only needs `numpy`):

    ```bash
    pip install numpy
    ```

3.  **Run the training script:**

    ```bash
    python main.py
    ```

-----

## Conclusion

This project aims to demystify neural networks by showing that at their core, they are a combination of simple mathematical operations and a logical learning process. By building one from scratch, we can truly appreciate the "magic" behind modern AI.

I hope you enjoyed this effort to explain a complex topic with a tasty analogy\!

Happy and Tasty Learning\! 🍰