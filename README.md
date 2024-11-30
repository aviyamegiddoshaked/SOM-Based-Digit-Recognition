<img width="608" alt="Screenshot 2024-11-30 at 15 29 46" src="https://github.com/user-attachments/assets/6ec7d534-1c41-45b2-ae04-b93333025968"># SOM-Based-Digit-Recognition

### What are SOM's?
Self-Organizing Maps (SOM) are an unsupervised type of artificial neural network designed for clustering and visualizing high-dimensional data. By projecting input data onto a grid of neurons in a lower-dimensional space, they preserve the topological relationships inherent in the data. 

During training, each neuron competes to become the "best matching unit" (BMU) for an input, based on similarity measures such as Euclidean distance. The network adjusts the weights of the neurons iteratively, capturing underlying patterns and structures within the dataset. 

This process creates a spatial map where similar data points are positioned near each other, making SOMs highly effective for tasks like pattern recognition, data visualization, and exploratory analysis.



## Overview

This project focuses on implementing a Self-Organizing Map (SOM) to recognize handwritten digits from a dataset. The aim is to explore and evaluate how effectively the SOM can represent and classify the various digit classes (0-9). The provided executable, designed for Windows, offers an interactive visualization, showcasing the learned representations of the digits.


## Initialization of Weights

The neuron weights were initialized using statistical properties of the dataset. Specifically, each neuron was initialized as:

The average of all data points (per pixel) + the standard deviation (per pixel) * RND,  

where **RND** is a random number uniformly drawn for each pixel (in each neuron) within the range [-0.2, 0.2].

This method ensures that each neuron starts with a meaningful value close to the overall structure of all digits while introducing symmetry-breaking through the random values.



## Training Process: Adjusting the BMU and Neighboring Neurons

### Explanation

The training process for the Self-Organizing Map (SOM) consists of two main steps: identifying the **Best Matching Unit (BMU)** and updating the weights of the BMU and its neighboring neurons.

1. **Finding the BMU**  
   For each input data point, the Euclidean distance is calculated between the input and all neurons in the network. The neuron with the smallest distance is identified as the BMU.

2. **Updating the Weights**  
   - The Euclidean distance between the BMU and every other neuron is computed to determine the level of influence the BMU has on its neighbors.
   - The weights are updated based on this influence, which decays exponentially with distance, following the formula:  
     ```math
     \text{influence} = e^{-\frac{\text{dist}^2}{2\sigma^2}}
     ```
     where:
     - `dist` is the distance from the BMU to the neuron being updated.
     - `σ` is a parameter controlling the radius of influence.

   - The weight update for a neuron is performed using the formula:  
     ```math
     W[i] += α ⋅ influence ⋅ (x - W[i])
     ```
     where:
     - `i` is the index of the weight being updated.
     - `x` is the input data point.
     - `α` is the learning rate.

3. **Dynamic Learning Parameters**  
   To ensure convergence, both the learning rate (`α`) and the influence radius (`σ`) are gradually reduced over time. The decay is computed exponentially as follows:  
   ```math
   α_t = 0.005 + (α_0 - 0.005) ⋅ e^{-\frac{5t}{n}}
   ```math
σ_t = σ_0 ⋅ e^{-\frac{t}{n/2}}


where:

n is the total number of iterations.
t is the current iteration index.
This systematic approach allows the SOM to refine its representation of the data iteratively, ensuring that the BMU and its neighbors adapt appropriately to the input patterns.


### Visualization Strategy


