
### What are SOM's?
Self-Organizing Maps (SOM) are an unsupervised type of artificial neural network designed for clustering and visualizing high-dimensional data. By projecting input data onto a grid of neurons in a lower-dimensional space, they preserve the topological relationships inherent in the data. 

During training, each neuron competes to become the "best matching unit" (BMU) for an input, based on similarity measures such as Euclidean distance. The network adjusts the weights of the neurons iteratively, capturing underlying patterns and structures within the dataset. 

This process creates a spatial map where similar data points are positioned near each other, making SOMs highly effective for tasks like pattern recognition, data visualization, and exploratory analysis.





<img width="608" alt="Screenshot 2024-11-30 at 15 29 46" src="https://github.com/user-attachments/assets/6ec7d534-1c41-45b2-ae04-b93333025968">








## Overview

This project focuses on implementing a Self-Organizing Map (SOM) to recognize handwritten digits from a dataset. The aim is to explore and evaluate how effectively the SOM can represent and classify the various digit classes (0-9). The provided executable, designed for Windows, offers an interactive visualization, showcasing the learned representations of the digits.


## Initialization of Weights

The neuron weights were initialized using statistical properties of the dataset. Specifically, each neuron was initialized as:

The average of all data points (per pixel) + the standard deviation (per pixel) * RND,  

where **RND** is a random number uniformly drawn for each pixel (in each neuron) within the range [-0.2, 0.2].

This method ensures that each neuron starts with a meaningful value close to the overall structure of all digits while introducing symmetry-breaking through the random values.


## Visualization Strategy
Each neuron in the SOM was visualized as an image, labeled in the format “Neuron (x, y) | Label: z,” where:

- `x` and `y` denote the neuron's coordinates within the SOM grid.
- `z` indicates the neuron's dominant label, representing the digit it most frequently corresponds to.

To improve visualization, distinct colors were assigned to each digit, making it easier to identify neuron clusters and their representation of different digits.



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
   α_t = 0.005 + (α_0 - 0.005) ⋅ e^{-\frac{5t}{n}} = σ_0 ⋅ e^{-\frac{t}{n/2}}


where:

- n is the total number of iterations.
- t is the current iteration index.
- This systematic approach allows the SOM to refine its representation of the data iteratively, ensuring that the BMU and its neighbors adapt appropriately to the input patterns.

## Error Metrics

The best solution is selected based on a combination of the following metrics:

1. **Quantization Error**:  
   This metric measures the average Euclidean distance between the input data points and the corresponding neuron weights. Lower values indicate that all input data, particularly from all different classes, are well-represented by the weights. For example, in a scenario where certain digits lack sufficient representation, this metric will yield higher error values.

2. **Topographic Error**:  
   This metric evaluates the preservation of neighborhood relationships between weights. In other words, it measures the smoothness of transitions between regions representing different classes in the weights. The error increases when there are more such transitions or when there are "blurred" neurons that do not belong distinctly to any single digit class.

Minimizing these metrics ensures that all classes are well-represented by the weights, transitions between them are smoother, and every neuron is clearly associated with a specific digit class.

## How to Run the SOM Program

**Note: This program is designed to run on Windows!**


1. Download the `main.exe` file along with the following data files:
   - `digits_test.csv` - The input data containing 10,000 grayscale images of digits (0-9), each represented as a 28x28 pixel matrix.
   - `digits_test_key.csv` - The labels for each image (used for evaluation after training)
   -  Make sure you have the necessary Python libraries installed. You can install them using the following command:

pip install numpy matplotlib pandas

2. Place all files in the same directory.

3. Open a **CMD window** in the directory containing the files and run the following command:
   ```cmd
   main.exe
   ```

4. Upon completion, the final error metrics will be printed, and the neurons will be displayed. 



