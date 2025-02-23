This is a toy active learning project. The task is to find the maximum temperature in a room with multiple heat sources. In an example generated using the code, the true heatmap is:

![Alt Text](True_heatmap.png)

The temperature is multi-modal and has stationary smoothness. The animation below shows the process of active learning to learn the temperature distribution and find the maximum:

![Alt Text](heatmap_animation.gif) 

The algorithm balances exploiting the peaks of the GP, fitted using GPytorch, with exploring areas with high uncertainty. Note that there is a uniform distribution of test points which is representative of the smoothness of the true function. There are additional points close to the maximum to try to pinpoint it. Given more points, the algorithm would likely further explore the maximum to find further gains.
