This is a toy example I used to learn the fundementals of active learning. The task is to find the maximum temperature in a room with multiple heat sources. It uses GPytorch to fit the GP to the objective space. In an example, the true heatmap is:

![Alt Text](True_heatmap.png)

and this is the process of active learning to learn the heatmap and find the maximum:

![Alt Text](heatmap_animation.gif) 

The algorithm balances exploiting the peaks of the fitted GP with exploring areas with high uncertainty.
