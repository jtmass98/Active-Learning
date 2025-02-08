This is a toy example I used to learn the fundementals of active learning. The task is to find the maximum temperature in a room with multiple heat sources. It uses GPytorch to fit the GP to the objective space. In an example, this is the true heatmap:
![Alt Text](True_heatmap.png)

and this is the process of active learning to learn the heatmap:
![Alt Text](heatmap_animation_2.gif) 

The algorithm balances exploiting the peaks of the fitted GP with exploring areas with high uncertainty.
