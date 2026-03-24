Each tile will generate troops every turn
Fertile : 3
Plains : 2 
Mountains : 1

We will keep track of which tile has how many troops

The logic for winning in a match is also different for different tiles
The your_n_troops > Defense_bonus*(1+ opoonent_ntroops + root(oppoonent_n_troops))
if it is between pponent troops and 1+ opoonent_ntroops + root(oppoonent_n_troops), there is a random chance of it succeeding which linearly depeends from near 0 to near 1 and is exactly 0 and 1 at the end points

The players will select their own starting point. 

The defense bonus depends on terrain it is 1 for plains and fertile and 2 for mountains 

