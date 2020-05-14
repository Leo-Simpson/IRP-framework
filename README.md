# Revisited Inventory Routing problem

Here we approximate the solution of an inventory routing problem, and compute a nice visualization of it. 


## Summary

The user have to enter the routes that the algorithm will consider (with the quantity of food that they should deliver at each point... for now).

On each day, the algorithm  minimizes the cost per quantity delivered (it is not necessarly the optimal solution after a couple of day, the cost is minimized on each day SEPARATELY) with the constraint of capactiy of customers and with the constraint of always having non-negative inventories.

The code can then generate an html page with the graphic. 

A simple example is also provided. 
