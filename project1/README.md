
The decision tree functions were implemented recursively, since recursion is excellent for solving problems that are broken down into smaller and often times repetitive
problems. Tree data structure, and by extension node data structure was used for the implementation, a choice of personal satisfaction. Other data structures such as 
arrays, linked lists, stacks, and queues are linear data structures which wouldn't work well for a decision tree model. The other non-linear data structure graph would 
presents other complications, such as not having a unique root node, and it's application for finding the shortest path as opposed to the objective of a decision tree.

Individual trees might have such variance in their accuracy because decision trees in general has high variance but low bias. As the tree is traversed, data is split over
and over, which means that the actual predications are also made by less and less data points. In addition, it can depend on how deep a tree is, since shallow trees aren't 
checking a lot of conditions, so over-fitting is not occurring as much. This is unlike the random forest as a whole, which generalizes better as it
combines the results or predictions of multiple decision trees, which are less-correlated from being randomized. Variance can be reduced and accurracy can be be improved
by adding bias since it is known that there is a relationship known as the bias-variance tradeoff. More training data can overcome the difference between cross-validation
error and training set error, or the pruning method can be applied to a tree, which reduces the decision tree's size.

The random forest benefits from using an odd number of individual trees because in binary responses, each decision tree gives one prediction(whether it be yes/no, 1/0, etc.).
All those predications are then recorded, with the majority being the final prediction, so an odd number of trees would prevent potential ties. With an even number of trees,
either external intervention or some sort of tie breaker must be implemented. 

One of the challenges/struggles to working with Python is that its basic syntax, or the rules that govern the programming language are considerably different than 
other familiar languages such as C/C++ or Java. For example, in C++ every variable must be declared with it's type before its first use, while Python requires 
neither variable declaration before use or their type. Another challenge was implementing the decision tree using nodes in Python. Not being familiar with Python meant
having to spend additional time researching, as it was similar in concept, but different in implementation to other languages. In terms of general obstacles faced 
however, the subject of data structures had been taught but forgotten with time, so a refresher was required before even approaching the project. A realization that 
occurred while working was that implementing a decision tree in code proved to be much harder than expected, despite having the appropriate knowledge on how to 
perform/solve one by hand. 
