# dualp
Master thesis on adaptive computational strategy (System1/System2) for deep neural networks. 
![poster](figures/poster.png)
The dual-process theory states that human cognition operates in two distinct modes: one
for rapid, habitual and associative processing, commonly referred to as "system 1", and the
second, with slower, deliberate and controlled processing, which we call "system 2". This
distinction points to an important underlying feature of human cognition: the ability to
switch adaptively to different computational strategies depending on the situation. This
ability has long been studied in various fields, and many hypothetical benefits seem to be
linked to it [56]. However, deep neural networks are often built without this ability to
optimally manage their computational resources. This limitation of current models is all the
more worrying as more and more recent work seems to show a linear relationship between
the computational capacity used and model performance during the evaluation phase [39].
To solve this problem, this thesis proposes different approaches and studies their impact
on models. First, we study a deep reinforcement learning agent that is able to allocate
more computation to more difficult situations. Our approach allows the agent to adapt its
computational resources according to the demands of the situation in which it finds itself,
which in addition to improving computation time, enhances transfer between related tasks
and generalization capacity. The central idea common to all our approaches is based on
cost-of-effort theories from the cognitive control literature, which stipulate that by making
the use of cognitive resources costly for the agent, and allowing it to allocate them when
making decisions, it will itself learn to deploy its computational capacity optimally.
We then study variations of the method on a reference deep learning task, to analyze
precisely how the model behaves and what the benefits of adopting such an approach are. We
also create our own task "Stroop MNIST" inspired by the Stroop test [74] used in psychology
to validate certain hypotheses about the behavior of neural networks employing our method.
We end by highlighting the strong links between dual learning and knowledge distillation
methods.
Finally, we approach the problem with energy-based models, by learning an energy land-
scape during training, the model can then during inference employ a computational capacity
7dependent on the difficulty of the example it is dealing with rather than a simple fixed for-
ward propagation having systematically the same computational cost. Despite unsuccessful
experimental results, we analyze the promise of such an approach and speculate on potential
improvements.
With our contributions, we hope to pave the way for algorithms that make better use of
their computational resources, and thus become more efficient in terms of cost and perfor-
mance, as well as providing a more intimate understanding of the links that exist between
certain machine learning methods and dual process theory.
