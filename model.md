### Models

Here, I go into detail about each model. This is mainly for me to understand the underlying processes of how these neural networks work.

**Recurrent Neural Networks (RNN)**

A recurrent neural network (RNN) is any network that contains a cycle within its network connections. That is, any network where the value of a unit is directly , or indirectly, dependent on earlier outputs as an input.

As with ordinary feedforward netowrks, an input vector representing the current input element <img src="https://render.githubusercontent.com/render/math?math=x_t">, is multiplied by a weight matrix and then passed through an activation function to compute an activation value for a layer of hidden units. This hidden layer is used to calculate a corresponding output, <img src="https://render.githubusercontent.com/render/math?math=y_t">. The recurrent link in this feedforward network augments the input to the computation at the hidden layer with the activation value of the hidden layer *from the preceding point in time*.

The hidden layer from the previous time step provides a form of memory, or context, that encodes earlier processing and informs the decisions to be made at later points in time. This type of architecture does not impose a fixed-length limit on this prior context; the context embodied in the previous hidden layers includes information extending back to the beginning of the sequence.

This addition of temporal dimension is not much different than non-recurrent architectures. Given an input vector and the values for the hidden layer from the previous time step, we're still performing standard feedforward calculations. 

The most significant change lies in the new set of weights that connect the hidden layer from the previous time step to the current hidden layer. Theses weights determine how the network should make use of past context in calculating the output of the current input. These connections are trained via backpropagation.

**Bidirectional LSTM**

In a simple recurrent network, the hidden state at a given time *t* represents everything the network knows about the sequence up to that point in the sequence. That is, the hiddent state at time *t* is the result of a function of the inputs from the start up through time *t*. This can be thought as the context of the network to the left of the current time: <img src="https://render.githubusercontent.com/render/math?math=h_{t}^{f} = RNN_{forward}(x_{1}^{t})">, where <img src="https://render.githubusercontent.com/render/math?math=h_{t}^{f}"> corresponds to the normal hidden state at time *t*, and represents everything the network has gleaned from the sequence to that point.

When we have access to the entire input sequence all at once, it can be helpful to take advantage of the context to the right of the current input as well. To recover such information, we can train an RNN on an input sequence in reverse. With this approach, the hidden state at time *t* represents information about the sequence to the right of the current input: <img src="https://render.githubusercontent.com/render/math?math=h_{t}^{b} = RNN_{backward}(x_{t}^{n})">, where the hidden state <img src="https://render.githubusercontent.com/render/math?math=h_{t}^{b}"> represents all the information we have discerned about the sequence from *t* to the end of the sequence.

Combining these forward and backward networks results in a bidirectional RNN. A BiRNN consists of two independent RNNs, on where the input is processed from the start to the end, and the other from the end to the start. We then combine the outputs of the two networks into a single representation that captures both the left and right contexts of an input at each point in time.

The outputs of the forward and backward pass can be concatentated, element-wise summed, or multiplied. The output at each step in time thus captures information to the left and to the right of the current input.

It is quite difficult to train RNNs for tasks that require a network to make use of information distant from the current point of processing. Despite having access to the entire preceding sequence, the information encoded in hidden states tends to be fairly local, more relevant to the most recent parts of the input sequence and recent decisions. However, it is often the case that distant information is critical to many language applications. 

One reason for the inability of RNNs to carry forward critical information is that the hidden layers, and, by extension, the weights that determine the values in the hidden layer, are being asked to perform two tasks simultaneously: provide information useful for the current decision, and updating and carrying forward information require for future decisions.

A second difficulty with training simple recurrent networks arises from the need to backpropagate the error signal back through time (the hidden layer at time *t* contributes to the loss at the next time step since it takes part in the calculation). As a result, during the backward pass of training, the hidden layers are subject to repeated multiplications, as determined by the length of the sequence. A frequent result of this process is that the gradients are eventually driven to zero - the **vanishing gradient** problem.

To address these issues, the network needs to learn to forget information that is no longer needed and to remember information required for decisions still to come.

Long short-term memory (LSTM) networks divide the context management problem into two sub-problems: removing information no longer needed from the context, and adding information likely to be needed for later decision making. The key to solving both problems is to learn how to manage this context rather than hard-coding a strategy into the architecture. LSTMs accomplish this by first adding an explicit context layer to the architecture (in addition to the usual recurrent hidden layer), and through the use of specialized neural units that make use of *gates* to control the flow of information into and out of the units that comprise the network layers. These gates are implemented through the use of additional weights that operate sequentially on the input, and previous hidden layer, and previous context layers.

The gates in an LSTM share a common design pattern: each consists of a feedforward layer, followed by a sigmoid activation function, followed by a pointwise multiplication with the layer being gated. The choice of the sigmoid as the activation function arises from its tendency to push its outputs to either 0 or 1. Combining this with a pointwise multiplication has an effect similar to that of a binary mask. Values in the layer being gated that align with values near 1 in the mask are passed through nearly unchanged while values corresponding to lower values are essentially erased.

The **forget gate** deletes information from the context that is no longer needed. The forget gate computes a weighted sum of the previous state's hidden layer and the current input and passes that through a sigmoid. This mask is then multiplied by the context vector to remove the information from context that is no longer required. The next task is to compute the actual information we need to extract from the previous hidden state and current inputs.

Next, we generate the mask for the **add gate** to select the information to add to the current context. Then we add this to the modified context vector to get a new context vector.

The final gate, **output gate**, is used to decide what information is required for the current hidden state (as opposed to what information needs to be preserved for future decisions).

**FastText**

[FastText](https://arxiv.org/abs/1607.01759) uses a hierarchical classifier instead of a flat structure, in which the different categories are organized into a tree. This reduces the time complexities of training and testing text classifiers from linear to logarithmic with respect to the number of classes. The depth in the tree of very frequent categories is therefore smaller than for infrequent categories, leading to further computational efficiency.

FastText represents a text by a low dimensional vector, which is obtained by summing vectors corresponding to the words appearing in the text. In fastText, a low dimensional vector is associated to each word of the vocabulary. This hidden representation is shared across all classifiers for different categories, allowing information about words learned for one category to be used by other categories. These kind of representations, bag of words, ignore word order. In fastText, vectors are used to represent ngrams to take into account local word order, important for many text classification problems.

**Convolutional Neural Network**

Refer to this [paper](https://arxiv.org/abs/1408.5882)

**DistilBERT**

[DistilBERT](https://arxiv.org/abs/1910.01108) is a small, fast, cheap, and light Transformer model trained by distilling BERT base. It has 40% less parameters, runs 60% faster while preserving over 95% of BERT's performances.

A Gated Recurrent Unit (GRU) was used in my DistilBERT model. GRU has a reset gate,*r*, and an update gate, *z*. The purpose of the reset gate is to decide which aspects of the previous hiddent state are relevant to the current context and what can be ignored. This is accomplished by performing an element-wise multiplication or *r* with the value of the previous hidden state. We then use this masked value in computing an intermediate representation for the new hidden state at time *t*.

The update gate *z* is to determine which aspects of this new state will be used directly in the new hidden state and which aspects of the previous state need to be preserved for future use. This is accomplished by using the values in *z* to interpolate between the old hidden state and the new one.