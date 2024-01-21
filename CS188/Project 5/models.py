import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        # Compute the dot product
        result = nn.as_scalar(self.run(x))
        return 1.0 if result >= 0 else -1.0

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        # Initialize the accuracy
        accuracy = 0.0

        # Iterate until accuracy is 100%
        while accuracy != 1.0:
            # Reset accuracy
            accuracy = 0.0
            counter = 0
            # Loop through data with a batch size of 1
            for x, y, in dataset.iterate_once(1):
                # Make predictions
                y_hat = self.get_prediction(x)

                # If the prediction is equal to the label, add +1 to the accuracy
                if y_hat == nn.as_scalar(y):
                    accuracy += 1
                else: # update weight
                    direction = x
                    multiplier = nn.as_scalar(y)
                    self.w.update(direction, multiplier)

                counter += 1


            # Divide the accuracy by the number of samples
            accuracy = accuracy / counter


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        # Here we implement two layers
        self.w1 = nn.Parameter(784, 512)
        self.b1 = nn.Parameter(1, 512)

        self.w2 = nn.Parameter(512, 1)
        self.b2 = nn.Parameter(1, 1)

        self.lr = 0.05


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"

        # Apply the first layer
        z1 = nn.Linear(x, self.w1) #sizes (1,1) * (1, 512) = (1, 512)
        z1 = nn.AddBias(z1, self.b1) # sizes (1,512) + (1,512) = (1, 512)
        z1 = nn.ReLU(z1) # size: (1, 512)

        # Second (output) layer
        z2 = nn.Linear(z1, self.w2) # sizes: (1, 512) * (512, 1) = (1,1)
        z2 = nn.AddBias(z2, self.b2) # sizes: (1,1) + (1,1) = (1,1)

        return z2

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        # Make predictions
        y_hat = self.run(x)

        # Compute loss
        loss = nn.SquareLoss(y_hat, y)

        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        # We will train until loss is < 0.02
        loss = float('inf')
        eps = 0.01
        batch_size = 200
        epoch = 1

        print('Training started!')
        while loss >= eps:
            epoch_loss = 0 # loss for current iteration
            for x, y in dataset.iterate_once(batch_size):

                # Compute loss
                loss = self.get_loss(x, y)

                # Compute gradient
                dw1, dw2, db1, db2 = nn.gradients(loss, [self.w1, self.w2, self.b1, self.b2])

                # Convert loss to scalar
                loss = nn.as_scalar(loss)

                # Update parameters
                self.w1.update(dw1, -self.lr) # w -= learning_rate * dw
                self.b1.update(db1, -self.lr) # b -= learning_rate * db
                self.w2.update(dw2, -self.lr)
                self.b2.update(db2, -self.lr)

                epoch_loss += loss

            # Print loss
            loss = epoch_loss
            print("Epoch {0}, loss: {1:.4f}".format(epoch, loss))
            epoch += 1

        print("Training ended! Final loss: {0:.4f}".format(loss))




class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        # Batch size and learning rate
        self.batch_size = 200
        self.lr = 0.05
        self.n_features = 784

        # Input layer weights
        self.w1 = nn.Parameter(self.n_features, 512)
        self.b1 = nn.Parameter(1, 512)

        # Hidden layer
        self.w2 = nn.Parameter(512, 256)
        self.b2 = nn.Parameter(1, 256)

        # Output layer
        self.w3 = nn.Parameter(256, 10)
        self.b3 = nn.Parameter(1, 10)

        

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

        # Apply input layer
        y = nn.Linear(x, self.w1) # sizes: (batch_size, 784) * (784, 512) = (batch_size, 512)
        y = nn.AddBias(y, self.b1) # sizes: (batch_size, 512) + (1, 512) = (batch_size, 512)
        y = nn.ReLU(y) # size: (batch_size, 512)

        # Hidden layer
        y = nn.Linear(y, self.w2) # sizes: (batch_size, 512) * (512, 256) = (batch_size, 256)
        y = nn.AddBias(y, self.b2) # sizes: (batch_size, 256) + (1, 256) = (batch_size, 256)
        y = nn.ReLU(y) # size: (batch_size, 256)

        # Output layer
        y = nn.Linear(y, self.w3) # sizes: (batch_size, 256) * (256, 10) = (batch_size, 10)
        y = nn.AddBias(y, self.b3) # sizes: (batch_size, 10) + (1, 10) = (batch_size, 10)

        return y


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        # Get predictions
        y_hat = self.run(x)

        # Calculate loss
        loss = nn.SoftmaxLoss(y_hat, y)

        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        # We train until accuracy > 98%
        accuracy = 0.0
        eps = 0.98 # minimum accuracy required for training to end

        epoch = 1
        while accuracy <= eps:
            for x, y in dataset.iterate_once(self.batch_size):
                # Get loss
                loss = self.get_loss(x, y)

                # Compute gradients
                dw1, dw2, dw3, db1, db2, db3 = nn.gradients(loss, [self.w1, self.w2, self.w3, self.b1, self.b2, self.b3])

                # Update parameters
                self.w1.update(dw1, -self.lr)
                self.w2.update(dw2, -self.lr)
                self.w3.update(dw3, -self.lr)
                self.b1.update(db1, -self.lr)
                self.b2.update(db2, -self.lr)
                self.b3.update(db3, -self.lr)

            
            accuracy = dataset.get_validation_accuracy()
            print("Epoch {0}, acc: {1:.4f}".format(epoch, accuracy))
            epoch += 1


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.output_size = len(self.languages)
        self.hidden_size = 1024

        self.lr = 0.05
        self.batch_size = 64

        # Input layer
        self.w1 = nn.Parameter(self.num_chars, self.hidden_size)
        self.b1 = nn.Parameter(1, self.hidden_size)

        # Output layer
        self.w2 = nn.Parameter(self.hidden_size, self.output_size)
        self.b2 = nn.Parameter(1, self.output_size)

        # Hidden layer
        self.w3 = nn.Parameter(self.output_size, self.hidden_size)
        self.b3 = nn.Parameter(1, self.hidden_size)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        
        h = None
        for i, x in enumerate(xs):
            # If we are at the first sample, we don't have output from previous sample
            if i == 0:
                # Pass sampe through first layer
                y = nn.Linear(x, self.w1) # sizes: (batch_size, n_features) * (n_features, hidden_size) = (batch_size, hidden_size)
                y = nn.AddBias(y, self.b1)
                y = nn.ReLU(y)

                # Now pass through output layer
                y = nn.Linear(y, self.w2) # sizes: (batch_size, hidden_size) * (hidden_size, output_size) = (batch_size, output_size)
                y = nn.AddBias(y, self.b2)
                # This output y is the 'h' vector for next sample
                h = y # (batch_size, output_size)
            else:
                # Combine the current sample and output from previous sample
                y = nn.Add(nn.Linear(x, self.w1), nn.Linear(h, self.w3)) # [(batch_size, n_features) * (n_features, hidden_size)] + [(batch_size, output_size) * (output_size, hidden_size)] = (batch_size, hidden_size)
                y = nn.AddBias(y, self.b1)
                y = nn.ReLU(y)

                y = nn.Linear(y, self.w2)
                y = nn.AddBias(y, self.b2)

                h = y
        return h


    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

        # Make predictions
        y_hat = self.run(xs)

        return nn.SoftmaxLoss(y_hat, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        # train until the accuracy is higher than 98%
        eps = 0.81
        accuracy = 0.0
        epoch = 1

        while accuracy <= eps:

            for x, y in dataset.iterate_once(self.batch_size):
                # Compute loss
                loss = self.get_loss(x, y)

                # Compute gradients
                dw1, dw2, dw3, db1, db2 = nn.gradients(loss, [self.w1, self.w2, self.w3, self.b1, self.b2])

                # Update weights
                self.w1.update(dw1, -self.lr)
                self.w2.update(dw2, -self.lr)
                self.w3.update(dw3, -self.lr)
                self.b1.update(db1, -self.lr)
                self.b2.update(db2, -self.lr)

            # Compute accuracy
            accuracy = dataset.get_validation_accuracy()
                
            # Print accuracy and loss
            print("Epoch {0}, accuracy: {1:.4f}, loss: {2:.4f}".format(epoch, accuracy, nn.as_scalar(loss)))
            epoch += 1


