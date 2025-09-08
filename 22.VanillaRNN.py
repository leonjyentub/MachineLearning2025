import numpy as np
import string

def string_to_one_hot(inputs: np.ndarray) -> np.ndarray:
    char_to_index = {char: i for i, char in enumerate(string.ascii_uppercase)}

    one_hot_inputs = []
    for row in inputs:
        one_hot_list = []
        for char in row:
            if char.upper() in char_to_index:
                one_hot_vector = np.zeros((len(string.ascii_uppercase), 1))
                one_hot_vector[char_to_index[char.upper()]] = 1
                one_hot_list.append(one_hot_vector)
        one_hot_inputs.append(one_hot_list)

    return np.array(one_hot_inputs)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    ##return np.exp(x-np.max(x)) / np.sum(np.exp(x), axis=0)
    return np.exp(x) / np.sum(np.exp(x), axis=0)

class InputLayer:
    inputs: np.ndarray
    U: np.ndarray = None
    delta_U: np.ndarray = None

    def __init__(self, inputs: np.ndarray, hidden_size: int):
        self.inputs = inputs
        self.U = np.random.uniform(low=0, high=1, size=(hidden_size, len(inputs[0])))
        self.delta_U = np.zeros_like(self.U)

    def get_input(self, time_step: int) -> np.ndarray:
        return self.inputs[time_step]

    def weighted_sum(self, time_step: int) -> np.ndarray:
        return self.U @ self.get_input(time_step)

    def calculate_deltas_per_step(self, time_step: int, delta_weighted_sum: np.ndarray):
        # (h_dimension, 1) @ (1, input_size) = (h_dimension, input_size)
        self.delta_U += delta_weighted_sum @ self.get_input(time_step).T

    def update_weights_and_bias(self, learning_rate: float) -> None:
        self.U -= learning_rate * self.delta_U

class HiddenLayer:
    states: np.ndarray = None
    W: np.ndarray = None
    delta_W: np.ndarray = None
    bias: np.ndarray = None
    delta_bias: np.ndarray = None
    next_delta_activation: np.ndarray = None

    def __init__(self, vocab_size: int, size: int):
        self.W = np.random.uniform(low=0, high=1, size=(size, size))
        self.bias = np.random.uniform(low=0, high=1, size=(size, 1))
        self.states = np.zeros(shape=(vocab_size, size, 1))
        self.next_delta_activation = np.zeros(shape=(size, 1))
        self.delta_bias = np.zeros_like(self.bias)
        self.delta_W = np.zeros_like(self.W)

    def get_hidden_state(self, time_step: int) -> np.ndarray:
        # If starting out at the beginning of the sequence, a[t-1] will return zeros
        if time_step < 0:
            return np.zeros_like(self.states[0])
        return self.states[time_step]

    def set_hidden_state(self, time_step: int, hidden_state: np.ndarray):
        self.states[time_step] = hidden_state

    def activate(self, weighted_input: np.ndarray, time_step: int) -> np.ndarray:
        previous_hidden_state = self.get_hidden_state(time_step - 1)
        # W @ h_prev => (h_dimension, h_dimension) @ (h_dimension, 1) = (h_dimension, 1)
        weighted_hidden_state = self.W @ previous_hidden_state
        # (h_dimension, 1) + (h_dimension, 1) + (h_dimension, 1) = (h_dimension, 1)
        weighted_sum = weighted_input + weighted_hidden_state + self.bias
        activation = np.tanh(weighted_sum)  # (h_dimension, 1)
        self.set_hidden_state(time_step, activation)
        return activation

    def calculate_deltas_per_step(self, time_step: int, delta_output: np.ndarray) -> np.ndarray:
        # (h_dimension, 1) + (h_dimension, 1) = (h_dimension, 1)
        delta_activation = delta_output + self.next_delta_activation
        # (h_dimension, 1) * scalar = (h_dimension, 1)
        delta_weighted_sum = delta_activation * (
            1 - self.get_hidden_state(time_step) ** 2
        )
        # (h_dimension, h_dimension) @ (h_dimension, 1) = (h_dimension, 1)
        self.next_delta_activation = self.W.T @ delta_weighted_sum

        # (h_dimension, 1) @ (1, h_dimension) = (h_dimension, h_dimension)
        self.delta_W += delta_weighted_sum @ self.get_hidden_state(time_step - 1).T

        # derivative of hidden bias is the same as dL_ds
        self.delta_bias += delta_weighted_sum
        return delta_weighted_sum

    def update_weights_and_bias(self, learning_rate: float) -> None:
        self.W -= learning_rate * self.delta_W
        self.bias -= learning_rate * self.delta_bias

class OutputLayer:
    states: np.ndarray = None
    V: np.ndarray = None
    bias: np.ndarray = None
    delta_bias: np.ndarray = None
    delta_V: np.ndarray = None

    def __init__(self, size: int, hidden_size: int):
        self.V = np.random.uniform(low=0, high=1, size=(size, hidden_size))
        self.bias = np.random.uniform(low=0, high=1, size=(size, 1))
        self.states = np.zeros(shape=(size, size, 1))
        self.delta_bias = np.zeros_like(self.bias)
        self.delta_V = np.zeros_like(self.V)

    def predict(self, hidden_state: np.ndarray, time_step: int) -> np.ndarray:
        # V @ h => (input_size, h_dimension) @ (h_dimension, 1) = (input_size, 1)
        # (input_size, 1) + (input_size, 1) = (input_size, 1)
        output = self.V @ hidden_state + self.bias
        prediction = softmax(output)
        self.set_state(time_step, prediction)
        return prediction

    def get_state(self, time_step: int) -> np.ndarray:
        return self.states[time_step]

    def set_state(self, time_step: int, prediction: np.ndarray):
        self.states[time_step] = prediction

    def calculate_deltas_per_step(
        self,
        expected: np.ndarray,
        hidden_state: np.ndarray,
        time_step: int,
    ) -> np.ndarray:
        # dL_do = dL_dyhat * dyhat_do = derivative of loss function * derivative of softmax
        # dL_do = step.y_hat - expected[step_number]
        delta_output = self.get_state(time_step) - expected  # (input_size, 1)

        # (input_size, 1) @ (1, hidden_size) = (input_size, hidden_size)
        self.delta_V += delta_output @ hidden_state.T

        # dL_dc += dL_do
        self.delta_bias += delta_output
        return self.V.T @ delta_output

    def update_weights_and_bias(self, learning_rate: float):
        self.V -= learning_rate * self.delta_V
        self.bias -= learning_rate * self.delta_bias

class VanillaRNN:
    hidden_layer: HiddenLayer
    output_layer: OutputLayer
    alpha: float  # learning rate
    input_layer: InputLayer = None

    def __init__(self, vocab_size: int, hidden_size: int, alpha: float) -> None:
        self.hidden_layer = HiddenLayer(vocab_size, hidden_size)
        self.output_layer = OutputLayer(vocab_size, hidden_size)
        self.hidden_size = hidden_size
        self.alpha = alpha

    def feed_forward(self, inputs: np.ndarray) -> OutputLayer:
        self.input_layer = InputLayer(inputs, self.hidden_size)
        for step in range(len(inputs)):
            weighted_input = self.input_layer.weighted_sum(step)
            activation = self.hidden_layer.activate(weighted_input, step)
            self.output_layer.predict(activation, step)
        return self.output_layer

    def backpropagation(self, expected: np.ndarray) -> None:
        for step_number in reversed(range(len(expected))):
            delta_output = self.output_layer.calculate_deltas_per_step(
                expected[step_number],
                self.hidden_layer.get_hidden_state(step_number),
                step_number,
            )
            delta_weighted_sum = self.hidden_layer.calculate_deltas_per_step(
                step_number, delta_output
            )
            self.input_layer.calculate_deltas_per_step(step_number, delta_weighted_sum)

        self.output_layer.update_weights_and_bias(self.alpha)
        self.hidden_layer.update_weights_and_bias(self.alpha)
        self.input_layer.update_weights_and_bias(self.alpha)

    def loss(self, y_hat, y) -> float:
        """
        Cross-entropy loss function - Calculating difference between 2 probability distributions.
        First, calculate cross-entropy loss for each time step with np.sum, which returns a numpy array
        Then, sum across individual losses of all time steps with sum() to get a scalar value.
        :param y_hat: predicted value
        :param y: expected value - true label
        :return: total loss
        """
        errors = -sum(y[i] * np.log(y_hat[i]) for i in range(len(y)))
        return sum(errors)

    def train(self, inputs: np.ndarray, expected: np.ndarray, epochs: int) -> None:
        for epoch in range(epochs):
            print(f"epoch={epoch}")
            for idx, input in enumerate(inputs):
                y_hats = self.feed_forward(input)
                self.backpropagation(expected[idx])
                print(f"Loss round: {self.loss([y for y in y_hats.states], expected[idx])}")

if __name__ == "__main__":
  inputs = np.array([
      ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"],
      ["Z","Y","X","W","V","U","T","S","R","Q","P","O","N","M","L","K","J","I","H","G","F","E","D","C","B","A"],
      ["B","D","F","H","J","L","N","P","R","T","V","X","Z","A","C","E","G","I","K","M","O","Q","S","U","W","Y"],
      ["M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","A","B","C","D","E","F","G","H","I","J","K","L"],
      ["H","G","F","E","D","C","B","A","L","K","J","I","P","O","N","M","U","T","S","R","Q","X","W","V","Z","Y"]
  ])

  expected = np.array([
      ["B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","A"],
      ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"],
      ["C","E","G","I","K","M","O","Q","S","U","W","Y","A","B","D","F","H","J","L","N","P","R","T","V","X","Z"], 
      ["N","O","P","Q","R","S","T","U","V","W","X","Y","Z","A","B","C","D","E","F","G","H","I","J","K","L","M"],
      ["I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","A","B","C","D","E","F","G","H"]
  ])
  
  one_hot_inputs = string_to_one_hot(inputs)
  one_hot_expected = string_to_one_hot(expected)

  # Forward pass through time, no gradient clipping yet so there will be gradient exploding problem
  # https://stackoverflow.com/a/33980220
  # https://stackoverflow.com/a/72494516
  rnn = VanillaRNN(vocab_size=len(string.ascii_uppercase), hidden_size=128, alpha=0.0001)
  rnn.train(one_hot_inputs, one_hot_expected, epochs=20)

  new_inputs = np.array([["B", "C", "D"]])
  for input in string_to_one_hot(new_inputs):
      predictions = rnn.feed_forward(input)
      output = np.argmax(predictions.states[-1])
      print(output) # index of the one-hot value of prediction
      print(string.ascii_uppercase[output]) # mapping one hot to character
