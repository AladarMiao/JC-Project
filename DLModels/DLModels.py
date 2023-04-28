import datetime as dt

class DLModel:
    def __init__(self):
        self.model = None
        self.history = None

    def train(self):
        pass

    def define_model(self):
        pass

    def save_model(self):

        # Get the current time as a string
        timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')

        # Construct the filename with the timestamp
        filename = f'my_model_{timestamp}.h5'

        # Save the model to the file with the timestamped name
        self.model.save(filename)

        # Print message indicating the model has been saved
        print(f'Model saved to {filename}')
