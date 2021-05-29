import parl

__all__ = ['ES']


class ES(parl.Algorithm):
    def __init__(self, model):
        """ES algorithm.
        
        Since parameters of the model is updated in the numpy level, `learn` function is not needed
        in this algorithm.
        Args:
            model(`parl.Model`): policy model of ES algorithm.
        """
        self.model = model

    def predict(self, obs):
        """Use the policy model to predict actions of observations.
        Args:
            obs(layers.data):  data layer of observations.
        Returns:
            tensor of predicted actions.
        """
        return self.model.predict(obs)