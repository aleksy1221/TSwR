import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManiuplatorModel(Tp)
        self.Kp = np.array([[10, 0], [0, 10]])
        self.Kd = np.array([[5, 0], [0, 5]])

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """
        q = x[:2]
        q_dot = x[2:]

        e = q_r - q
        e_dot = q_r_dot - q_dot

        v = q_r_ddot + np.dot(self.Kd, e_dot) + np.dot(self.Kp, e)

        # Compute the control input
        M = self.model.M(x)
        C = self.model.C(x)
        tau = np.dot(M, v) + np.dot(C, q_dot)

        return tau

