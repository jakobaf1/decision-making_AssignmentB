# Define dummy policy function
class DummyPolicy:
    def select_action(self, state):
        return {"HeatPowerRoom1" : 0, "HeatPowerRoom2" : 0, "VentilationON" : 0}