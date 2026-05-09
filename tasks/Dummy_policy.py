# Define dummy policy function
class DummyPolicy:
    def select_action(self, state):
        return {"p1" : 0, "p2" : 0, "v" : 0}