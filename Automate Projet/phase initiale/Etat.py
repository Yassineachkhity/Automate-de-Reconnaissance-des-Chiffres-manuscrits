class Etat:
    def __init__(self, nom):
        self._nom = nom
        self._transitions = []
        self._initial = False
        self._final = False

    def __repr__(self):
        return self._nom

    def ajouter_transition(self, transition):
        self._transitions.append(transition)

    def supprimer_transition(self, transition):
        self._transitions.remove(transition)

    def get_nom(self):
        return self._nom

    def set_nom(self, nom):
        self._nom = nom

    def get_transitions(self):
        return self._transitions

    def set_transitions(self, transitions):
        self._transitions = transitions

    def is_initial(self):
        return self._initial

    def set_initial(self, initial):
        self._initial = initial

    def is_final(self):
        return self._final

    def set_final(self, final):
        self._final = final
