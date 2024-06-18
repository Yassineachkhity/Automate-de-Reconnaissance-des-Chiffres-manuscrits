class Alphabet:
    def __init__(self, symboles):
        self._symboles = set(symboles)

    def __repr__(self):
        return str(self._symboles)

    def ajouter_symbole(self, symbole):
        self._symboles.add(symbole)

    def supprimer_symbole(self, symbole):
        self._symboles.remove(symbole)

    def get_symboles(self):
        return self._symboles

    def set_symboles(self, symboles):
        self._symboles = set(symboles)
