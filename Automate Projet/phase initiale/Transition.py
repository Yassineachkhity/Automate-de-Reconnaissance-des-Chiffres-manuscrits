class Transition:
    def __init__(self, etat_source, symbole, etat_destination):
        self._etat_source = etat_source
        self._symbole = symbole
        self._etat_destination = etat_destination

    def __repr__(self):
        return f"({self._etat_source}, {self._symbole}, {self._etat_destination})"

    def get_etat_source(self):
        return self._etat_source

    def set_etat_source(self, etat_source):
        self._etat_source = etat_source

    def get_symbole(self):
        return self._symbole

    def set_symbole(self, symbole):
        self._symbole = symbole

    def get_etat_destination(self):
        return self._etat_destination

    def set_etat_destination(self, etat_destination):
        self._etat_destination = etat_destination
