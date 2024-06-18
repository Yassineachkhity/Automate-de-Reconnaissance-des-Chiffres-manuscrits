from collections import defaultdict, deque
from itertools import chain, combinations
import graphviz
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import tensorflow as tf
from Alphabet import *
from Transition import *
from Etat import *

class Automate:
    def __init__(self, alphabet, etats, etats_initiaux, etats_finaux):
        self.set_alphabet(alphabet)
        self.set_etats(etats)
        self.set_etats_initiaux(etats_initiaux)
        self.set_etats_finaux(etats_finaux)
        self.set_transitions([])
        for etat in etats_initiaux:
            etat.set_initial(True)
        for etat in etats_finaux:
            etat.set_final(True)

    def ajouter_transition(self, transition):
        self.get_transitions().append(transition)
        transition.get_etat_source().ajouter_transition(transition)

    def supprimer_transition(self, transition):
        self.get_transitions().remove(transition)
        transition.get_etat_source().supprimer_transition(transition)

    def ajouter_etat(self, etat):
        self.get_etats().append(etat)

    def supprimer_etat(self, etat):
        self.get_etats().remove(etat)
        for transition in list(self.get_transitions()):
            if transition.get_etat_source() == etat or transition.get_etat_destination() == etat:
                self.supprimer_transition(transition)

    def ajouter_etat_initial(self, etat):
        etat.set_initial(True)
        self.get_etats_initiaux().append(etat)

    def supprimer_etat_initial(self, etat):
        etat.set_initial(False)
        self.get_etats_initiaux().remove(etat)

    def ajouter_etat_final(self, etat):
        etat.set_final(True)
        self.get_etats_finaux().append(etat)

    def supprimer_etat_final(self, etat):
        etat.set_final(False)
        self.get_etats_finaux().remove(etat)
    
    def lire_automate(self, entree):
        alphabet, etats, etats_initiaux, etats_finaux, transitions = entree
        self.set_alphabet(Alphabet(alphabet))
        self.set_etats([Etat(str(etat)) for etat in etats])
        self.set_etats_initiaux([etat for etat in self.get_etats() if etat.get_nom() in map(str, etats_initiaux)])
        self.set_etats_finaux([etat for etat in self.get_etats() if etat.get_nom() in map(str, etats_finaux)])
        for transition in transitions:
            etat_source, symbole, etat_destination = transition
            etat_source_obj = next(etat for etat in self.get_etats() if etat.get_nom() == str(etat_source))
            etat_destination_obj = next(etat for etat in self.get_etats() if etat.get_nom() == str(etat_destination))
            self.ajouter_transition(Transition(etat_source_obj, symbole, etat_destination_obj))
    
    def get_alphabet(self):
        return self._alphabet

    def set_alphabet(self, alphabet):
        self._alphabet = alphabet

    def get_etats(self):
        return self._etats

    def set_etats(self, etats):
        self._etats = etats

    def get_etats_initiaux(self):
        return self._etats_initiaux

    def set_etats_initiaux(self, etats_initiaux):
        self._etats_initiaux = etats_initiaux

    def get_etats_finaux(self):
        return self._etats_finaux

    def set_etats_finaux(self, etats_finaux):
        self._etats_finaux = etats_finaux

    def get_transitions(self):
        return self._transitions

    def set_transitions(self, transitions):
        self._transitions = transitions

    def est_complet(self):
        """
        Vérifie si l'automate est complet.
        """
        for etat in self.get_etats():
            transitions = set(transition.get_symbole() for transition in etat.get_transitions())
            if set(self.get_alphabet().get_symboles()) != transitions:
                return False
        return True
    
    def est_deterministe(self):
        """
        Vérifie si l'automate est déterministe.
        """
        for etat in self.get_etats():
            transitions = defaultdict(list)
            for transition in etat.get_transitions():
                transitions[transition.get_symbole()].append(transition.get_etat_destination())
            for destinations in transitions.values():
                if len(destinations) > 1:
                    return False
        return True

    def rendre_deterministe(self):
        """
        Convertit l'automate en un automate déterministe (AFN -> AFD).
        """
        if self.est_deterministe():
            return

        # Initialisation de la nouvelle structure d'automate déterministe
        nouvel_automate = Automate(self.get_alphabet(), [], [], [])
        nouvelle_etats_map = {}
        nouvelle_transitions = []

        # Création de l'état initial déterministe à partir de l'ensemble des états initiaux de l'AFN
        etat_initial_deterministe = frozenset(self.get_etats_initiaux())
        nouvelle_etats_map[etat_initial_deterministe] = Etat(str(etat_initial_deterministe))
        nouvel_automate.ajouter_etat(nouvelle_etats_map[etat_initial_deterministe])
        nouvel_automate.ajouter_etat_initial(nouvelle_etats_map[etat_initial_deterministe])

        # Utilisation d'une queue pour gérer les états à explorer
        queue = deque([etat_initial_deterministe])
        etats_visites = set()

        while queue:
            etat_courant = queue.popleft()
            if etat_courant in etats_visites:
                continue
            etats_visites.add(etat_courant)

            for symbole in self.get_alphabet().get_symboles():
                nouvel_etat = frozenset(
                    destination
                    for etat in etat_courant
                    for transition in etat.get_transitions()
                    if transition.get_symbole() == symbole
                    for destination in [transition.get_etat_destination()]
                )
                if not nouvel_etat:
                    continue

                if nouvel_etat not in nouvelle_etats_map:
                    nouvelle_etats_map[nouvel_etat] = Etat(str(nouvel_etat))
                    nouvel_automate.ajouter_etat(nouvelle_etats_map[nouvel_etat])
                    queue.append(nouvel_etat)

                nouvelle_transition = Transition(nouvelle_etats_map[etat_courant], symbole, nouvelle_etats_map[nouvel_etat])
                nouvelle_transitions.append(nouvelle_transition)
                nouvel_automate.ajouter_transition(nouvelle_transition)

                if any(etat in self.get_etats_finaux() for etat in nouvel_etat):
                    nouvel_automate.ajouter_etat_final(nouvelle_etats_map[nouvel_etat])

        self.set_alphabet(nouvel_automate.get_alphabet())
        self.set_etats(nouvel_automate.get_etats())
        self.set_etats_initiaux(nouvel_automate.get_etats_initiaux())
        self.set_etats_finaux(nouvel_automate.get_etats_finaux())
        self.set_transitions(nouvel_automate.get_transitions())

    def rendre_complet(self):
        """
        Rend l'automate complet s'il ne l'est pas.
        """
        if self.est_complet():
            return

        # Créer un nouvel état puits
        etat_puits = Etat("puits")
        self.ajouter_etat(etat_puits)

        # Ajouter les transitions manquantes vers l'état puits
        for etat in self.get_etats():
            for symbole in self.get_alphabet().get_symboles():
                transitions_existantes = [transition for transition in etat.get_transitions() if transition.get_symbole() == symbole]
                if not transitions_existantes:
                    self.ajouter_transition(Transition(etat, symbole, etat_puits))
                     
    def afficher_graphe(self):
        """
        Affiche le graphe de transition de l'automate.
        """
        dot = graphviz.Digraph()

        for etat in self.get_etats():
            dot.node(etat.get_nom(), shape='circle')

        for etat in self.get_etats_initiaux():
            dot.node(etat.get_nom(), shape='circle', style='filled', color='yellow')

        for etat in self.get_etats_finaux():
            dot.node(etat.get_nom(), shape='doublecircle')

        for transition in self.get_transitions():
            dot.edge(str(transition.get_etat_source()), str(transition.get_etat_destination()), label=transition.get_symbole())

        dot.render('minimiser', view=True)
        
        
            # Fonctions pour le modèle de reconnaissance des chiffres : Partie application

    
    # Fonction de la partie application
    
    def charger_image(self, filepath):
        img = cv.imread(filepath)
        return img
    
    def conversion_grayscale(self, img):
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    def inversion_couleurs(self, img):
        return cv.bitwise_not(img)
    
    def redimensionnement(self, img, size=(28, 28)):
        return cv.resize(img, size, interpolation=cv.INTER_AREA)
    
    def normalisation(self, img):
        return img / 255.0
    
    def prediction(self, img, model):
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)
        prediction = model.predict(img)
        return np.argmax(prediction)


def automate_to_list(automate):
    alphabet = automate.get_alphabet().get_symboles()
    etats = automate.get_etats()
    etats_initiaux = automate.get_etats_initiaux()
    etats_finaux = automate.get_etats_finaux()
    transitions = automate.get_transitions()

    alphabet_list = [symbole for symbole in alphabet]
    etats_list = [etat.get_nom() for etat in etats]
    etats_initiaux_list = [etat.get_nom() for etat in etats_initiaux]
    etats_finaux_list = [etat.get_nom() for etat in etats_finaux]
    transitions_list = [(transition.get_etat_source().get_nom(), transition.get_symbole(), transition.get_etat_destination().get_nom()) for transition in transitions]

    return alphabet_list, etats_list, etats_initiaux_list, etats_finaux_list, transitions_list

def mot(etat, alphabet, transitions):
    for transition in transitions:
        if transition[0] == etat and transition[1] == alphabet:
            return transition[2]
    return None

def chercher_cle(dictionnaire, element_recherche):
    for cle, liste_valeurs in dictionnaire.items():
        if element_recherche in liste_valeurs:
            return cle
    return None

def equivalent(etat1, etat2, classes, alphabets, transitions):
    if any(etat1 in classe and etat2 in classe for classe in classes.values()):
        for alphabet in alphabets:
            classe_transition1 = chercher_cle(classes, mot(etat1, alphabet, transitions))
            classe_transition2 = chercher_cle(classes, mot(etat2, alphabet, transitions))
            if classe_transition1 != classe_transition2:
                return False
        return True
    return False

def minimiser(automate):
    if not automate.est_deterministe():
        automate.rendre_deterministe()
    if not automate.est_complet():
        automate.rendre_complet()

    donnees = automate_to_list(automate)
    alphabet, etats, etats_initiaux, etats_finaux, transitions_list = donnees
    
    classesEquivalence = {
        1: etats_finaux,
        2: [etat for etat in etats if etat not in etats_finaux]
    }
    nouvClassesEquivalence = {}
    continuer = True

    while continuer:
        nouvClassesEquivalence = {}
        for etat in etats:
            classe_attribuee = False
            for classe_id, classe in nouvClassesEquivalence.items():
                if equivalent(classe[0], etat, classesEquivalence, alphabet, transitions_list):
                    classe.append(etat)
                    classe_attribuee = True
                    break
            if not classe_attribuee:
                nouvClassesEquivalence[len(nouvClassesEquivalence) + 1] = [etat]

        if classesEquivalence == nouvClassesEquivalence:
            continuer = False
        else:
            classesEquivalence = nouvClassesEquivalence

    nouvEtats = [tuple(classe) for classe in classesEquivalence.values()]
    nouvEtatsinitiale = [tuple(classe) for classe in classesEquivalence.values() if any(etat in classe for etat in etats_initiaux)]
    nouvEtatsFinale = [tuple(classe) for classe in classesEquivalence.values() if any(etat in classe for etat in etats_finaux)]

    nouvTransition = []
    for classe in classesEquivalence.values():
        for alpha in alphabet:
            etatCible = mot(classe[0], alpha, transitions_list)
            if etatCible is not None:
                classeSource = next(tuple(cl) for cl in nouvEtats if classe[0] in cl)
                classeCible = next(tuple(cl) for cl in nouvEtats if etatCible in cl)
                nouvTransition.append((classeSource, alpha, classeCible))

    automate_minimise = Automate(Alphabet(alphabet), [], [], [])
    automate_minimise.lire_automate([alphabet, nouvEtats, nouvEtatsinitiale, nouvEtatsFinale, nouvTransition])
    
    return automate_minimise


 
        


# exmple d'utilisation partie initiale
if __name__ == "__main__":
    
    
    entree = [['a', 'b', 'c', 'd', 'e'],
              [1, 2, 3, 4, 5, 6],
              [1, 3, 6],
              [6],
              [(1, 'a', 2),
               (1, 'a', 4),
               (2, 'a', 2),
               (2, 'c', 5),
               (2, 'd', 5),
               (3, 'b', 2),
               (3, 'b', 4),
               (4, 'b', 4),
               (4, 'c', 5),
               (4, 'd', 5),
               (5, 'e', 6)]]
    
    # entree = [['a', 'b'],
    #       [0, 1, 2, 3],
    #       [0],
    #       [2, 3],
    #       [(0, 'a', 1),
    #        (0, 'b', 1),
    #        (1, 'a', 2),
    #        (1, 'b', 2),
    #        (2, 'a', 2),
    #        (2, 'b', 1),
    #        (3, 'b', 0),
    #        (3, 'a', 2)]]

    automate = Automate(None, [], [], [])
    automate.lire_automate(entree)

# # Ajout de transition
#     etat_source = automate.get_etats()[0]
#     symbole = 'f'
#     etat_destination = automate.get_etats()[5]
#     nouvelle_transition = Transition(etat_source, symbole, etat_destination)
#     automate.ajouter_transition(nouvelle_transition)

# # Suppression d'une transition
#     transition_a_supprimer = automate.get_transitions()[0]
#     automate.supprimer_transition(transition_a_supprimer)

# # Ajout d'un nouvel état
#     nouvel_etat = Etat("7")
#     automate.ajouter_etat(nouvel_etat)

# # Suppression d'un état
#     etat_a_supprimer = automate.get_etats()[0]
#     automate.supprimer_etat(etat_a_supprimer)

# # Ajout d'un état initial
#     automate.ajouter_etat_initial(nouvel_etat)

# # Suppression d'un état initial
#     etat_initial_a_supprimer = automate.get_etats_initiaux()[0]
#     automate.supprimer_etat_initial(etat_initial_a_supprimer)

# # Ajout d'un état final
#     automate.ajouter_etat_final(nouvel_etat)

# # Suppression d'un état final
#     etat_final_a_supprimer = automate.get_etats_finaux()[0]
#     automate.supprimer_etat_final(etat_final_a_supprimer)
    
    # automate.afficher_graphe()
    # automate.rendre_complet()
    # automate.afficher_graphe()
    # automate.rendre_deterministe()
    # automate.afficher_graphe()
    minimiser(automate)
    automate.afficher_graphe()


    
# # Exemple d'utilisation pour la reconnaisssance de chiffre manuscrit (enlever les commentaires pour tester):
# #Charger le modèle de reconnaissance des chiffres manuscrits
#     model = tf.keras.models.load_model('handwritten_digits.model')

# #Créer l'automate avec les états et transitions appropriés
#     alphabet = Alphabet(['load', 'grayscale', 'invert', 'resize', 'normalize', 'predict'])
    
#     etats = [Etat('initial'), Etat('loaded'), Etat('grayscaled'), Etat('inverted'), Etat('resized'), Etat('normalized'), Etat('predicted')]
    
#     automate = Automate(alphabet, etats, [etats[0]], [etats[-1]])

# # Définir les transitions pour le processus de reconnaissance de chiffre
#     transitions = [
#         Transition(etats[0], 'load', etats[1]),
#         Transition(etats[1], 'grayscale', etats[2]),
#         Transition(etats[2], 'invert', etats[3]),
#         Transition(etats[3], 'resize', etats[4]),
#         Transition(etats[4], 'normalize', etats[5]),
#         Transition(etats[5], 'predict', etats[6]),
#     ]

#     for transition in transitions:
#         automate.ajouter_transition(transition)

# # Simuler le processus de reconnaissance pour une image donnée
#     for i in range(1, 12):
#         image_path = f'C:/Users/yassine/OneDrive/Bureau/Automate Projet/phase application/digits/digit{i}.png'

# # Suivre les transitions de l'automate
#         img = automate.charger_image(image_path)
#         img = automate.conversion_grayscale(img)
#         img = automate.inversion_couleurs(img)
#         img = automate.redimensionnement(img)
#         img = automate.normalisation(img)
#         pred = automate.prediction(img, model)
#         automate.afficher_graphe()


# # Affichage de l'image et de la prédiction
#         plt.imshow(img, cmap=plt.cm.binary)
#         plt.title(f"The number is probably a {pred}")
#         plt.show()
    