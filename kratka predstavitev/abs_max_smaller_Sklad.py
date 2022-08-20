class Sklad:
    def __init__(self):
        self.podatki= []

    def prazen(self):
        return len(self.podatki) == 0

    def vstavi(self, num):
        self.podatki.append(num)

    def odstrani(self):
        if self.prazen():
            raise Exception('Stack Underflow')
        return self.podatki.pop()

    def vrh(self):
        if self.prazen():
            return None
        return self.podatki[-1]

def naslednji_najmanjsi_el(tab):
    '''
        Funkcija s pomočjo sklada iz podanega seznama vrne seznam
        najbližjih in najmanjših elementov.
    '''
    if tab == [] or len(tab)==1:
        return False
    
    sklad = Sklad()
    rez = []
    # s for zanko pogledamo vse elemente v tabeli
    for i in range(len(tab)-1, -1, -1): # desna -> leva
        if sklad.prazen(): #če je sklad prazen, ni naslednjega min. 
            rez.append(0)
            sklad.vstavi(tab[i]) #v sklad vstavimo el iz tabele
        elif not sklad.prazen(): # v primeru, da sklad ni prazen
            while(not sklad.prazen() and tab[i] < sklad.vrh()): # preverjamo el. v skladu dokler ga ne izpraznemo in
                # dokler so vrhnji el iz sklada večji od obravnavanega el.
                #odstranjujemo vrhnje oz. večje elemente
                sklad.odstrani()
            if sklad.prazen(): # če nismo našli manjšega 
                rez.append(0)
            else:
                rez.append(sklad.vrh()) # našli smo manjši el.
            sklad.vstavi(tab[i])
        
    
    return rez



def maxEl(tabela):
    '''
    Funkcija vrne največji element med abolutnimi razlikami iz dveh tabel.
    '''
    desni = naslednji_najmanjsi_el(tabela) #seznam desnih najmanjših el
    tabD = desni[::-1]  #obreno tabelo rešitev, ker smo pregledovali el od desne->levi
    
    #leve najmanjše el. dobimo enako kot desne, le da smo obrnili tabelo
    tabL = naslednji_najmanjsi_el(tabela[::-1]) #tabela levih najmanjših el
   
    rezultat = []
    for l,d in zip(tabL,tabD):  #absolutne razlike med el. iz dveh tab.
        rezultat.append(abs(l-d))
    return rezultat, max(rezultat)



