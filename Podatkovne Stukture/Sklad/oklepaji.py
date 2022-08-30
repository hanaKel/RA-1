oklep = {
    '(': ')',
    '{': '}',
    '[': ']',
    '<': '>'
}


def oklepaji(niz):
    '''Preverim, če so oklepaji pravilno uporabljeni'''
    sklad = Sklad()  # naredim prazen sklad
    for el in niz:
        # s for zanko preverim vse elemente v nizu
        if el in oklep.keys() :
            #predklepaj dodam v sklad
            sklad.vstavi(el)
        elif el in oklep.values(): #enako za zaklepaje preverim če je v sl. oklep
            if sklad.prazen(): # če ni nič v novem skladu vrne False
                return False   
            vrh = sklad.poberi() # dobivam vrhnji el iz sklada
            if el != oklep[vrh]:
                return False # funkcija vrne False, če element na vrhu
                             # ni tak, kot tisti ki bi moral biti na vrsti
    return sklad.prazen()
