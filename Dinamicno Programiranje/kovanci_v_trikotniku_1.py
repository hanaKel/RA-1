cache = {}
pot = []
stevec_levo = 0
stevec_desno = 0
lahko_levo = True
lahko_desno = True
def kovanci_v_trikotniku_osnovna_funkcija(trikotnik, stevec_levo, stevec_desno, lahko_levo, lahko_desno):
    '''
    Vrne maksimalno vsoto poti v trikotniku
    '''

    if stevec_levo >= 2:
        lahko_levo = False
    else:
        lahko_levo = True
    if stevec_desno >= 2:
        lahko_desno = False
    else:
        lahko_desno = True 


    if trikotnik in cache:
        return cache[trikotnik]

    if len(trikotnik) == 1:
        return trikotnik[0][0]

    levi_podtrikotnik = ()
    for par in trikotnik[1:]:
        levi_podtrikotnik += (par[:-1],)
    desni_podtrikotnik = ()
    for par in trikotnik[1:]:
        desni_podtrikotnik += (par[1:],)

    if lahko_levo:
        desno = 0
        stevec_desno = 0
        levo = kovanci_v_trikotniku_osnovna_funkcija(levi_podtrikotnik, stevec_levo+1, stevec_desno, lahko_levo, lahko_desno)
    elif lahko_desno:
        levo = 0
        stevec_levo = 0
        desno = kovanci_v_trikotniku_osnovna_funkcija(desni_podtrikotnik, stevec_levo, stevec_desno+1, lahko_levo, lahko_desno)

    vsota = trikotnik[0][0] + max(levo, desno)

    cache[trikotnik] = vsota
    return vsota

print(kovanci_v_trikotniku_osnovna_funkcija(((3,), (5, 8), (6, 3, 1), (30, 2, 2, 29)), stevec_levo, stevec_desno, lahko_levo, lahko_desno))
print(pot)