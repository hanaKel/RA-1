
cache = {}
def kovanci_v_trikotniku_osnovna_funkcija(trikotnik):
    '''
    Vrne maksimalno vsoto poti v trikotniku
    '''

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

    print(levi_podtrikotnik)
    print(desni_podtrikotnik)
    vsota = trikotnik[0][0] + max(kovanci_v_trikotniku_osnovna_funkcija(levi_podtrikotnik), kovanci_v_trikotniku_osnovna_funkcija(desni_podtrikotnik))
    cache[trikotnik] = vsota
    return vsota

print(kovanci_v_trikotniku_osnovna_funkcija(((3,), (5, 8), (6, 3, 1), (30, 2, 2, 2))))
