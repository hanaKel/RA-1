def problem_preprostega_nahrbtnika(tab_predmetov, volumen_nahrbtnika):
    '''
    Vrne maksimalno vrednost nahrbtnika. 
    '''
    urejena_tab = sorted(tab_predmetov, key=lambda x:x[0]/x[1], reverse=True)
    print(urejena_tab)
    volumen = 0
    naslednji_volumen = urejena_tab[0][1]
    cena = 0
    kje = 0
    while naslednji_volumen < volumen_nahrbtnika:
        naslednji_volumen += urejena_tab[kje+1][1]
        volumen += urejena_tab[kje][1]
        cena += urejena_tab[kje][0]
        kje += 1
    
    ostanek_prostora = volumen_nahrbtnika - volumen
    cena += (urejena_tab[kje][0]/urejena_tab[kje][1]) * ostanek_prostora

    return cena 

print(problem_preprostega_nahrbtnika([(2, 5), (30, 5)], 6))
