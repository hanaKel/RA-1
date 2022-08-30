def kovanci_v_vrsti_osnovna_funkcija(tab_kovancev):
    '''
    Funkcija kot argument dobi tabelo kovancev. 
    Vrne pa maksimalno vsoto.
    '''
    tab = [0, tab_kovancev[0]]
    indeks = 1
    vsota = tab_kovancev[0]
    for i in tab_kovancev[1:]:
        # določimo vsote 
        nova_vsota = i + tab[0]
        stara_vsota = tab[1]
        # naslednji kovanec vzamemo 
        if nova_vsota > stara_vsota:
            tab.append(nova_vsota)
            tab = tab[1:]
            vsota = nova_vsota
        # naslednjega kovanca ne vzamemo 
        else:
            tab.append(stara_vsota)
            tab = tab[1:]
            vsota = stara_vsota
        indeks += 1
    return vsota


print(kovanci_v_vrsti_osnovna_funkcija([2, 34, 32]))