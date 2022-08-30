def kovanci_v_vrsti(tab_kovancev):
    '''
    Funkcija kot argument dobi tabelo kovancev. 
    Vrne pa množico vseh rešitev (indeksi kovancev) ki vrnejo maksimalno vsoto.
    '''
    if len(tab_kovancev) == 0:
        return 0
    tab = [0, tab_kovancev[0]]

    mn_i = set()
    mn_i_1 = set()
    mn_i_2 = set()

    kovanci_i = (0,)
    kovanci_i_1 = ()
    kovanci_i_2 = ()

    mn_i.add(kovanci_i)
    mn_i_1.add(kovanci_i_1)
    mn_i_2.add(kovanci_i_2)

    indeks = 1
    for i in tab_kovancev[1:]:
        mn_i_2 = mn_i_1.copy()
        mn_i_1 = mn_i.copy()
        mn_i = set()
        # določimo vsote 
        nova_vsota = i + tab[0]
        stara_vsota = tab[1]
        # naslednji kovanec vzamemo 
        if nova_vsota > stara_vsota:
            tab.append(nova_vsota)
            tab = tab[1:]
            for kovanci in mn_i_2:
                mn_i.add(kovanci + (indeks,))
        # naslednjega kovanca ne vzamemo 
        elif nova_vsota < stara_vsota:
            tab.append(stara_vsota)
            tab = tab[1:]
            for kovanci in mn_i_1:
                mn_i.add(kovanci)
        # vrednosti sta enake, zato gledamo obe kombinaciji
        else:
            tab.append(stara_vsota)
            tab = tab[1:]
            # v mnozico dodamo obe možnosti
            for kovanci in mn_i_1:
                mn_i.add(kovanci)
            for kovanci in mn_i_2:
                mn_i.add(kovanci + (indeks,))

        indeks += 1
    return mn_i

# =======================================
# TESTI:
# =======================================

print(kovanci_v_vrsti([]))
print(kovanci_v_vrsti([2]))
print(kovanci_v_vrsti([2, 5, 3]))
print(kovanci_v_vrsti([2, 5, 3, 434]))
print(kovanci_v_vrsti([2, 2, 2, 2]))
