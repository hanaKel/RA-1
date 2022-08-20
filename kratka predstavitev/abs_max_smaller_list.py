def maxRazlika(tab):
        '''
            Funkcija ustvari dve tabeli iz desnih in levih najmanjših najbližjih elemetov.
            Izračuna absolutne razlike med elementi iz obeh tabel in vrne največjo razliko.
        '''
        if tab == [] or len(tab)==1:
            return False
        
        desni=[] # v to tabelo shranimo desne najbližje in najmanjše vrednosti
        for i in range(0, len(tab)): #za vsak element iz tabele bomo poiskali najbližjega
            next = 0  # če manjšega ni 
            for j in range(i + 1, len(tab)): #vsak element desno od izbranega preverimo ali je manjši
                    #začnemo z naslednjim el.
                if tab[i] > tab[j]: #desni el. je manjši od trenutnega
                    next = tab[j]  #če smo ga našli ga shranimo v tabelo
                    break
            desni.append(next)
            
        levi = []
        obratniTab=tab[::-1]   #obrnemo tabelo in ponovno iščemo desne najbližje tako dobim leve najbližje
        for i in range(0, len(obratniTab)):
            next = 0
            for j in range(i + 1, len(obratniTab)):
                if obratniTab[i] > obratniTab[j]:
                    next = obratniTab[j]
                    break
            levi.append(next)  # na koncu še enkrat obrnjemo tabelo, da dobimo res leve najbližje 

        razlike=[]
        for l,d in zip(desni, levi[::-1]):  #med vsemi elementi iz novih tabel izračunamo absolutne razlike
            razlike.append(abs(l-d))
        print(" Tabela desnih najbližjih elementov: ",desni)
        print("Tabela levih najbližjih elementov: ",levi[::-1])
        print("Tabela absolutnih razlik med levimi in desnimi najbližjimi elementi: ",razlike)
        print("Največja absolutna razlika med levimi in desnimi najbližjimi elementi: ")
        return max(razlike)  # vrnemo največjo razliko

