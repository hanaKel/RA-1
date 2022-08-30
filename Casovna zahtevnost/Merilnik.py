import time                         # Štoparica
import matplotlib.pyplot as plt     # Risanje grafov
import random                       # Za generiranje primerov


def oceni_potreben_cas(fun, gen_primerov, n, k):
    """ Funkcija oceni potreben čas za izvedbo funkcije `fun` na primerih
    dolžine `n`. Za oceno generira primere primerne dolžine s klicom
    `gen_primerov(n)`, in vzame povprečje časa za `k` primerov. """
    casi = list()
    for _ in range(k):
        primer = gen_primerov(n)
        cas = izmeri_cas(fun, primer)
        casi.append(cas)
    return sum(casi)/k
    

def izmeri_cas(fun, primer):
    """ Izmeri čas izvajanja funkcije `fun` pri argumentu `primer`. """

    # NAMIG: klic funkcije `time.time()` vam vrne trenuten čas. Če izmerite
    # čas pred izračunom funkcije in čas po končanem izračunu, vam razlika
    # časov pove čas izvajanja.
    pred = time.time()
    fun(primer)
    po = time.time()
    return (po - pred)
    # NAMIG: `k`-krat generirajte nov testni primer velikosti `n` z klicom
    # `gen_primerov(n)` in izračunajte povprečje časa, ki ga funkcija porabi za
    # te testne primere.



def narisi_in_pokazi_graf(fun, gen_primerov, sez_n, k=10):
    """ Funkcija nariše graf porabljenega časa za izračun `fun` na primerih
    generiranih z `gen_primerov`, glede na velikosti primerov v `sez_n`. Za
    oceno uporabi `k` ponovitev. """

    # NAMIG: preprost graf lahko narišemo z `plt.plot(sez_x, sez_y, 'r')`, ki z
    # rdečo črto poveže točke, ki jih definirata seznama `sez_x` in `sez_y`. Da
    # se graf prikaže uporabniku, uporabimo ukaz `plt.show()`. Za lepše grafe
    # si poglejte primere knjižnice [matplotlib.pyplot] (ki smo jo preimenovali
    # v `plt`).

    casi = [oceni_potreben_cas(fun, gen_primerov, n, k) for n in sez_n]
    plt.plot(sez_n, casi, 'r')
    plt.show()


def izpisi_case(fun, gen_primerov, sez_n, k=10):
    """ Funkcija izpiše tabelo časa za izračun `fun` na primerih generiranih z
    `gen_primerov`, glede na velikosti primerov v `sez_n`. Za oceno uporabi `k`
    ponovitev. """

    # Seznam časov, ki jih želimo tabelirati
    casi = [oceni_potreben_cas(fun, gen_primerov, n, k) for n in sez_n]

    # za lepšo poravnavo izračunamo širino levega stolpca
    pad = max(len(str(n)) for n in sez_n)
    

    # izpiši glavo tabele
    """ POJASNILO: če uporabimo `{:n}` za `.format`, bo metoda vstavila
    argument, in nato na desno dopolnila presledke, dokler ni skupna dolžina
    niza enaka vrednosti `n`. Če želimo širino prilagoditi glede na neko
    spremenljivko, to storimo kot prikazuje spodnja vrstica (torej z
    `{:{pad}}` kjer moramo nato podati vrednost za `pad`)."""
    print("{:{pad}} | Čas izvedbe".format("n", pad=pad))

    # horizontalni separator
    sep_len = pad + max(len(" | Čas izvedbe"), 3 + max(len(str(t)) for t in casi))  # DOPOLNITE KODO (črta naj bo široka kot najširša vrstica)
    print("-"*sep_len)

    # izpiši vrstice
    for n, t in zip(sez_n, casi):
        print("{:{pad}} | {}".format(str(n), t, pad = pad))
    # DOPOLNITE KODO

def prikaz_dveh_grafov(fun1, fun2, gen_primerov, sez_n, k=10):
    cas1 = [oceni_potreben_cas(fun1, gen_primerov, n, k) for n in sez_n]
    cas2 = [oceni_potreben_cas(fun2, gen_primerov, n, k) for n in sez_n]
    plt.plot(sez_n, cas1, 'r', label = 'f(x)')
    plt.plot(sez_n, cas2, 'g', label = 'g(x)')
    plt.legend (loc = 'upper left')
    plt.show()


def primerjaj_case(fun1, fun2, gen_primerov, sez_n, k=10):
    cas1 = [oceni_potreben_cas(fun1, gen_primerov, n, k) for n in sez_n]
    cas2 = [oceni_potreben_cas(fun2, gen_primerov, n, k) for n in sez_n]
    pad = max(len(str(n)) for n in sez_n)
    pad2 = max(len(" | Čas izvedbe f(x)"), max(len(str(t)) for t in cas1))
    print("{:{pad}} | Čas izvedbe f(x) | Čas izvedbe g(x)".format("n",'', pad=pad))
    sep_len = pad + max(len(" | Čas izvedbe f(x) | Čas izvedbe g(x)") + pad2, 6 + max(len(str(t)) for t in cas1) + max(len(str(t)) for t in cas2))
    print("-" * sep_len)
    pad2 = max(len("Čas izvedbe f(x)"), max(len(str(t)) for t in cas1))
    for n, t1, t2, in zip(sez_n, cas1, cas2):
        print("%-10d | %-10.20f | %.20f" % (n, t1, t2))
##        print("{:{pad}} | {:{pad2}} | {}".format(str(n), t1, t2, pad=pad, pad2=pad2))


# -----------------------------------------------------------------------------
# Nekaj hitrih testnih funkcij

def test_fun_lin(sez):
    # linearna časovna zahtevnost
    x = 0
    for n in sez:
        x += n
    return x


def test_fun_kvad(sez):
    # kvadratična časovna zahtevnost
    x = 0
    for n in sez:
        for n in sez:
            x += n
    return x


def test_gen_sez(n):
    # generira seznam naključnih števil
    return [random.randint(-n, n) for _ in range(n)]


# Drugi zanimivi primeri
# - Funkcija `sorted`, ki uredi seznam. Na grafu se lahko opazi da ni linearna,
#   temveč O(n*log(n)).
# - Napišite naivno funkcijo za fibonaccijevo zaporedje, ki ima eksponentno
#   časovno zahtevnost.
# - Primerjajte vstavljanje na začetek `sez.insert(0, x)` in vstavljanje na
#   konec `sez.append(x)`. Za signifikantne rezultate potrebujete sezname
#   dolžine vsaj miljon, katerih vsebina ni pomembna, zato napišite nov
#   generator primerov, ki naredi seznam ničel (da bo koda hitrejša, saj je
#   `random.randint` počasen v primerjavi s konstanto).


# -----------------------------------------------------------------------------
# Ker je datoteka mišljena kot knjižnica, imejte vse 'primere izvajanja'
# zavarovane z `if __name__ == '__main__':`, da se izvedejo zgolj če datoteko
# poženemo in se ne izvedejo če datoteko uvozimo kot knjižnico.

# primer za izpis tabele (potrebuje nekaj 10 sekund)
##if __name__ == '__main__':
##    sez_n = [256*(2**k) for k in range(5)]  # eksponentni razmik med `n`ji
##    izpisi_case(test_fun_kvad, test_gen_sez, sez_n, k=10)
""" V RAZMISLEK: Vsaka vrstica v tabeli podvoji dolžino. Za kolikšen faktor se
poveča čas? Je to skladno s pričakovanji? """

# primer za izpis tabele (potrebuje nekaj 10 sekund)
##if __name__ == '__main__':
##    sez_n = [10000*k for k in range(10)]  # za graf je boljše linearno
##    narisi_in_pokazi_graf(test_fun_lin, test_gen_sez, sez_n, k=10)
""" V RAZMISLEK: Število ponovitev poskusov `k` nastavite na 1, 5, 10 in 20 ter
se prepričajte o smiselnosti večih poskusov. Ali nam graf prikazuje oceno
maksimalne, minimalne ali povprečne časovne zahtevnosti? Kaj bi vzeli namesto
povprečja za pričakovano časovno zahtevnost? """
if __name__ == '__main__':
##    sez_n = [100*k for k in range(10)]
##    prikaz_dveh_grafov(test_fun_lin, test_fun_kvad, test_gen_sez, sez_n, k=10)

    sez_n = [25 * (2 ** k) for k in range(5)]
    primerjaj_case(test_fun_kvad, test_fun_kvad, test_gen_sez, sez_n, k=10)

##    sez_n = [256 * (2 ** k) for k in range(5)]
##    primerjaj_case(test_fun_kvad, test_fun_kvad, test_gen_sez, sez_n, k=10)
