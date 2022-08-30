import time                         # Štoparica

import matplotlib.pyplot as plt     # Risanje grafov
import random                       # Za generiranje primerov


def izmeri_cas(fun, primer):
    """Izmeri čas izvajanja funkcije `fun` pri argumentu `primer`."""
    # NAMIG: klic funkcije `time.time()` vam vrne trenuten čas. Če izmerite
    # čas pred izračunom funkcije in čas po končanem izračunu, vam razlika
    # časov pove čas izvajanja.

    raise NotImplementedError


def oceni_potreben_cas(fun, gen_primerov, n, k):
    """ Funkcija oceni potreben čas za izvedbo funkcije `fun` na primerih
    dolžine `n`. Za oceno generira primere primerne dolžine s klicom
    `gen_primerov(n)`, in vzame povprečje časa za `k` primerov. """

    # NAMIG: `k`-krat generirajte nov testni primer velikosti `n` s klicem
    # `gen_primerov(n)` in izračunajte povprečje časa, ki ga funkcija porabi za
    # te testne primere.

    raise NotImplementedError


def narisi_in_pokazi_graf(fun, gen_primerov, sez_n, k=10):
    """ Funkcija nariše graf porabljenega časa za izračun `fun` na primerih
    generiranih z `gen_primerov`, glede na velikosti primerov v `sez_n`. Za
    oceno uporabi `k` ponovitev. """

    # NAMIG: preprost graf lahko narišemo z `plt.plot(sez_x, sez_y, 'r')`, ki z
    # rdečo črto poveže točke, ki jih definirata seznama `sez_x` in `sez_y`. Da
    # se graf prikaže uporabniku, uporabimo ukaz `plt.show()`. Za lepše grafe
    # si poglejte primere knjižnice [matplotlib.pyplot] (ki smo jo preimenovali
    # v `plt`).

    raise NotImplementedError


def izpisi_case(fun, gen_primerov, sez_n, k=10):
    """ Funkcija izpiše tabelo časa za izračun `fun` na primerih generiranih z
    `gen_primerov`, glede na velikosti primerov v `sez_n`. Za oceno uporabi `k`
    ponovitev. """

    # Seznam časov, ki jih želimo tabelirati
    casi = []  # DOPOLNITE KODO

    # za lepšo poravnavo izračunamo širino levega stolpca
    pad = None  # DOPOLNITE KODO

    # izpiši glavo tabele
    """ POJASNILO: če uporabimo `{:n}` za `.format`, bo metoda vstavila
    argument, in nato na desno dopolnila presledke, dokler ni skupna dolžina
    niza enaka vrednosti `n`. Če želimo širino prilagoditi glede na neko
    spremenljivko, to storimo kot prikazuje spodnja vrstica (torej z
    `{:{pad}}` kjer moramo nato podati vrednost za `pad`)."""
    print("{:{pad}} | Čas izvedbe".format("n", pad=pad))
    # horizontalni separator
    sep_len = None  # DOPOLNITE KODO (črta naj bo široka kot najširša vrstica)
    print("-"*sep_len)

    # izpiši vrstice
    # DOPOLNITE KODO

    raise NotImplementedError


# -----------------------------------------------------------------------------
# Nekaj hitrih testnih funkcij

def test_fun_lin(sez):
    "Testna funkcija z linearno časovno zahtevnostjo."
    x = 0
    for n in sez:
        x += n
    return x


def test_fun_kvad(sez):
    "Testna funkcija s kvadratično časovno zahtevnostjo."
    x = 0
    for n in sez:
        for n in sez:
            x += n
    return x


def test_gen_sez(n):
    "Generira testni seznam dolžine n."
    return [random.randint(-n, n) for _ in range(n)]

# -----------------------------------------------------------------------------
# Ker je datoteka mišljena kot knjižnica, imejte vse 'primere izvajanja'
# zavarovane z `if __name__ == '__main__':`, da se izvedejo zgolj če datoteko
# poženemo in se ne izvedejo če datoteko uvozimo kot knjižnico.