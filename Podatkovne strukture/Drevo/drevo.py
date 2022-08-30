class Drevo:

    def __init__(self, *args, **kwargs):
        if args:
            self.prazno = False
            self.vsebina = args[0]
            self.levo = kwargs.get('levo', Drevo())
            self.desno = kwargs.get('desno', Drevo())
        else:
            self.prazno = True

    def __repr__(self, zamik = ''):
        if self.prazno:
          return 'Drevo()'.format(zamik)
        elif self.levo.prazno and self.desno.prazno:
          return 'Drevo({1})'.format(zamik, self.vsebina)
        else:
          return 'Drevo({1},\n{0}      levo = {2},\n{0}      desno = {3})'.\
            format(
              zamik,
              self.vsebina,
              self.levo.__repr__(zamik + '             '),
              self.desno.__repr__(zamik + '              ')
            )

    def __eq__(self, other):
        return ((self.prazno and other.prazno) or
                (not self.prazno and not other.prazno and
                 self.vsebina == other.vsebina and
                 self.levo == other.levo and
                 self.desno == other.desno))
