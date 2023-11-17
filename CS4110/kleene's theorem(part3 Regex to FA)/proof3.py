class FA:
    ''''''
    class State:
        def __init__(self, name, isStart = False, isFinal = False   ):
            self.name = name
            self.transition = {}
            self.isFinal = isFinal
            self.isStart = isStart

        '''add transition to state(state is string)
            :arg symbol: symbol(letter) to add transition
            :arg nextState: next state to add transition
            :return: none'''
        def addTransition(self, symbol, nextState):
            self.transition[symbol] = nextState

        '''get path from this state to next state by letter'''
        def getPath(self, letter):
            return self.transition[letter]
        def isFinal(self):
            return self.isFinal
        def isStart(self):
            return self.isStart

        def __str__(self):
            s = ""
            if self.isStart:  s += "-"
            if self.isFinal:  s += "+"
            return f"{s+self.name: >7}"

        def __repr__(self):
            s = ""
            if self.isStart:  s += "-"
            if self.isFinal:  s += "+"

            s = f"{s + self.name:>7} :  {self.transition['a'] if 'a' in self.transition else '':>5}  {self.transition['b'] if 'b' in self.transition else '':>5} "
            return s

    def __init__(self, name):
        self.start = []
        self.final = []
        self.name = name
        self._states = {}



    '''add state to FA
        :arg name: name of state
        :arg isStart: is start state
        :arg isFinal: is final state
        :return: none
    '''
    def addState(self, name, isStart = False, isFinal = False):
        ''' add state to Fallback
        '''

        if name not in self._states:
            self._states[name] = self.State(name,  isStart, isFinal)
        else:
            print("State already exists " + name)

        if isFinal:
            self.final.append(name)
        if isStart:
            self.start.append(name)

    '''get state from FA
        :arg name: name of state to get
        :return: state if already added, else print error'''
    def getState(self, name):
        ''' get state from FA
        '''
        if name in self._states:
            return self._states[name]
        else:
            return None
    '''get all states from FA'''
    def getAllStates(self):
        ''' get all states from FA
        '''
        return self._states

    '''add transition to state
        :arg name: name of state(if already added) to add transition
        :arg symbol: symbol(letter) to add transition
        :arg nextState: next state to add transition
        :return: none'''
    def addTransition(self, name, symbol, nextState):
        ''' add transition to state
        '''
        if name in self._states:
            self._states[name].addTransition(symbol, nextState)

    def __repr__(self):
        s = f"{' ':>7} :  {'a':>5}  {'b':>5} \n"
        for state in self._states:
            s += f"{repr(self._states[state])}\n"
        return s
class Example():
    '''example Regex to FA FA1 * FA2 = FA3 page 119'''
    def example1(self):
        '''define FA1 with x1 start, x2, x3 final'''
        fa1 = FA("FA1")
        fa1.addState("x1", isStart=True)
        fa1.addState("x2")
        fa1.addState("x3", isFinal=True)

        '''add trainsition in this format state: a b respectively
            1: 2 1
            2: 3 1
            3: 3 3
        '''
        fa1.addTransition("x1", "a", "x2")
        fa1.addTransition("x1", "b", "x1")
        fa1.addTransition("x2", "a", "x3")
        fa1.addTransition("x2", "b", "x1")
        fa1.addTransition("x3", "a", "x3")
        fa1.addTransition("x3", "b", "x3")


        #####################3 FA2

        fa2 = FA("FA2")
        fa2.addState("y1", isStart=True)
        fa2.addState("y2", isFinal=True)

        '''add trainsition in this format state: a b respectively
            1: 1 2
            2: 1 2
            
        '''
        fa2.addTransition("y1", "a", "y1")
        fa2.addTransition("y1", "b", "y2")
        fa2.addTransition("y2", "a", "y1")
        fa2.addTransition("y2", "b", "y2")

        return fa1, fa2

    '''example Regex to FA FA1 * FA2 = FA3 page 122'''
    def example2(self):
        '''define FA1 with x1 start, x2, x3 final'''
        fa1 = FA("FA1")
        fa1.addState("x1", isStart=True)
        fa1.addState("x2")
        fa1.addState("x3", isFinal=True)

        '''add trainsition in this format state: a b respectively
            1: 2 3
            2: 2 2
            3: 3 3
        '''
        fa1.addTransition("x1", "a", "x2")
        fa1.addTransition("x1", "b", "x3")
        fa1.addTransition("x2", "a", "x2")
        fa1.addTransition("x2", "b", "x2")
        fa1.addTransition("x3", "a", "x3")
        fa1.addTransition("x3", "b", "x3")

        ###################### FA2

        fa2 = FA("FA2")
        fa2.addState("y1", isStart=True)
        fa2.addState("y2", isFinal=True)

        '''add trainsition in this format state: a b respectively
            1: 1 2
            2: 1 2

        '''
        fa2.addTransition("y1", "a", "y1")
        fa2.addTransition("y1", "b", "y2")
        fa2.addTransition("y2", "a", "y1")
        fa2.addTransition("y2", "b", "y2")

        return fa1, fa2

    '''example Regex to FA FA1 * FA2 = FA3 page 124'''
    def example3(self):
        '''define FA1 with x1 start, x2, x3 final'''
        fa1 = FA("FA1")
        fa1.addState("x1", isStart=True, isFinal=True)
        fa1.addState("x2", isFinal=True)
        fa1.addState("x3")

        '''add trainsition in this format state: a b respectively
            1: 2 1
            2: 3 1
            3: 3 3
        '''
        fa1.addTransition("x1", "a", "x2")
        fa1.addTransition("x1", "b", "x1")
        fa1.addTransition("x2", "a", "x3")
        fa1.addTransition("x2", "b", "x1")
        fa1.addTransition("x3", "a", "x3")
        fa1.addTransition("x3", "b", "x3")

        ###################### FA2

        fa2 = FA("FA2")
        fa2.addState("y1", isStart=True)
        fa2.addState("y2", isFinal=True)

        '''add trainsition in this format state: a b respectively
            1: 2 2
            2: 1 1
            '''
        fa2.addTransition("y1", "a", "y2")
        fa2.addTransition("y1", "b", "y2")
        fa2.addTransition("y2", "a", "y1")
        fa2.addTransition("y2", "b", "y1")

        return fa1, fa2


class proof3(FA):

    def __init__(self, name):
        super().__init__(name)
        self.org = {}



    def __addState__(self, name, *args, isStart = False, isFinal  = False):
        if name not in self._states:
            self.addState(name, isStart, isFinal)
            self.org[name] = sorted(args[0]) # used for __findState__ list comparison, dont add [0]


    def __findState__(self, *args):
        '''check if *arg is an array, if not return none'''
        if not isinstance(args[0], list):
            return None
        for state in self.org:
            if self.org[state] == sorted(args[0]):
                return state
        return None

    def __getDestState__(self, fa, state, letter):
        if state in fa._states:
            return fa._states[state].getPath(letter)



    def connectLanguage(self, fa1, fa2):
        self.fa1 = fa1
        self.fa2 = fa2

        '''use queue to store state to check, iterative recursive'''
        q = []
        ti = 't1'
        xi = fa1._states[fa1.start[0]]
        orgState = [xi.name]
        if fa1._states[fa1.start[0]].isFinal:
            orgState.append(fa2.start[0])
        self.__addState__(ti, orgState,isStart=True )
        q.append(ti)

        while len(q) > 0:
            '''get state from queue'''
            state = q.pop(0)
            stateArray = self.org[state]
            for l in ['a', 'b']:
                nextStates = []
                isFa2Final = False

                for s in stateArray:
                    nextS = self.__getDestState__(fa1, s, l) or self.__getDestState__(fa2, s, l)

                    if nextS not in nextStates:
                        nextStates.append(nextS)
                        if fa1.getState(nextS) != None:
                            if fa1._states[nextS].isFinal:
                                nextStates.append(fa2.start[0])
                        #check for fa2 final state
                        if fa2.getState(nextS) != None:
                            if fa2._states[nextS].isFinal:
                                isFa2Final = True

                tNext = self.__findState__(nextStates)
                if tNext == None:
                    tNext = 't' + str(len(self.org) + 1)
                    self.__addState__(tNext, nextStates, isFinal=isFa2Final)

                    q.append(tNext)
                self.addTransition(state, l, tNext)





    '''get state status, if start, final, start and final'''
    def getStateStatus(self, name):
        s = f""
        states = name
        for st in states:
            sign = ""
            if self.fa1.getState(st) != None:
                sign += "-" if self.fa1._states[st].isStart else ""
                sign += "+" if self.fa1._states[st].isFinal else ""
            if self.fa2.getState(st) != None:
                sign += "-" if self.fa2._states[st].isStart else ""
                sign += "+" if self.fa2._states[st].isFinal else ""
            s += f"{sign + st: >7} "
        return s

    '''print new state of FA3 in accordance to FA1 and FA2 state in a table format'''

    def printState(self):
        print('Orginal State')
        for o in self.org:
            print(f"{self._states[o]} : "
                  f"{self.getStateStatus(self.org[o])} ")

    def __repr__(self):
        print("FA3 table")
        s = f"{' ':>7} :  {'a':>5}  {'b':>5} \n"
        for state in self._states:
            s += f"{repr(self._states[state])}\n"
        return s



def main():
    '''print hellworld'''


    fa = proof3("test")
    fa1, fa2 = Example().example2()
    fa.connectLanguage(fa2, fa1)
    print("FA1")
    print(fa1)
    print("FA2")
    print(fa2)

    print(fa)
    fa.printState()
    ################call main#####################
if __name__ == "__main__":
    main()  # call main function