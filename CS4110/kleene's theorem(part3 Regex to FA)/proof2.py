'''Class for creating a Finite Automata'''
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
            print("State does not exist")
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

###################### Kleene's Theorem  ########################################
''''
    Class to create a Kleene's Theorem  proof of part 2
    if there is a FA1 and FA2, then there is a FA3 that is a language of FA1 and FA2
    FA3 = FA1 + FA2
    
    inherete from FA
'''
class proof2Class(FA):
    def __init__(self, name):
        super().__init__(name)
        self.org = {}

    def __addState__(self, name, x, y, isStart = False, isFinal = False):
        if name not in self._states:
            self.org[name] = [x, y]
            super().addState(name, isStart, isFinal)

    '''for finding wether or not state w is already created from x and y'''
    def __findState__(self, x, y):
        for s, v in self.org.items():
            if x in v and y in v:
                return s
        return None

    '''for getting destination state from current state and letter'''
    def __getDestState__(self,fa, name, letter):
        if name in fa._states:
            return fa._states[name].getPath(letter)
        return None

    '''add two Language FA1, FA2 to create FA3
        :arg fa1: FA1
        :arg fa2: FA2
        :return: none'''

    def addLanguage(self, fa1, fa2):
        q = []
        self.fa1 = fa1
        self.fa2 = fa2
        wi = 't1'
        xi = fa1._states[fa1.start[0]]
        yi = fa2._states[fa2.start[0]]
        self.__addState__(wi, xi.name, yi.name, True, xi.isFinal or yi.isFinal)

        #add wi start to queue
        q.append(wi)

        while len(q) > 0:
            wi = q.pop(0)
            xi = fa1.getState(self.org[wi][0])
            yi = fa2.getState(self.org[wi][1])

            for l in ['a', 'b']:
                xNext = self.__getDestState__(fa1, xi.name, l)
                yNext = self.__getDestState__(fa2, yi.name, l)

                if xNext is not None and yNext is not None:
                    wNext = self.__findState__(xNext, yNext)
                    if wNext is None:
                        wNext = 't' + str(len(self._states) + 1)
                        self.__addState__(wNext, xNext, yNext, False, fa1.getState(xNext).isFinal or fa2.getState(yNext).isFinal)
                        q.append(wNext)
                    self.addTransition(wi, l, wNext)
    def clean(self):
        self.org = {}
        self._states = {}
        self.fa2 = None
        self.fa1 = None

    '''get state status, if start, final, start and final'''
    def getStateStatus(self, name):
        if name in self._states:
            s = ""
            if self._states[name].isStart:  s += "-"
            if self._states[name].isFinal:  s += "+"
            return f"{s + name: >7}"

    '''print new state of FA3 in accordance to FA1 and FA2 state in a table format'''
    def printOrg(self):
        print('Orginal State')
        for o in self.org:
            print(f"{self.getStateStatus(o):>5} : "
                  f"{self.fa1.getState(self.org[o][0])} {self.fa2.getState(self.org[o][1])}")

    def __repr__(self):
        print("FA3 table")
        s = f"{' ':>7} :  {'a':>5}  {'b':>5} \n"
        for state in self._states:
            s += f"{repr(self._states[state])}\n"
        return s


#############################seperator function#################################
#FA1

class Example:
    '''
    example proof of Regex to FA page 109
    '''
    def example1(self):
        def FA1():
                fa1 = FA("fa1")
                fa1.addState("x1",True)
                fa1.addState("x2")
                fa1.addState("x3",isFinal=True)

                fa1.addTransition("x1","a","x2")
                fa1.addTransition("x1","b","x1")
                fa1.addTransition("x2","a","x3")
                fa1.addTransition("x2","b","x1")
                fa1.addTransition("x3","a","x3")
                fa1.addTransition("x3","b","x3")

                print(fa1)
                return fa1
            #FA2
        def FA2():
                fa2 = FA("fa2")

                '''create four states with letter y, y1 is start state and final state'''
                fa2.addState("y1",True,True)
                fa2.addState("y2")
                fa2.addState("y3")
                fa2.addState("y4")

                '''add transitio with format where input is  state: a, b respectively
                    1: 3,2
                    2: 4,1
                    3:1,4
                    4:2, 3
                '''
                fa2.addTransition("y1","a","y3")
                fa2.addTransition("y1","b","y2")
                fa2.addTransition("y2","a","y4")
                fa2.addTransition("y2","b","y1")
                fa2.addTransition("y3","a","y1")
                fa2.addTransition("y3","b","y4")
                fa2.addTransition("y4","a","y2")
                fa2.addTransition("y4","b","y3")

                print(fa2)
                return fa2

        return FA1(), FA2()

    '''Example inside proof of theore 6 page 113'''
    def example2(self):
        def FA1():
            fa1 = FA("fa1")
            fa1.addState("x1",True)
            fa1.addState("x2")
            fa1.addState("x3",isFinal=True)

            fa1.addTransition("x1","a","x2")
            fa1.addTransition("x1","b","x1")
            fa1.addTransition("x2","a","x3")
            fa1.addTransition("x2","b","x1")
            fa1.addTransition("x3","a","x3")
            fa1.addTransition("x3","b","x3")

            print(fa1)
            return fa1
        #FA2
        def FA2():
            fa2 = FA("fa2")
            '''add two state y1 and y2, y1 start, y2 final'''
            fa2.addState("y1",True,isFinal=True)
            fa2.addState("y2")

            '''add four state with format state: a, b respectively
                1:1 2
                2:1 2
            '''
            fa2.addTransition("y1","a","y2")
            fa2.addTransition("y1","b","y1")
            fa2.addTransition("y2","a","y1")
            fa2.addTransition("y2","b","y2")

            print(fa2)
            return fa2

        return FA1(), FA2()

    '''Example inside proof of theore 6 page 115'''
    def example3(self):
        def FA1():
            '''add two states x1, x2, x1 start, x2 final'''
            fa1 = FA("fa1")
            fa1.addState("x1",True)
            fa1.addState("x2",isFinal=True)

            '''add four state with format state: a, b respectively
                1:2 1
                2:2 1'''
            fa1.addTransition("x1","a","x2")
            fa1.addTransition("x1","b","x1")
            fa1.addTransition("x2","a","x2")
            fa1.addTransition("x2","b","x1")

            print(fa1)
            return fa1
        #FA2
        def FA2():
            fa2 = FA("fa2")
            '''create two states y1, y2, y1 start, y2 final'''
            fa2.addState("y1",True)
            fa2.addState("y2",isFinal=True)

            '''add four state with format state: a, b respectively
                1:2 2
                2:1 1
            '''
            fa2.addTransition("y1","a","y2")
            fa2.addTransition("y1","b","y2")
            fa2.addTransition("y2","a","y1")
            fa2.addTransition("y2","b","y1")

            print(fa2)
            return fa2
        return FA1(), FA2()

    '''Example inside proof of theore 6 page 115-116'''
    def example4(self):

        def FA1():
            '''create FA, 2 states, x1 start, x2 final'''
            fa1 = FA("fa1")
            fa1.addState("x1",True)
            fa1.addState("x2",isFinal=True)

            '''add four state with format state: a, b respectively
                1:2 1
                2:2 1                
            '''
            fa1.addTransition("x1","a","x2")
            fa1.addTransition("x1","b","x1")
            fa1.addTransition("x2","a","x2")
            fa1.addTransition("x2","b","x1")

            print(fa1)
            return fa1

        #FA2
        def FA2():
            '''create FA, 2 states, y1 start, y2 final'''
            fa2 = FA("fa2")
            fa2.addState("y1",True)
            fa2.addState("y2",isFinal=True)

            '''add four state with format state: a, b respectively
                1:1 2
                2:1 2
            '''
            fa2.addTransition("y1","a","y1")
            fa2.addTransition("y1","b","y2")
            fa2.addTransition("y2","a","y1")
            fa2.addTransition("y2","b","y2")

            print(fa2)
            return fa2

        return FA1(), FA2()

    '''Exercise 3  page 143'''
    def exercise3(self):
        def FA1():
            '''create two state w1 start, w2 final'''
            fa1 = FA("fa1")
            fa1.addState("w1",True)
            fa1.addState("w2",isFinal=True)

            ''' add four state with format state: a, b respectively
                1:2 1   
                2:2 1
            '''
            fa1.addTransition("w1","a","w2")
            fa1.addTransition("w1","b","w1")
            fa1.addTransition("w2","a","w2")
            fa1.addTransition("w2","b","w1")
            print("FA1")
            print(fa1)
            return fa1

        def FA2():
            '''three states, x1 start, x2, x3 final'''
            fa2 = FA("fa2")
            fa2.addState("x1",True)
            fa2.addState("x2")
            fa2.addState("x3",isFinal=True)

            '''add four state with format state: a, b respectively
                1:2 1
                2:2 3
                3:3 3
            '''
            fa2.addTransition("x1","a","x2")
            fa2.addTransition("x1","b","x1")
            fa2.addTransition("x2","a","x2")
            fa2.addTransition("x2","b","x3")
            fa2.addTransition("x3","a","x3")
            fa2.addTransition("x3","b","x3")
            print("FA2")
            print(fa2)
            return fa2

        def FA3():
            '''four states, y1 start, y2, y3, y4 final'''
            fa3 = FA("fa3")
            fa3.addState("y1",True)
            fa3.addState("y2")
            fa3.addState("y3")
            fa3.addState("y4",isFinal=True)

            '''add four state with format state: a, b respectively
            
                1: 2 3
                2: 4 3
                3: 2 4
                4: 4 4 '''
            fa3.addTransition("y1","a","y2")
            fa3.addTransition("y1","b","y3")
            fa3.addTransition("y2","a","y4")
            fa3.addTransition("y2","b","y3")
            fa3.addTransition("y3","a","y2")
            fa3.addTransition("y3","b","y4")
            fa3.addTransition("y4","a","y4")
            fa3.addTransition("y4","b","y4")
            print("FA3")
            print(fa3)
            return fa3

        def FA4():
            '''two states z1 start final, z2'''
            fa4 = FA("fa4")
            fa4.addState("z1",True,isFinal=True)
            fa4.addState("z2")

            '''add four state with format state: a, b respectively
                1: 2 1
                2: 2 2
            '''
            fa4.addTransition("z1","a","z2")
            fa4.addTransition("z1","b","z1")
            fa4.addTransition("z2","a","z2")
            fa4.addTransition("z2","b","z2")
            print("FA4")
            print(fa4)
            return fa4

        return FA1(), FA2(), FA3(), FA4()

    def excercise5(self):
        '''create FA, 2 states, x1 start, x2, x3 final'''
        fa1 = FA("fa1")
        fa1.addState("x1",True)
        fa1.addState("x2")
        fa1.addState("x3",isFinal=True)

        '''add four state with format state: a, b respectively
            1:2 1
            2:3 1
            3: 3 3
        '''
        fa1.addTransition("x1","a","x2")
        fa1.addTransition("x1","b","x1")
        fa1.addTransition("x2","a","x3")
        fa1.addTransition("x2","b","x1")
        fa1.addTransition("x3","a","x3")
        fa1.addTransition("x3","b","x3")


        '''create FA, 2 states, y1 start, y2 final'''
        fa2 = FA("fa2")
        fa2.addState("y1",True)
        fa2.addState("y2",isFinal=True)

        '''add four state with format state: a, b respectively
        1:1 2
        2:1 2
        '''
        fa2.addTransition("y1","a","y1")
        fa2.addTransition("y1","b","y2")
        fa2.addTransition("y2","a","y1")
        fa2.addTransition("y2","b","y2")

        return fa1, fa2


    '''example for chapter 8 regular languagepage 176(191)'''
    def example6(self):
        '''create fa 3 states, x1 start-final, x2 final, x3'''
        fa1 = FA("fa1")
        fa1.addState("x1",True,isFinal=True)
        fa1.addState("x2",isFinal=True)
        fa1.addState("x3")

        '''add four state with format state: a, b respectively
            1:2 1
            2:3 1
            3: 3 3
        '''
        fa1.addTransition("x1","a","x2")
        fa1.addTransition("x1","b","x1")
        fa1.addTransition("x2","a","x3")
        fa1.addTransition("x2","b","x1")
        fa1.addTransition("x3","a","x3")
        fa1.addTransition("x3","b","x3")

        '''create FA, 2 states, y1 start, y2 final'''
        fa2 = FA("fa2")
        fa2.addState("y1",True)
        fa2.addState("y2",isFinal=True)


        '''add states, format state: a, b respectively
            1: 2 1
            2: 1 2
        '''
        fa2.addTransition("y1","a","y2")
        fa2.addTransition("y1","b","y1")
        fa2.addTransition("y2","a","y1")
        fa2.addTransition("y2","b","y2")

        return fa1, fa2
    '''chapter 9 example, start with a, and end with a'''
    def example7(self):
        '''define fa1 with 3 states, x1 start, x2 final, x3'''
        fa1 = FA("fa1")
        fa1.addState("x1",True)
        fa1.addState("x2",isFinal=True)
        fa1.addState("x3")

        '''add states, format state: a, b respectively
            1:2 3
            2:2 2
            3:3 3
        '''
        fa1.addTransition("x1","a","x2")
        fa1.addTransition("x1","b","x3")
        fa1.addTransition("x2","a","x2")
        fa1.addTransition("x2","b","x2")
        fa1.addTransition("x3","a","x3")
        fa1.addTransition("x3","b","x3")


        '''define fa2 with 2 states, y1 start, y2 final'''
        fa2 = FA("fa2")
        fa2.addState("y1",True)
        fa2.addState("y2",isFinal=True)


        '''add states, format state: a, b respectively
            1:2 1
            2:2 1
        '''
        fa2.addTransition("y1","a","y2")
        fa2.addTransition("y1","b","y1")
        fa2.addTransition("y2","a","y2")
        fa2.addTransition("y2","b","y1")

        return fa1, fa2
    def example8(self):
        '''define fa1 with 3 states, x1 start, x2 final, x3'''
        fa1 = FA("fa1")
        fa1.addState("y1",True)
        fa1.addState("y2")
        fa1.addState("y3",isFinal=True)

        '''add states, format state: a, b respectively
            1: 2 1
            2: 3 1
            3: 3 3
        '''
        fa1.addTransition("y1","a","y2")
        fa1.addTransition("y1","b","y1")
        fa1.addTransition("y2","a","y3")
        fa1.addTransition("y2","b","y1")
        fa1.addTransition("y3","a","y3")
        fa1.addTransition("y3","b","y3")



        '''define fa2 with 2 states, y1 start, y2 final'''
        fa2 = FA("fa2")
        fa2.addState("x1",True)
        fa2.addState("x2",isFinal=True)


        '''add states, format state: a, b respectively
            1:2 1
            2:2 1
        '''
        fa2.addTransition("x1","a","x2")
        fa2.addTransition("x1","b","x1")
        fa2.addTransition("x2","a","x2")
        fa2.addTransition("x2","b","x1")

        return fa2, fa1

    #exercise chapter 11 p217
    def example9(self):
        ''' fa1 with 5 stats, x1 start end, x2 end, x3 end, x4 end, x5'''
        fa1 = FA("fa1")
        fa1.addState("x1",True,isFinal=True)
        fa1.addState("x2",isFinal=True)
        fa1.addState("x3",isFinal=True)
        fa1.addState("x4",isFinal=True)
        fa1.addState("x5")

        '''add states, format state: a, b respectively
            1:2 3
            2:2 4
            3:5 3
            4:5 4
            5:5 5'''
        fa1.addTransition("x1","a","x2")
        fa1.addTransition("x1","b","x3")
        fa1.addTransition("x2","a","x2")
        fa1.addTransition("x2","b","x4")
        fa1.addTransition("x3","a","x5")
        fa1.addTransition("x3","b","x3")
        fa1.addTransition("x4","a","x5")
        fa1.addTransition("x4","b","x4")
        fa1.addTransition("x5","a","x5")
        fa1.addTransition("x5","b","x5")

        ''' fa2 with 3 stats, y1 start end, y2 end, y3 '''
        fa2 = FA("fa2")
        fa2.addState("y1",True,isFinal=True)
        fa2.addState("y2",isFinal=True)
        fa2.addState("y3")

        '''add states, format state: a, b respectively
        1:1 2
        2:3 2
        3:3 3'''
        fa2.addTransition("y1","a","y1")
        fa2.addTransition("y1","b","y2")
        fa2.addTransition("y2","a","y3")
        fa2.addTransition("y2","b","y2")
        fa2.addTransition("y3","a","y3")
        fa2.addTransition("y3","b","y3")

        return fa1, fa2

    # exercise 2 chapter 11 p217
    def example10(self):
        '''fa1 with 2 states, x1 start , x2 end'''
        fa1 = FA("fa1")
        fa1.addState("x1",True)
        fa1.addState("x2",isFinal=True)

        '''add states, format state: a, b respectively
            1:2 1
            2:2 1'''
        fa1.addTransition("x1","a","x2")
        fa1.addTransition("x1","b","x1")
        fa1.addTransition("x2","a","x2")
        fa1.addTransition("x2","b","x1")

        '''fa2 with 3 states, y1 start, y2 end, y3'''
        fa2 = FA("fa2")
        fa2.addState("y1",True)
        fa2.addState("y2",isFinal=True)
        fa2.addState("y3")

        '''add states, format state: a, b respectively
            1:2 3
            2:2 3
            3:2 3'''
        fa2.addTransition("y1","a","y2")
        fa2.addTransition("y1","b","y3")
        fa2.addTransition("y2","a","y2")
        fa2.addTransition("y2","b","y3")
        fa2.addTransition("y3","a","y2")
        fa2.addTransition("y3","b","y3")

        return fa1, fa2


##############################main function#####################################


def main():
    # print("Huy Nguyen")
    # print("CS4110")
    # print("Kleene's Theorem")


    example = Example()
    fa1, fa2 = example.example10()
    print("*" * 50)
    print("FA1")
    print(fa1)
    print("*" * 50)
    print("FA2")
    print(fa2)
    print("*" * 50)
    fa = proof2Class("fa")
    #bug can only run one add at a time for one fa
    #fa.addLanguage(fa1, fa2)
    #fa.addLanguage(fa1, fa3)
    fa.addLanguage(fa1, fa2)
    print(fa)
    fa.printOrg()








main()