# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

'''
CS4580
introduction to pythona assignment

'''


def part0(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'{name}')  # Press Ctrl+8 to toggle the breakpoint.


'''
Part 1 (from: https://projecteuler.net/problem=20)
n! means n × (n − 1) × ... × 3 × 2 × 1

For example, 10! = 10 × 9 × ... × 3 × 2 × 1 = 3628800,
and the sum of the digits in the number 10! is 3 + 6 + 2 + 8 + 8 + 0 + 0 = 27.

Find the sum of the digits in the number 100!

Write the answer as:

Part 1: <answer>
'''


def part1(n):
    fact = 1
    for i in range(1, n + 1):
        fact *= i
    rem = fact
    tot = 0
    while rem >= 1:
        div = rem % 10
        rem = rem // 10
        tot += div
    print(f'Part 1: {tot}')


'''
The Fibonacci sequence is defined by the recurrence relation:

Fn = Fn−1 + Fn−2, where F1 = 1 and F2 = 1.
Hence the first 12 terms will be:

F1 = 1
F2 = 1
F3 = 2
F4 = 3
F5 = 5
F6 = 8
F7 = 13
F8 = 21
F9 = 34
F10 = 55
F11 = 89
F12 = 144
The 12th term, F12, is the first term to contain three digits.

What is the index of the first term in the Fibonacci sequence to contain 1000 digits?

Hint: Do not use recursion!'''


def part2(l):
    fn_1 = 1
    fn = 1
    i = 2
    while len(str(fn)) < l:
        temp = fn
        fn = fn + fn_1
        fn_1 = temp
        i += 1
    print(f"Part 2: {i}")


'''A palindromic number reads the same both ways. The largest palindrome made from the product of two 2-digit numbers is 9009 = 91 × 99.

Find the largest palindrome made from the product of two 3-digit numbers.

Write the answer as:

Part 3: <answer>
'''


def part3():
    def isPad(x):
        s = str(x)
        for i in range(len(s) // 2):
            if s[i] != s[len(s) - 1 - i]:
                return False
        return True

    maxx = 0
    for i in range(100, 1000):
        for j in range(i, 1000):
            pro = i * j
            if isPad(pro):
                if pro > maxx: maxx = pro
    print(f"Part 3: {maxx}")


if __name__ == '__main__':
    part0('Huy D. Nguyen')
    part1(100)
    part2(1000)
    part3()
