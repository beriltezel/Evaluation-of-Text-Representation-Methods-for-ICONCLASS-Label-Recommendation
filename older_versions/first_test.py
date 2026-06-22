from iconclass import init

ic = init()
a = ic["53A2"]

print(a)
print(repr(a))
print(a())
print(a('de'))

for child in a:
    print(child('it'))

for p in a.path():
    print(repr(p))

for child in a:
    print(child())