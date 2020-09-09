#!/usr/bin/env python3

import re

braces = re.compile(r'{([^}]*)}')
math   = re.compile(r'\$([^$]*)\$')
value  = re.compile(r'{\\it\s+[dD]efault:\s*}\s*')
descr  = re.compile(r'{\\it\s+[dD]escription:\s*}\s*')
types  = re.compile(r'{\\it\s+[pP]ossible\s*[vV]alues:\s*}\s*')

# split on braces : str -> [str]
def get_args(line):
    return braces.findall(line)

def fix_math(line):
    m = math.search(line)
    if m is None:
        return line

    line = line[:m.start()] + ":math:`%s`"%m[0][1:-1] + fix_math( line[m.end():] )
    return line

class Prm:
    def __init__(self, name):
        self.desc  = []
        self.state = "clear"
        with open(name) as f:
            for line in f:
                self.parse( line.replace(r"\_", "_") )
        self.clear()

    def clear(self):
        if self.state == "desc":
            for line in self.desc:
                line = line.strip()
                if line != "":
                    print("   " + line)
            print("")
        self.desc = []
        self.state = "clear"

    def parse(self, line):
        if r"\subsection" in line:
            name = get_args(line)[0]
            if r'\tt' in name:
                name = name[name.index(r'\tt')+4:]
            print(name)
            print('-'*len(name))
            print('')
            return self.clear()

        if '[prmindexfull]' in line:
            name = get_args(line)[0]
            print(".. data:: %s\n"%name.replace('!', '::'))
            return self.clear()

        d = value.search(line)
        if d is not None:
            print("   :value: %s"%line[d.end():].strip())
            return self.clear()

        d = types.search(line)
        if d is not None:
            line = line[d.end():].strip()
            if "boolean" in line:
                line = "true | false"
            else:
                line = fix_math(line)
            print("   :type:  %s\n"%line)
            return self.clear()

        d = descr.search(line)
        if d is not None:
            self.state = "desc"
            line = line[d.end():]

        if self.state == "desc":
            self.desc.append(line)

# Print out the file header.
print(""".. default-domain:: py

.. _parameters:

DFT-FE Input Parameter File Reference
=====================================
""")

p = Prm("parameters.tex")

