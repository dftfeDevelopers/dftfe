import os
import numpy as np
import periodictable as pdt

import pyprocar
from pyprocar.scripts import *
    
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import re

import xml.etree.ElementTree as ET
from yattag import Doc, indent
from scipy.integrate import simps

import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

class Plotters:
    def __init__(self, filesPath = None, bandsDatFile = None, kptsFile = None, 
                 coordinatesFile = None, latticeVecFile = None,pseudoPotFile = None, 
                 dosDataFile= None, eLimit = None, isPeriodic= None):
        self.filesPath = filesPath
        self.bandsDatFile = bandsDatFile
        self.kptsFile = kptsFile
        self.coordinatesFile = coordinatesFile
        self.latticeVecFile = latticeVecFile
        self.pseudoPotFile = pseudoPotFile
        self.dosDataFile = dosDataFile
        self.eLimit = eLimit
        self.isPeriodic = isPeriodic
        self.outdir = filesPath + "outfiles/"
        self.scaleFactor = 0.5291772105638411 # scale factor (bohr to angstrom)
        self.hartreeToEv = 27.211386024367243

        '''
        filesPath: str: path to the directory where the input files are stored,
        bandsDatFile: str: name of the file containing the bandstructure data,
        kptsFile: str: name of the file containing the kpoints data,
        coordinatesFile: str: name of the file containing the coordinates data,
        latticeVecFile: str: name of the file containing the lattice vectors data,
        pseudoPotFile: str: name of the file containing the pseudopotential data,
        dosDataFile: str: name of the file containing the DOS data,
        eLimit: list: energy limits for the bandstructure and DOS plots,
        isPeriodic: bool: True if the system is periodic, False otherwise
        '''

        with open(filesPath+"fermiEnergy.out") as f:
            for line in f:
                if (len(line) != 0):
                    self.eFermi = float(line.strip()) # in Ha

        if self.isPeriodic:
            self.ionPosVecType = "Direct"
        else:
            self.ionPosVecType = "Cartesian"

        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)

    def createPoscar(self):
        with open(self.outdir+'POSCAR', 'w') as f:
            f.write("This is a commented line\n") # Here you can write the name of the system (it's a commented line in POSCAR file)
            self.latticeVecs =[] # in Angstrom

            with open(self.filesPath+ self.latticeVecFile) as f1:
                f.write("{}\n".format(self.scaleFactor)) 
                lines = f1.readlines()
                for line in lines:
                    f.write(line)
                    self.latticeVecs.append((np.array(list(map(float,line.strip().split()))))*self.scaleFactor)

            self.ionData =[]   

            with open(self.filesPath+self.coordinatesFile) as f1:
                lines = f1.readlines()
                lines = list(map(lambda x: x.strip(), lines))
                while '' in lines: # To remove the blank lines
                    lines.remove('')
                
                atomNumCount ={}
                atomType ={}
                c = 1

                for i, line in enumerate(lines):
                    line = line.strip()
                    atomNum = int(line.strip().split()[0])
                    atomName = str(pdt.elements[atomNum])
                    temp = {}
                    temp['name'] = atomName
                    temp['positions'] = np.array(list(map(float,line.strip().split()[2:5])))
                    temp['valence'] = int(line.strip().split()[1])

                    self.ionData.append(temp)
                    
                    if atomNum in atomNumCount.keys():
                        atomNumCount[atomNum] += 1
                    else:
                        atomNumCount[atomNum] = 1

                    if atomName in atomType.keys():
                        pass
                    else:
                        atomType[atomName] = c
                        c+=1

                for dat in self.ionData:
                    name= dat['name']
                    dat['atomType'] = atomType[name]
                    dat['count'] = atomNumCount[pdt.elements.symbol(name).number]
                

                
                for key in atomNumCount.keys():
                    f.write("{} ".format(pdt.elements[key]))
                
                f.write("\n")
                
                for value in atomNumCount.values():
                    f.write("{} ".format(value))
                
                f.write("\n{}\n".format(self.ionPosVecType))
                
                for line in lines:
                    atomNum = int(line.strip().split()[0])
                    newLine = ' '.join(line.strip().split()[2:5])
                    newLine += ' {}\n'.format(pdt.elements[atomNum])
                    f.write(newLine)

        self.numIons = sum(atomNumCount.values())
        self.numTypes = len(atomNumCount.keys())

    def createKpts(self):
        self.kptW = []

        with open(self.filesPath + self.kptsFile) as f:
            for line in f:
                line = line.replace(',',' ')
                self.kptW.append(line.strip().split()[:4])

    def createProcar(self):
        with open(self.outdir + "PROCAR", "w") as f:
            f.write("This is a commented line\n") 
            with open(self.filesPath+ self.bandsDatFile) as f1:
                line = f1.readline()
                numKpts, self.numBandsPerKpt = list(map(int,line.strip().split()[:2]))
  
                f.write("# of k-points:  {}         # of bands:   {}         # of ions:    {}\n\n".format( numKpts,self.numBandsPerKpt,self.numIons))
                
                for line in f1:
                    l = list(map(float,line.strip().split()))
                    k, b, e, occ = int(l[0]), int(l[1]), l[2], l[3]
                    
                    if (b) % (self.numBandsPerKpt) == 0:
                        f.write (" k-point     {} :    {} {} {}     weight = {}\n\n".format(k+1, self.kptW[k][0], self.kptW[k][1], self.kptW[k][2], self.kptW[k][3]))
                    
                    f.write ("band     {} # energy   {} # occ.  {}\n\n".format(b+1, e * self.hartreeToEv, occ ))
                    f.write ("ion      s     py     pz     px    dxy    dyz    dz2    dxz  x2-y2    tot\n")
                    
                    for i in range(self.numIons):
                        f.write(str(i+1)+"    0 "*10 + "\n")  # for now all are taken as 0, later to be changed to actual values
                    f.write("tot {} \n\n".format("    0 "*10))

    def createOutcar(self):
        with open(self.outdir + "OUTCAR","w") as f:
            f.write(" E-fermi :   {}".format(self.eFermi*self.hartreeToEv)) # Only the Fermi energy part from OUTCAR is needed for bandstructure
        
    def createVasprun(self):
        with open(self.filesPath + self.pseudoPotFile) as f:
            for line in f:
                temp = str(pdt.elements[int(line.strip().split()[0])])
                for dat in self.ionData:
                    if dat['name']== temp:
                        dat['pseudo_pot'] = line.strip().split()[1]
        
        energies =[]
        dosVals = []
        with open(self.filesPath+self.dosDataFile) as f:
            for line in f:
                val = line.strip().split()[:2]
                energies.append(float(val[0])) # in the DFT-FE the printed value is already in eV 
                                            # The Fermi energy is also subtracted from it originally.
                dosVals.append(float(val[1]))
        dosIntegrated = []
        for i in range(len(energies)):
            temp = simps(dosVals[:i+1], x= energies[:i+1])
            dosIntegrated.append(temp)

        doc, tag, text = Doc().tagtext()

        with tag("modelling"):
            with tag ("generator"):
                pass
            with tag ("incar"):
                pass
            with tag ("primitive_cell"):
                pass
            # with tag ("kpoints"):
            #     with tag("generation"):
            #         pass
            #     with tag("varray", name = 'kpointlist'):
            #         pass
            #     with tag("varray", name = 'weights'):
            #         pass
            with tag ("parameters"):
                pass
            with tag ("atominfo"):
                with tag('atoms'):
                    text(self.numIons)
                with tag("types"):
                    text(self.numTypes)
                with tag("array", name = "atoms"):
                    with tag("dimension", dim = "1"):
                        text("ion")
                    with tag("field", type="string"):
                        text("element")
                    with tag("field", type="int"):
                        text("atomtype")
                    with tag("set"):
                        for dat in self.ionData:
                            with tag('rc'):
                                with tag('c'):
                                    text(dat['name'])
                                with tag('c'):
                                    text(dat['atomType'])
                with tag("array", name="atomtypes"):
                    with tag("dimension", dim="1"):
                        text("type")

                    with tag("field", type="int"):
                        text("atomspertype")
                    with tag("field", type="string"):
                        text("element")
                    with tag("field"):
                        text("mass")
                    with tag("field"):
                        text("valence")
                    with tag("field", type="string"):
                        text("pseudopotential")
                    
                    with tag("set"):
                        included =[]
                        for dat in self.ionData:
                            if dat['name'] not in included:
                                included.append(dat['name'])  
                                with tag('rc'):
                                    with tag('c'):
                                        text(dat['count'])
                                    with tag('c'):
                                        text(dat['name'])
                                    with tag('c'):
                                        text(pdt.elements.symbol(dat['name']).mass)
                                    with tag('c'):
                                        text(dat['valence'])
                                    with tag('c'):
                                        text(dat['pseudo_pot'])

            # with tag ("structure"):
            #     pass
            with tag ("calculation"):

                with tag ("structure"):
                    with tag('crystal'):
                        with tag("varray", name = 'basis'):
                            for val in self.latticeVecs:
                                with tag("v"):
                                    temp = '\t{}\t{}\t{}\t'.format(val[0], val[1], val[2])
                                    text(temp)
                    with tag("varray", name = "positions"):
                        for dat in self.ionData:
                            temp = dat['positions']
                            with tag("v"):
                                text("\t{}\t{}\t{}\t".format(temp[0], temp[1], temp[2]))

                with tag("dos"):
                    with tag('i', name="efermi"):
                        # text(self.eFermi*self.hartreeToEv)
                        text (0.0)
                    with tag('total'):
                        with tag('array'):
                            with tag("dimension", dim = "1"):
                                text("gridpoints")
                            with tag("dimension", dim = "2"):
                                text("spin")
                            with tag("field"):
                                text('energy')
                            with tag("field"):
                                text('total')
                            with tag("field"):
                                text('integrated')
                            with tag('set'):
                                with tag('set', comment = 'spin 1'):
                                    for i in range(len(energies)):
                                        with tag('r'):
                                            text("{}\t{}\t{}".format(energies[i], dosVals[i],dosIntegrated[i])) 
                                    

                
            
        result = indent(
            doc.getvalue(),
            indentation = ' '*4,
            newline = '\r\n'
        )

        with open(self.outdir+"vasprun.xml",'w') as f:
            f.write(result)


    def plotBandStr(self):

        self.createPoscar()
        self.createKpts()
        self.createProcar()
        self.createOutcar()

        splKticks =[]

        kticks = []
        knames = []
        with open(self.filesPath + self.kptsFile) as f:
            for lineNum,line in enumerate(f):
                if '#' in line:
                    kticks.append(lineNum)
                    knames.append(re.split('#', line)[-1])

                if '|' in re.split('#', line)[-1]:
                        splKticks.append(lineNum)
                        
        gph = pyprocar.bandsplot(
                        code='vasp',
                        mode='plain',
                        show = False,
                        elimit = self.eLimit,
                        dirname = self.outdir)
                        
        if len(splKticks) !=0: 
            for i in range(self.numBandsPerKpt):
                xdat = gph[1].get_lines()[i].get_xdata()
                for pt in splKticks:
                    xdat[pt+1] = xdat[pt]
                    try:
                        for j in range(pt+2, len(xdat)):
                            xdat[j] = xdat[j]-1 
                    except IndexError:
                        pass
                gph[1].get_lines()[i].set_xdata(xdat)
                
            for pt in splKticks:     
                for k in range(len(kticks)):
                    if kticks[k] > xdat[pt +1]:
                        kticks[k] = kticks[k] - 1
                

        if kticks and knames:
            gph[1].set_xticks(kticks, knames)
            for x in kticks:
                gph[1].axvline(x, color='k', linewidth = 0.01)  # Add a vertical line at xticks values

        gph[1].set_xlim(None, kticks[-1])   
                
        gph[1].yaxis.set_major_locator(MultipleLocator(1.0))
        gph[1].grid(True)

        gph[0].savefig(self.outdir+'bandsplot.png', dpi = 500)

    def plotDos(self):
        self.createPoscar()
        self.createKpts()
        self.createProcar()
        self.createOutcar()
        self.createVasprun()

        gph = pyprocar.dosplot(
                        code='vasp',
                        mode='plain',
                        show = False,
                        elimit = self.eLimit,
                        dirname = self.outdir)
        gph[1].grid(True)
        gph[1].set_xlabel('E - E$_f$ (eV)')
        gph[1].set_ylabel('DOS')
        gph[0].savefig(self.outdir+'dosplot.png', dpi = 500)
