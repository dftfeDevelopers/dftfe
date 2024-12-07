import os
import glob
import numpy as np
import periodictable as pdt
import sys

import pyprocar
from pyprocar.scripts import *
    
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator
import re
from typing import List, Tuple

import xml.etree.ElementTree as ET
from yattag import Doc, indent
from scipy.integrate import simpson

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import warnings
warnings.filterwarnings("ignore")

class Plotters:
    def __init__(self, filesPath = None, bandsDatFile = None, kptsFile = None, 
                 coordinatesFile = None, latticeVecFile = None,pseudoPotFile = None, 
                 dosDataFile= None, items = None, stack_orbitals = False, 
                 atoms:List[int] = None, stack_species = False, orbitals:List[int] = None, numSpins =1,
                 spins = None,overlay_mode = True, only_tdos = True, eLimit = None, dosLimit = None, isPeriodic= None, plot_total = None):
        
        
        self.filesPath = filesPath
        self.bandsDatFile = bandsDatFile
        self.kptsFile = kptsFile
        self.coordinatesFile = coordinatesFile
        self.latticeVecFile = latticeVecFile
        self.pseudoPotFile = pseudoPotFile
        self.dosDataFile = dosDataFile
        self.items = items
        self.stack_orbitals = stack_orbitals
        self.atoms = atoms
        self.stack_species = stack_species
        self.orbitals = orbitals
        self.numSpins = numSpins
        self.spins = spins
        self.overlay_mode = overlay_mode
        self.only_tdos = only_tdos
        self.eLimit = eLimit
        self.dosLimit = dosLimit
        self.isPeriodic = isPeriodic
        self.plot_total = plot_total
        self.outdir = filesPath + "pyprocar_outputs/"
        self.hartreeToEv = 27.211386024367243

        temp_items = dict()

        # Can't  plot 2 orbitals of the same atom simultaneusly
        if (self.items != None):
            for key, val in self.items.items():
                temp_items[key] = []
                for orb in val:
                    if orb == 's':
                        temp_items[key] += [0]
                    elif orb == 'p':
                        temp_items[key] += [1,2,3]
                    elif orb == 'd':
                        temp_items[key] += [4,5,6,7,8]

            self.items = temp_items
        '''
        filesPath: str: path to the directory where the input files are stored,
        bandsDatFile: str: name of the file containing the bandstructure data,
        kptsFile: str: name of the file containing the kpoints data,
        coordinatesFile: str: name of the file containing the coordinates data,
        latticeVecFile: str: name of the file containing the lattice vectors data,
        pseudoPotFile: str: name of the file containing the pseudopotential data,
        dosDataFile: str: name of the file containing the DOS data,
        eLimit: List: energy limits for the bandstructure, DOS and PDOS plots,
        dosLimit: List: DOS limits for the DOS and PDOS plots,
        isPeriodic: bool: True if the system is periodic or semi periodic, False otherwise,
        only_tdos: bool: If True, only the total DOS is plotted,
        plot_total: bool: If True, then the total DOS is plotted along with the PDOS,
        spins: List: List of spin-channels to be included in the plot, 
        items : dict: keys are the specific species and values being the corresponding orbitals (available options 's','p','d') for each species.
        
        
        *************************
        for better understanding of the below parameters please refer to the pyprocar user guide
        "https://romerogroup.github.io/pyprocar/examples/index.html"
        ************************
        
        overlay_mode: bool: If True, plot is in 'overlay' mode, otherwise 'stack' mode
        stack_orbitals: bool: If True, PDOS corresponding to the orbitals of the atoms mentioned in the "atoms" variable are plotted
        atoms: List: List of the atoms to be considered for PDOS calculation. It has meaning only if "stack_orbitals" is true. If nothing specified, all the atoms are considered
        stack_species: bool: If True, PDOS corresponding to the atoms for the orbitals mentioned in the "orbitals" variable are plotted
        orbitals: List: List of the orbitals to be considered for PDOS calculation. It has meaning only if "stack_species" is true. If nothing specified, all the orbitals are considered
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
        with open(self.filesPath+ self.bandsDatFile) as f1:
            line = f1.readline()
            self.numKpts, self.numBandsPerKpt = list(map(int,line.strip().split()[:2]))

    def createPoscar(self):
        with open(self.outdir+'POSCAR', 'w') as f:
            f.write("This is a commented line\n") # Here you can write the name of the system (it's a commented line in POSCAR file)
            self.latticeVecs =[] # in Angstrom

            with open(self.filesPath+ self.latticeVecFile) as f1:
                f.write("{}\n".format(1.0)) 
                lines = f1.readlines()
                for line in lines:
                    if len(line.strip()) != 0:
                        vecVal = (np.array(list(map(float,line.strip().split()))))*0.5291772105638411
                        self.latticeVecs.append(vecVal)
                        f.write("{}\t{}\t{}\n".format(vecVal[0], vecVal[1], vecVal[2]))
            # list of length = number of atoms. contains dictionary whose kyes are name, positions, valence, and pseudopotential file name 
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
                    if self.isPeriodic:
                        temp['positions'] = np.array(list(map(float,line.strip().split()[2:5])))
                    else:
                        CoordsVal =  np.array(list(map(float,line.strip().split()[2:5])))* 0.5291772105638411+ np.sum(self.latticeVecs,axis=0)/2
                        temp['positions'] = CoordsVal

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
                
                for i,line in enumerate(lines):
                    atomNum = int(line.strip().split()[0])
                    dat = self.ionData[i]['positions']
                    newLine = '{}\t{}\t{}\t{}\n'.format(dat[0], dat[1], dat[2], pdt.elements[atomNum])
                    f.write(newLine)

        self.numIons = sum(atomNumCount.values())
        self.numTypes = len(atomNumCount.keys())

    def createKpts(self, forDOS = False):
        self.kptW = []
        if forDOS:
            for _ in range(self.numKpts):
                self.kptW.append([0,0,0,0])
        else:
            with open(self.filesPath + self.kptsFile) as f:
                for line in f:
                    line = line.replace(',',' ')
                    self.kptW.append(line.strip().split()[:4])

    def createProcar(self):
        with open(self.outdir + "PROCAR", "w") as f:
            f.write("This is a commented line\n") 
            with open(self.filesPath+ self.bandsDatFile) as f1:
                line = f1.readline()
                f.write("# of k-points:  {}         # of bands:   {}         # of ions:    {}\n\n".format( self.numKpts,self.numBandsPerKpt,self.numIons))
                occupationIdx = 3
                if self.numSpins == 2:
                    occupationIdx = 4
                for line in f1:
                    l = list(map(float,line.strip().split()))
                    k, b, e, occ = int(l[0]), int(l[1]), l[2], l[occupationIdx]
                    
                    if (b) % (self.numBandsPerKpt) == 0:
                        f.write (" k-point     {} :    {} {} {}     weight = {}\n\n".format(k+1, self.kptW[k][0], self.kptW[k][1], self.kptW[k][2], self.kptW[k][3]))
                    
                    f.write ("band     {} # energy   {} # occ.  {}\n\n".format(b+1, e * self.hartreeToEv, occ ))
                    f.write ("ion      s     py     pz     px    dxy    dyz    dz2    dxz  x2-y2    tot\n")
                    
                    for i in range(self.numIons):
                        f.write(str(i+1)+"    0 "*10 + "\n")  # for now all are taken as 0, later to be changed to actual values
                    f.write("tot {} \n\n".format("    0 "*10))
            
            if self.numSpins == 2:
                with open(self.filesPath+ self.bandsDatFile) as f1:
                    line = f1.readline()
                    f.write("# of k-points:  {}         # of bands:   {}         # of ions:    {}\n\n".format( self.numKpts,self.numBandsPerKpt,self.numIons))
                    
                    for line in f1:
                        l = list(map(float,line.strip().split()))
                        k, b, e, occ = int(l[0]), int(l[1]), l[3], l[5]
                        
                        if (b) % (self.numBandsPerKpt) == 0:
                            f.write (" k-point     {} :    {} {} {}     weight = {}\n\n".format(k+1, self.kptW[k][0], self.kptW[k][1], self.kptW[k][2], self.kptW[k][3]))
                        
                        f.write ("band     {} # energy   {} # occ.  {}\n\n".format(b+1, e * self.hartreeToEv, occ ))
                        f.write ("ion      s     py     pz     px    dxy    dyz    dz2    dxz  x2-y2    tot\n")
                        
                        for i in range(self.numIons):
                            f.write(str(i+1)+"    0 "*10 + "\n")  # for now all are taken as 0, later to be changed to actual values
                        f.write("tot {} \n\n".format("    0 "*10))


    def createOutcar(self):
        with open(self.outdir + "OUTCAR","w") as f:
            f.write("E-fermi :   {}".format(self.eFermi*self.hartreeToEv)) # Only the Fermi energy part from OUTCAR is needed for bandstructure
        
    def createVasprun(self):
        with open(self.filesPath + self.pseudoPotFile) as f:
            for line in f:
                temp = str(pdt.elements[int(line.strip().split()[0])])
                for dat in self.ionData:
                    if dat['name']== temp:
                        dat['pseudo_pot'] = line.strip().split()[1]

        pdosValsDict= dict()
        gotPdosEnergies = False
        for atomIdx in range(self.numIons):
            pdosValsDict[atomIdx] = dict()
            for spinIdx in range(self.numSpins):
                pdosValsDict[atomIdx][spinIdx] = dict()

            pdosFileNamePattern = os.path.join(self.filesPath, 'pdosData_atom#{}*'.format(atomIdx))
            matching_files = glob.glob(pdosFileNamePattern)

            for file_path in matching_files:
                orbitalPattern = r'\((.*?)\)'
                match = re.search(orbitalPattern, file_path)
                if match:
                    orbitalName = (match.group(1))[1]
                else:
                    print('No match found')
                    sys.exit()

                data = np.loadtxt(file_path)

                if gotPdosEnergies == False:
                    pdosEnergies = data[:,0]
                    gotPdosEnergies = True

                if self.numSpins == 1:
                    if orbitalName == 's':
                        if 's' in pdosValsDict[atomIdx][0].keys():
                            pdosValsDict[atomIdx][0]['s'] += data[:,2] # All the 's' orbitals are added
                        else:
                            pdosValsDict[atomIdx][0]['s'] = data[:,2]

                    elif orbitalName == 'p':
                        if "px" in pdosValsDict[atomIdx][0].keys():
                            pdosValsDict[atomIdx][0]['py'] += data[:,2]
                            pdosValsDict[atomIdx][0]['pz'] += data[:,3]
                            pdosValsDict[atomIdx][0]['px'] += data[:,4]
                        else:
                            pdosValsDict[atomIdx][0]['py'] = data[:,2]
                            pdosValsDict[atomIdx][0]['pz'] = data[:,3]
                            pdosValsDict[atomIdx][0]['px'] = data[:,4]

                    elif orbitalName == 'd':
                        if "dxy" in pdosValsDict[atomIdx][0].keys():
                            pdosValsDict[atomIdx][0]['dxy'] += data[:,2]
                            pdosValsDict[atomIdx][0]['dyz'] += data[:,3]
                            pdosValsDict[atomIdx][0]['dz2'] += data[:,4]
                            pdosValsDict[atomIdx][0]['dxz'] += data[:,5]
                            pdosValsDict[atomIdx][0]['dx2-y2'] += data[:,6]
                        else:
                            pdosValsDict[atomIdx][0]['dxy'] = data[:,2]
                            pdosValsDict[atomIdx][0]['dyz'] = data[:,3]
                            pdosValsDict[atomIdx][0]['dz2'] = data[:,4]
                            pdosValsDict[atomIdx][0]['dxz'] = data[:,5]
                            pdosValsDict[atomIdx][0]['dx2-y2'] = data[:,6]

                elif self.numSpins == 2:
                    if orbitalName == 's':
                        if 's' in pdosValsDict[atomIdx][0].keys():
                            pdosValsDict[atomIdx][0]['s'] += data[:,3]
                            pdosValsDict[atomIdx][1]['s'] += data[:,4]
                        else:
                            pdosValsDict[atomIdx][0]['s'] = data[:,3]
                            pdosValsDict[atomIdx][1]['s'] = data[:,4]
                    elif orbitalName == 'p':
                        if "px" in pdosValsDict[atomIdx][0].keys():
                            pdosValsDict[atomIdx][0]['py'] += data[:,3]
                            pdosValsDict[atomIdx][1]['py'] += data[:,4]
                            pdosValsDict[atomIdx][0]['pz'] += data[:,5]
                            pdosValsDict[atomIdx][1]['pz'] += data[:,6]
                            pdosValsDict[atomIdx][0]['px'] += data[:,7]
                            pdosValsDict[atomIdx][1]['px'] += data[:,8]
                        else:
                            pdosValsDict[atomIdx][0]['py'] = data[:,3]
                            pdosValsDict[atomIdx][1]['py'] = data[:,4]
                            pdosValsDict[atomIdx][0]['pz'] = data[:,5]
                            pdosValsDict[atomIdx][1]['pz'] = data[:,6]
                            pdosValsDict[atomIdx][0]['px'] = data[:,7]
                            pdosValsDict[atomIdx][1]['px'] = data[:,8]

                    elif orbitalName == 'd':
                        if "dxy" in pdosValsDict[atomIdx][0].keys():
                            pdosValsDict[atomIdx][0]['dxy'] += data[:,3]
                            pdosValsDict[atomIdx][1]['dxy'] += data[:,4]
                            pdosValsDict[atomIdx][0]['dyz'] += data[:,5]
                            pdosValsDict[atomIdx][1]['dyz'] += data[:,6]
                            pdosValsDict[atomIdx][0]['dz2'] += data[:,7]
                            pdosValsDict[atomIdx][1]['dz2'] += data[:,8]
                            pdosValsDict[atomIdx][0]['dxz'] += data[:,9]
                            pdosValsDict[atomIdx][1]['dxz'] += data[:,10]
                            pdosValsDict[atomIdx][0]['dx2-y2'] += data[:,11]
                            pdosValsDict[atomIdx][1]['dx2-y2'] += data[:,12]
                        else:
                            pdosValsDict[atomIdx][0]['dxy'] = data[:,3]
                            pdosValsDict[atomIdx][1]['dxy'] = data[:,4]
                            pdosValsDict[atomIdx][0]['dyz'] = data[:,5]
                            pdosValsDict[atomIdx][1]['dyz'] = data[:,6]
                            pdosValsDict[atomIdx][0]['dz2'] = data[:,7]
                            pdosValsDict[atomIdx][1]['dz2'] = data[:,8]
                            pdosValsDict[atomIdx][0]['dxz'] = data[:,9]
                            pdosValsDict[atomIdx][1]['dxz'] = data[:,10]
                            pdosValsDict[atomIdx][0]['dx2-y2'] = data[:,11]
                            pdosValsDict[atomIdx][1]['dx2-y2'] = data[:,12]
        
        dosData = np.loadtxt(self.filesPath+self.dosDataFile)

        energies = dosData[:,0]
        dosVals = dosData[:,1]
        dosIntegrated = []
        if self.numSpins == 2:
            dosValsDown = dosData[:,2]
            dosIntegratedDown = []

        for i in range(len(energies)):
            temp = simpson(dosVals[:i+1], x= energies[:i+1])
            dosIntegrated.append(temp)
            if self.numSpins == 2:
                temp = simpson(dosValsDown[:i+1], x= energies[:i+1])
                dosIntegratedDown.append(temp)

        doc, tag, text = Doc().tagtext()

        with tag("modeling"):
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
                        text(self.eFermi*self.hartreeToEv)
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
                                if self.numSpins == 2:
                                    with tag('set', comment = 'spin 2'):
                                        for i in range(len(energies)):
                                            with tag('r'):
                                                text("{}\t{}\t{}".format(energies[i], dosValsDown[i],dosIntegratedDown[i]))

                    with tag('partial'):
                        with tag('array'):
                            with tag("dimension", dim="1"):
                                text('gridpoints')
                            with tag("dimension", dim="2"):
                                text('spin')
                            with tag("dimension", dim="3"):
                                text('ion')
                            with tag("field"):
                                text("energy")
                            with tag("field"):
                                text("s")
                            with tag("field"):
                                text("py")
                            with tag("field"):
                                text("pz")
                            with tag("field"):
                                text("px")
                            with tag("field"):
                                text("dxy")
                            with tag("field"):
                                text("dyz")
                            with tag("field"):
                                text("dz2")
                            with tag("field"):
                                text("dxz")
                            with tag("field"):
                                text("x2-y2")
                            with tag("set"):
                                for atomIdx in range(self.numIons):
                                    with tag("set", comment="ion {}".format(atomIdx+1)):
                                        with tag("set", comment="spin 1"):  # to be implemented for spin
                                            for energyIdx in range(len(pdosEnergies)):
                                                val = [0 for i in range(10)]
                                                val[0] = pdosEnergies[energyIdx]
                                                for key,dVal in pdosValsDict[atomIdx][0].items():
                                                    temp = dVal[energyIdx]
                                                    if key == 's':
                                                        val[1] = temp
                                                    elif key == 'py':
                                                        val[2] = temp
                                                    elif key == 'pz':
                                                        val[3] = temp
                                                    elif key == 'px':
                                                        val[4] = temp
                                                    elif key == 'dxy':
                                                        val[5] = temp
                                                    elif key == 'dyz':
                                                        val[6] = temp
                                                    elif key == 'dz2':
                                                        val[7] = temp
                                                    elif key == 'dxz':
                                                        val[8] = temp
                                                    elif key == 'dx2-y2':
                                                        val[9] = temp
                                                
                                                with tag('r'):
                                                    text("\t"+"\t".join(list(map(str,val))))
                                        if self.numSpins == 2:
                                            with tag("set", comment="spin 2"):  # to be implemented for spin
                                                for energyIdx in range(len(pdosEnergies)):
                                                    val = [0 for i in range(10)]
                                                    val[0] = pdosEnergies[energyIdx]
                                                    for key,dVal in pdosValsDict[atomIdx][1].items():
                                                        temp = dVal[energyIdx]
                                                        if key == 's':
                                                            val[1] = temp
                                                        elif key == 'py':
                                                            val[2] = temp
                                                        elif key == 'pz':
                                                            val[3] = temp
                                                        elif key == 'px':
                                                            val[4] = temp
                                                        elif key == 'dxy':
                                                            val[5] = temp
                                                        elif key == 'dyz':
                                                            val[6] = temp
                                                        elif key == 'dz2':
                                                            val[7] = temp
                                                        elif key == 'dxz':
                                                            val[8] = temp
                                                        elif key == 'dx2-y2':
                                                            val[9] = temp
                                                    
                                                    with tag('r'):
                                                        text("\t"+"\t".join(list(map(str,val))))
            
        result = indent(
            doc.getvalue(),
            indentation = ' '*2,
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
                        spins = self.spins,
                        fermi = self.eFermi*self.hartreeToEv,
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
        gph[1].yaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        
        gph[1].grid(True)
        gph[0].savefig(self.outdir+'bandsplot.png', dpi = 500)

    def plotDos(self):
        self.createPoscar()
        self.createKpts(forDOS=True)
        self.createProcar()
        self.createOutcar()
        self.createVasprun()

        if ((self.stack_orbitals == True or self.stack_species == True or self.items != None) and self.only_tdos == False):
            if self.stack_orbitals:
                print('plotting stack_orbitals')

                if self.overlay_mode:
                    mode = 'overlay_orbitals'
                else:
                    mode = 'stack_orbitals'

                if self.atoms == None:
                    self.atoms = list(np.arange(self.numIons))
                gph = pyprocar.dosplot(
                            code='vasp',
                            mode= mode,
                            spins = self.spins,
                            atoms = self.atoms,
                            fermi = self.eFermi*self.hartreeToEv,
                            show = False,
                            elimit = self.eLimit,
                            dirname = self.outdir,
                            plot_total = self.plot_total)
                
            elif self.stack_species:

                print('plotting stack_species')

                if self.overlay_mode:
                    mode = 'overlay_species'
                else:
                    mode = 'stack_species'
                if self.orbitals == None:
                    self.orbitals = list(np.arange(9))
                gph = pyprocar.dosplot(
                            code='vasp',
                            mode= mode,
                            spins = self.spins,
                            orbitals = self.orbitals,
                            fermi = self.eFermi*self.hartreeToEv,
                            show = False,
                            elimit = self.eLimit,
                            dirname = self.outdir,
                            plot_total = self.plot_total)
            else:
                print('plotting items')
                gph = pyprocar.dosplot(
                        code='vasp',
                        mode='overlay',
                        items = self.items,
                        spins = self.spins,
                        fermi = self.eFermi*self.hartreeToEv,
                        show = False,
                        elimit = self.eLimit,
                        dirname = self.outdir,
                        plot_total = self.plot_total)
                        # savefig='dos_plain_ver.png')
        else:
            print("plotting total")
            gph = pyprocar.dosplot(
                            code='vasp',
                            mode='plain',
                            spins = self.spins,
                            fermi = self.eFermi*self.hartreeToEv,
                            show = False,
                            elimit = self.eLimit,
                            dirname = self.outdir)
            
        gph[1].grid(True)
        gph[1].set_xlabel('E - E$_f$ (eV)')
        gph[1].set_ylabel('DOS')
        gph[1].set_ylim(self.dosLimit)
        for line in gph[1].get_lines():
            line.set_linewidth(0.7)
        gph[0].savefig(self.outdir+'dosplot.png', dpi = 500)