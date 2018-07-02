// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE authors.
//
// This file is part of the DFT-FE code.
//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the DFT-FE distribution.
//
// ---------------------------------------------------------------------
//
// @author  Phani Motamarri 
//

#include "PeriodicTable.h"
#include <cassert>

namespace dftfe {

  namespace pseudoUtils
  {

    ////////////////////////////////////////////////////////////////////////////////
    int PeriodicTable::z(std::string symbol) const
    {
      std::map<std::string,int>::const_iterator i = zmap.find(symbol);
      assert( i != zmap.end() );
      return (*i).second;
    }

    ////////////////////////////////////////////////////////////////////////////////
    std::string PeriodicTable::symbol(int z) const
    {
      assert(z>0 && z<=ptable.size());
      return ptable[z-1].symbol;
    }

    ////////////////////////////////////////////////////////////////////////////////
    std::string PeriodicTable::configuration(int z) const
    {
      assert(z>0 && z<=ptable.size());
      return ptable[z-1].config;
    }

    ////////////////////////////////////////////////////////////////////////////////
    std::string PeriodicTable::configuration(std::string symbol) const
    {
      return ptable[z(symbol)-1].config;
    }

    ////////////////////////////////////////////////////////////////////////////////
    double PeriodicTable::mass(int z) const
    {
      assert(z>0 && z<=ptable.size());
      return ptable[z-1].mass;
    }

    ////////////////////////////////////////////////////////////////////////////////
    double PeriodicTable::mass(std::string symbol) const
    {
      return ptable[z(symbol)-1].mass;
    }

    ////////////////////////////////////////////////////////////////////////////////
    int PeriodicTable::size(void) const
    {
      return ptable.size();
    }

    ////////////////////////////////////////////////////////////////////////////////
    PeriodicTable::PeriodicTable(void)
    {
      ptable.push_back(Element(1,"H","1s1",1.00794));
      ptable.push_back(Element(2,"He","1s2",4.00260));
      ptable.push_back(Element(3, "Li","1s2 2s1",     6.941));
      ptable.push_back(Element(4, "Be","1s2 2s2",     9.01218));
      ptable.push_back(Element(5, "B", "1s2 2s2 2p1",10.811));
      ptable.push_back(Element(6, "C", "1s2 2s2 2p2",12.0107));
      ptable.push_back(Element(7, "N", "1s2 2s2 2p3",14.00674));
      ptable.push_back(Element(8, "O", "1s2 2s2 2p4",15.9994));
      ptable.push_back(Element(9, "F", "1s2 2s2 2p5",18.9884));
      ptable.push_back(Element(10,"Ne","1s2 2s2 2p6",20.1797));

      ptable.push_back(Element(11,"Na","[Ne] 3s1",    22.98977));
      ptable.push_back(Element(12,"Mg","[Ne] 3s2",    24.3050));
      ptable.push_back(Element(13,"Al","[Ne] 3s2 3p1",26.98154));
      ptable.push_back(Element(14,"Si","[Ne] 3s2 3p2",28.0855));
      ptable.push_back(Element(15,"P", "[Ne] 3s2 3p3",30.97376));
      ptable.push_back(Element(16,"S", "[Ne] 3s2 3p4",32.066));
      ptable.push_back(Element(17,"Cl","[Ne] 3s2 3p5",35.4527));
      ptable.push_back(Element(18,"Ar","[Ne] 3s2 3p6",39.948));

      ptable.push_back(Element(19,"K", "[Ar] 4s1",39.0983));
      ptable.push_back(Element(20,"Ca","[Ar] 4s2",40.078));
      ptable.push_back(Element(21,"Sc","[Ar] 3d1 4s2",44.95591));
      ptable.push_back(Element(22,"Ti","[Ar] 3d2 4s2",47.867));
      ptable.push_back(Element(23,"V", "[Ar] 3d3 4s2",50.9415));
      ptable.push_back(Element(24,"Cr","[Ar] 3d5 4s1",51.9961));
      ptable.push_back(Element(25,"Mn","[Ar] 3d5 4s2",54.93805));
      ptable.push_back(Element(26,"Fe","[Ar] 3d6 4s2",55.845));
      ptable.push_back(Element(27,"Co","[Ar] 3d7 4s2",58.9332));
      ptable.push_back(Element(28,"Ni","[Ar] 3d8 4s2",58.6934));
      ptable.push_back(Element(29,"Cu","[Ar] 3d10 4s1",63.546));
      ptable.push_back(Element(30,"Zn","[Ar] 3d10 4s2",65.39));
      ptable.push_back(Element(31,"Ga","[Ar] 3d10 4s2 4p1",69.723));
      ptable.push_back(Element(32,"Ge","[Ar] 3d10 4s2 4p2",72.61));
      ptable.push_back(Element(33,"As","[Ar] 3d10 4s2 4p3",74.9216));
      ptable.push_back(Element(34,"Se","[Ar] 3d10 4s2 4p4",78.96));
      ptable.push_back(Element(35,"Br","[Ar] 3d10 4s2 4p5",79.904));
      ptable.push_back(Element(36,"Kr","[Ar] 3d10 4s2 4p6",83.80));

      ptable.push_back(Element(37,"Rb","[Kr] 5s1",85.4678));
      ptable.push_back(Element(38,"Sr","[Kr] 5s2",87.62));
      ptable.push_back(Element(39,"Y" ,"[Kr] 4d1 5s2",88.90585));
      ptable.push_back(Element(40,"Zr","[Kr] 4d2 5s2",91.224));
      ptable.push_back(Element(41,"Nb","[Kr] 4d4 5s1",92.90638));
      ptable.push_back(Element(42,"Mo","[Kr] 4d5 5s1",95.94));
      ptable.push_back(Element(43,"Tc","[Kr] 4d5 5s2",98.0));
      ptable.push_back(Element(44,"Ru","[Kr] 4d7 5s1",101.07));
      ptable.push_back(Element(45,"Rh","[Kr] 4d8 5s1",102.9055));
      ptable.push_back(Element(46,"Pd","[Kr] 4d10",106.42));
      ptable.push_back(Element(47,"Ag","[Kr] 4d10 5s1",107.8682));
      ptable.push_back(Element(48,"Cd","[Kr] 4d10 5s2",112.411));
      ptable.push_back(Element(49,"In","[Kr] 4d10 5s2 5p1",114.818));
      ptable.push_back(Element(50,"Sn","[Kr] 4d10 5s2 5p2",118.710));
      ptable.push_back(Element(51,"Sb","[Kr] 4d10 5s2 5p3",121.760));
      ptable.push_back(Element(52,"Te","[Kr] 4d10 5s2 5p4",127.60));
      ptable.push_back(Element(53,"I" ,"[Kr] 4d10 5s2 5p5",126.90447));
      ptable.push_back(Element(54,"Xe","[Kr] 4d10 5s2 5p6",131.29));

      ptable.push_back(Element(55,"Cs","[Xe] 6s1",132.90545));
      ptable.push_back(Element(56,"Ba","[Xe] 6s2",137.327));
      ptable.push_back(Element(57,"La","[Xe] 5d1 6s2",138.9055));
      ptable.push_back(Element(58,"Ce","[Xe] 4f1 5d1 6s2",140.116));
      ptable.push_back(Element(59,"Pr","[Xe] 4f3 6s2",140.90765));
      ptable.push_back(Element(60,"Nd","[Xe] 4f4 6s2",144.24));
      ptable.push_back(Element(61,"Pm","[Xe] 4f5 6s2",145.0));
      ptable.push_back(Element(62,"Sm","[Xe] 4f6 6s2",150.36));
      ptable.push_back(Element(63,"Eu","[Xe] 4f7 6s2",151.964));
      ptable.push_back(Element(64,"Gd","[Xe] 4f7 5d1 6s2",157.25));
      ptable.push_back(Element(65,"Tb","[Xe] 4f9 6s2",158.92534));
      ptable.push_back(Element(66,"Dy","[Xe] 4f10 6s2",162.50));
      ptable.push_back(Element(67,"Ho","[Xe] 4f11 6s2",164.93032));
      ptable.push_back(Element(68,"Er","[Xe] 4f12 6s2",167.26));
      ptable.push_back(Element(69,"Tm","[Xe] 4f13 6s2",168.93421));
      ptable.push_back(Element(70,"Yb","[Xe] 4f14 6s2",173.04));
      ptable.push_back(Element(71,"Lu","[Xe] 4f14 5d1 6s2",174.967));
      ptable.push_back(Element(72,"Hf","[Xe] 4f14 5d2 6s2",178.49));
      ptable.push_back(Element(73,"Ta","[Xe] 4f14 5d3 6s2",180.9479));
      ptable.push_back(Element(74,"W" ,"[Xe] 4f14 5d4 6s2",183.84));
      ptable.push_back(Element(75,"Re","[Xe] 4f14 5d5 6s2",186.207));
      ptable.push_back(Element(76,"Os","[Xe] 4f14 5d6 6s2",190.23));
      ptable.push_back(Element(77,"Ir","[Xe] 4f14 5d7 6s2",192.217));
      ptable.push_back(Element(78,"Pt","[Xe] 4f14 5d9 6s1",195.078));
      ptable.push_back(Element(79,"Au","[Xe] 4f14 5d10 6s1",196.96655));
      ptable.push_back(Element(80,"Hg","[Xe] 4f14 5d10 6s2",200.59));
      ptable.push_back(Element(81,"Tl","[Xe] 4f14 5d10 6s2 6p1",204.3833));
      ptable.push_back(Element(82,"Pb","[Xe] 4f14 5d10 6s2 6p2",207.2));
      ptable.push_back(Element(83,"Bi","[Xe] 4f14 5d10 6s2 6p3",208.98038));
      ptable.push_back(Element(84,"Po","[Xe] 4f14 5d10 6s2 6p4",209.0));
      ptable.push_back(Element(85,"At","[Xe] 4f14 5d10 6s2 6p5",210.0));
      ptable.push_back(Element(86,"Rn","[Xe] 4f14 5d10 6s2 6p6",222.0));

      ptable.push_back(Element(87,"Fr","[Rn] 7s1",223.0));
      ptable.push_back(Element(88,"Ra","[Rn] 7s2",226.0));
      ptable.push_back(Element(89,"Ac","[Rn] 6d1 7s2",227.0));
      ptable.push_back(Element(90,"Th","[Rn] 6d2 7s2",232.0381));
      ptable.push_back(Element(91,"Pa","[Rn] 5f2 6d1 7s2",231.03588));
      ptable.push_back(Element(92,"U" ,"[Rn] 5f3 6d1 7s2",238.0289));
      ptable.push_back(Element(93,"Np","[Rn] 5f4 6d1 7s2",237.0));
      ptable.push_back(Element(94,"Pu","[Rn] 5f5 6d1 7s2",244.0));

      for ( int i = 0; i < ptable.size(); i++ )
	zmap[ptable[i].symbol] = i+1;
    }

  }
}
