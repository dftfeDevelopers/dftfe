//source file for implementation of boundary conditions

//boundary condition function
template <int dim>
class OnebyRBoundaryFunction : public Function<dim>{
public:
  Table<2,double>* atomLocationsPtr;
  OnebyRBoundaryFunction(Table<2,double>& atomLocations): Function<dim>(), atomLocationsPtr(&atomLocations){}
  virtual double value (const Point<dim> &p, const unsigned int component = 0) const{
    double value=0;
    //loop for multiple atoms
    for (unsigned int z=0; z<atomLocationsPtr->size()[0]; z++){
      Point<3> atom((*atomLocationsPtr)[z][1],(*atomLocationsPtr)[z][2],(*atomLocationsPtr)[z][3]);
      value+= -(*atomLocationsPtr)[z][0]/p.distance(atom);
    }
    return value;
  }
};
