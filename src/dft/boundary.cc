//source file for implementation of boundary conditions

//boundary condition function
template <int dim>
class OnebyRBoundaryFunction : public Function<dim>{
public:
  OnebyRBoundaryFunction(): Function<dim>(){}
  virtual double value (const Point<dim> &p, const unsigned int component = 0) const{
    return -atomCharge/sqrt(p.square());
  }
};
