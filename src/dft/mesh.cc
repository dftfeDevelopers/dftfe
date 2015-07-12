//source file for all mesh reading/generation functions

//Generate triangulation.
void dftClass::mesh(){
  computing_timer.enter_section("mesh"); 
  //GridGenerator::hyper_cube (triangulation, -1, 1);
  //triangulation.refine_global (6);
  //Read mesh written out in UCD format
  //static const Point<3> center = Point<3>();
  //static const HyperBallBoundary<3, 3> boundary(center,radius);
  //static const StraightBoundary<3, 3> boundary();
  GridIn<3> gridin;
  gridin.attach_triangulation(triangulation);
  //Read mesh in UCD format generated from Cubit
  std::ifstream f(meshFileName);
  gridin.read_ucd(f);
  //triangulation.set_boundary(0, boundaryClass);
  //triangulation.set_boundary(0, StraightBoundary<3>());
  triangulation.refine_global (n_refinement_steps);
  computing_timer.exit_section("mesh"); 
}
