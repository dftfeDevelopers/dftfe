//Utility functions to read external files relevant to DFT

//Function to initial guess of rho
void readFile(std::vector<std::vector<double> > &data){
  unsigned int numColumns=2; //change this for being generic
  std::vector<double> rowData(numColumns, 0.0);
  std::ifstream readFile;
  readFile.open(rhoFileName);
  if (readFile.is_open()) {
    while (!readFile.eof()) {
      for(unsigned int i=0; i <numColumns; i++)
	readFile>>rowData[i];
      data.push_back(rowData);
    }
  }
  readFile.close();
  return;
}
