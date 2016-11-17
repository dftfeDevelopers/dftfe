//Utility functions to read external files relevant to DFT
void readFile(unsigned int numColumns, std::vector<std::vector<double> > &data, std::string fileName){
  std::vector<double> rowData(numColumns, 0.0);
  std::ifstream readFile(fileName.c_str());
  if(readFile.fail()) {
    std::cerr<< "Error opening file: " << fileName.c_str() << std::endl;
    exit(-1);
  }
  if (readFile.is_open()) {
    while (!readFile.eof()) {
      for(unsigned int i = 0; i < numColumns; i++){
	readFile>>rowData[i];
      }
      data.push_back(rowData);
    }
  }
  readFile.close();
  return;
}
