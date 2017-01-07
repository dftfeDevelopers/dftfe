//Utility functions to read external files relevant to DFT
void readFile(unsigned int numColumns, std::vector<std::vector<double> > &data, std::string fileName){
  std::vector<double> rowData(numColumns, 0.0);
  std::ifstream readFile(fileName.c_str());
  if(readFile.fail()) {
    std::cerr<< "Error opening file: " << fileName.c_str() << std::endl;
    exit(-1);
  }

  //
  // String to store line and word
  //
  std::string readLine;
  std::string word;

  //
  // column index
  //
  int columnCount;

  if(readFile.is_open())
    {
      while (std::getline(readFile, readLine))
	{
	  std::istringstream iss(readLine);
        
	  columnCount = 0; 

	  while(iss >> word && columnCount < numColumns)
	    rowData[columnCount++] = atof(word.c_str());
     
	  data.push_back(rowData);
	}
    }
  readFile.close();
  return;
}
