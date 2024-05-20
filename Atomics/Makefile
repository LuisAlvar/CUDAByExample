compiler = g++

solutionName = CPUHistogram

CPUHistogram: CPUHistogram.o
	$(compiler) -o $(solutionName) CPUHistogram.o

CPUHistogram.o: CPUHistogram.cpp
	$(compiler) -c CPUHistogram.cpp

clean:
	rm *.o $(solutionName)
